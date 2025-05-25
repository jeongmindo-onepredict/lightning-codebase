# src/models/fgvc_pmal.py - Complete PMAL & PMD Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Accuracy
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from typing import Optional, Dict, Any, Tuple, List
import timm
import copy
from collections import OrderedDict

try:
    import wandb
except ImportError:
    wandb = None

# ============================================================================
# SAM (Sharpness-Aware Minimization) Implementation
# ============================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    dtype=torch.float32
                )
        return norm.to(shared_device)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# ============================================================================
# Denoising-Recognition Head (DRH) Implementation
# ============================================================================
class DenoisingRecognitionHead(nn.Module):
    """
    Complete DRH implementation following the paper's Figure 2.
    Includes both Recognition Sub-head and Denoising Sub-head.
    """
    def __init__(self, input_channels: int, num_classes: int, input_size: Tuple[int, int], 
                 target_size: Tuple[int, int] = (448, 448), D: int = 1024, 
                 D_prime: int = 256, D_double_prime: int = 64):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size  # (H, W)
        self.target_size = target_size  # (H', W')
        self.D = D
        
        # Recognition Sub-head: Benc + Brec
        self.recognition_encoder = nn.Sequential(
            # Benc: Two conv layers with BN and ReLU
            nn.Conv2d(input_channels, D//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(D//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(D//2, D, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            # Global Maximum Pooling
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        
        self.recognition_classifier = nn.Sequential(
            # Brec: BN + FC + BN + ELU + FC (classifier)
            nn.BatchNorm1d(D),
            nn.Linear(D, D//2),
            nn.BatchNorm1d(D//2),
            nn.ELU(inplace=True),
            nn.Linear(D//2, num_classes)
        )
        
        # Denoising Sub-head: Restoration Block (Bres)
        self.restoration_block = self._build_restoration_block(input_channels, D_prime)
        
        # Skip Block (Bski)
        self.skip_block = nn.Sequential(
            nn.Conv2d(3, D_double_prime, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(D_double_prime, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def _build_restoration_block(self, input_channels: int, D_prime: int) -> nn.Module:
        """
        Build restoration block with 4 upsampling modules (M1_up to M4_up)
        Following the paper's architecture with PixelShuffle layers.
        """
        H, W = self.input_size
        H_prime, W_prime = self.target_size
        
        # Calculate upsampling scale for M1_up
        scale_h = H_prime // (8 * H) if H_prime // (8 * H) >= 1 else 1
        scale_w = W_prime // (8 * W) if W_prime // (8 * W) >= 1 else 1
        
        modules = []
        current_channels = input_channels
        
        # M1_up: Initial upsampling
        if scale_h > 1 or scale_w > 1:
            modules.extend([
                nn.PixelShuffle(max(scale_h, scale_w)),
                nn.Conv2d(current_channels // (max(scale_h, scale_w) ** 2), D_prime, 
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ])
        else:
            modules.extend([
                nn.Conv2d(current_channels, D_prime, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ])
        
        current_channels = D_prime
        
        # M2_up, M3_up, M4_up: 2x2 upsampling modules
        for i in range(3):
            modules.extend([
                nn.PixelShuffle(2),
                nn.Conv2d(current_channels // 4, current_channels // 2, 
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ])
            current_channels = current_channels // 2
        
        # Final layer to output 3 channels
        modules.append(nn.Conv2d(current_channels, 3, kernel_size=3, stride=1, padding=1, bias=False))
        
        return nn.Sequential(*modules)
    
    def forward(self, feature_map: torch.Tensor, noisy_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DRH.
        
        Args:
            feature_map: Feature map from backbone CNN [B, C, H, W]
            noisy_image: Noisy input image [B, 3, H', W']
            
        Returns:
            recognition_output: Classification logits [B, num_classes]
            denoised_image: Denoised image [B, 3, H', W']
        """
        # Recognition Sub-head
        encoded_features = self.recognition_encoder(feature_map)
        recognition_output = self.recognition_classifier(encoded_features)
        
        # Denoising Sub-head
        # Restoration block
        try:
            restored_image = self.restoration_block(feature_map)
            # Resize to target size if needed
            if restored_image.shape[2:] != self.target_size:
                restored_image = F.interpolate(restored_image, size=self.target_size, 
                                             mode='bilinear', align_corners=False)
        except Exception as e:
            print(f"Restoration block error: {e}, using zeros")
            restored_image = torch.zeros_like(noisy_image)
        
        # Skip block
        skip_output = self.skip_block(noisy_image)
        
        # Combine restoration and skip outputs
        denoised_image = restored_image + skip_output
        
        return recognition_output, denoised_image

# ============================================================================
# TResNet Feature Extractor
# ============================================================================
class TResNetFeatureExtractor(nn.Module):
    """Extract features from TResNet backbone at multiple scales."""
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to extract features from the last 3 stages."""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        # Register hooks on TResNet body layers
        if hasattr(self.backbone, 'body'):
            # Extract from layer2, layer3, layer4 (last 3 stages)
            if hasattr(self.backbone.body, 'layer2'):
                hook = self.backbone.body.layer2.register_forward_hook(get_activation('layer2'))
                self.hooks.append(hook)
            if hasattr(self.backbone.body, 'layer3'):
                hook = self.backbone.body.layer3.register_forward_hook(get_activation('layer3'))
                self.hooks.append(hook)
            if hasattr(self.backbone.body, 'layer4'):
                hook = self.backbone.body.layer4.register_forward_hook(get_activation('layer4'))
                self.hooks.append(hook)
        else:
            print("Warning: Could not find TResNet body structure")
    
    def forward(self, x):
        """Extract multi-scale features."""
        # Clear previous features
        self.features = {}
        
        # Forward through backbone
        _ = self.backbone(x)
        
        # Return features from the 3 stages
        feat1 = self.features.get('layer2', None)
        feat2 = self.features.get('layer3', None) 
        feat3 = self.features.get('layer4', None)
        
        return feat1, feat2, feat3
    
    def __del__(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()

# ============================================================================
# PMAL Network Architecture
# ============================================================================
class PMALNetwork(nn.Module):
    """
    Progressive Multi-task Anti-noise Learning Network.
    Implements the complete PMAL framework with 3 DRHs.
    """
    
    def __init__(self, backbone, num_classes: int, img_size: int = 448):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Remove classification head from backbone
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()
        elif hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            backbone.classifier = nn.Identity()
        
        # Feature extractor
        self.feature_extractor = TResNetFeatureExtractor(backbone)
        
        # Get feature dimensions
        self._get_feature_dimensions()
        
        # Build DRHs for the 3 stages
        self._build_drhs()
        
        # Main classifier (for K+1 step)
        self.main_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat3_dim, num_classes)
        )
    
    def _get_feature_dimensions(self):
        """Determine feature dimensions by running a dummy forward pass."""
        self.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
            if next(self.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            feat1, feat2, feat3 = self.feature_extractor(dummy_input)
            
            if feat1 is not None:
                self.feat1_dim = feat1.shape[1]
                self.feat1_size = feat1.shape[2:]
            else:
                self.feat1_dim = 512  # fallback
                self.feat1_size = (56, 56)
            
            if feat2 is not None:
                self.feat2_dim = feat2.shape[1]
                self.feat2_size = feat2.shape[2:]
            else:
                self.feat2_dim = 1024  # fallback
                self.feat2_size = (28, 28)
            
            if feat3 is not None:
                self.feat3_dim = feat3.shape[1]
                self.feat3_size = feat3.shape[2:]
            else:
                self.feat3_dim = 2048  # fallback
                self.feat3_size = (14, 14)
        
        print(f"Feature dimensions - Stage1: {self.feat1_dim}, Stage2: {self.feat2_dim}, Stage3: {self.feat3_dim}")
        self.train()
    
    def _build_drhs(self):
        """Build DRH modules for each stage."""
        target_size = (self.img_size, self.img_size)
        
        self.drh1 = DenoisingRecognitionHead(
            input_channels=self.feat1_dim,
            num_classes=self.num_classes,
            input_size=self.feat1_size,
            target_size=target_size,
            D=1024, D_prime=256, D_double_prime=64
        )
        
        self.drh2 = DenoisingRecognitionHead(
            input_channels=self.feat2_dim,
            num_classes=self.num_classes,
            input_size=self.feat2_size,
            target_size=target_size,
            D=1024, D_prime=256, D_double_prime=64
        )
        
        self.drh3 = DenoisingRecognitionHead(
            input_channels=self.feat3_dim,
            num_classes=self.num_classes,
            input_size=self.feat3_size,
            target_size=target_size,
            D=1024, D_prime=256, D_double_prime=64
        )
    
    def forward(self, x, return_features=False):
        """
        Forward pass through PMAL network.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            If return_features=False: main classifier output
            If return_features=True: (drh_outputs, main_output, features)
        """
        # Extract multi-scale features
        feat1, feat2, feat3 = self.feature_extractor(x)
        
        if not return_features:
            # Main classifier output only (for inference)
            if feat3 is not None:
                main_output = self.main_classifier(feat3)
            else:
                # Fallback: use backbone directly
                backbone_out = self.feature_extractor.backbone(x)
                if len(backbone_out.shape) > 2:
                    backbone_out = F.adaptive_avg_pool2d(backbone_out, 1).flatten(1)
                main_output = self.main_classifier(backbone_out)
            return main_output
        
        # Return all outputs and features (for training)
        return feat1, feat2, feat3
    
    def forward_drh(self, stage: int, feature_map: torch.Tensor, noisy_image: torch.Tensor):
        """Forward through specific DRH stage."""
        if stage == 1:
            return self.drh1(feature_map, noisy_image)
        elif stage == 2:
            return self.drh2(feature_map, noisy_image)
        elif stage == 3:
            return self.drh3(feature_map, noisy_image)
        else:
            raise ValueError(f"Invalid stage: {stage}")

# ============================================================================
# PMD (Progressive Multi-task Distilling) Network
# ============================================================================
class PMDNetwork(nn.Module):
    """
    Progressive Multi-task Distilling Network.
    Student network that learns from PMAL teacher.
    """
    
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # Remove classification head
        if hasattr(backbone, 'head'):
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif hasattr(backbone, 'fc'):
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            self.backbone = backbone
        
        # Feature extractor (same structure as teacher)
        self.feature_extractor = TResNetFeatureExtractor(self.backbone)
        
        # Get dimensions
        self._get_feature_dimensions()
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat3_dim, num_classes)
        )
    
    def _get_feature_dimensions(self):
        """Get feature dimensions."""
        self.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 448, 448)
            if next(self.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            feat1, feat2, feat3 = self.feature_extractor(dummy_input)
            
            self.feat1_dim = feat1.shape[1] if feat1 is not None else 512
            self.feat2_dim = feat2.shape[1] if feat2 is not None else 1024
            self.feat3_dim = feat3.shape[1] if feat3 is not None else 2048
        
        self.train()
    
    def forward(self, x, return_features=False):
        """Forward pass."""
        if return_features:
            # Return intermediate features for distillation
            feat1, feat2, feat3 = self.feature_extractor(x)
            main_output = self.classifier(feat3) if feat3 is not None else self.classifier(x)
            return feat1, feat2, feat3, main_output
        else:
            # Main output only
            feat1, feat2, feat3 = self.feature_extractor(x)
            return self.classifier(feat3) if feat3 is not None else self.classifier(x)

# ============================================================================
# Main Lightning Module
# ============================================================================
class TResNetPMALClassifier(L.LightningModule):
    """
    Complete PMAL + PMD implementation with Progressive Learning.
    """
    
    def __init__(
        self,
        backbone_name: str = "tresnet_xl",
        pretrained: bool = True,
        img_size: int = 448,
        num_classes: int = 400,
        learning_rate: float = 2e-3,
        weight_decay: float = 5e-4,
        use_sam: bool = True,
        sam_rho: float = 0.05,
        noise_std: float = 0.05,
        alpha: float = 100.0,  # PMD feature distillation weight
        lambda_reg: float = 5e-4,  # L2 regularization
        teacher_path: Optional[str] = None,
        distill_phase_ratio: float = 0.5,
        mode: str = "pmal",  # "pmal" or "pmd"
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Hyperparameters
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        self.noise_std = noise_std
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.distill_phase_ratio = distill_phase_ratio
        self.mode = mode

        if use_sam:
            self.automatic_optimization = False
        
        # Load backbone
        self.backbone = self._load_backbone(backbone_name, pretrained, img_size)
        
        # Build networks
        if mode == "pmal":
            self.network = PMALNetwork(self.backbone, num_classes, img_size)
            self.teacher = None
            if teacher_path:
                self.teacher = self._load_teacher(teacher_path)
        elif mode == "pmd":
            # PMD mode requires teacher
            assert teacher_path is not None, "PMD mode requires teacher_path"
            self.teacher = self._load_teacher(teacher_path)
            self.network = PMDNetwork(self.backbone, num_classes)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Noise augmentation
        self._setup_noise_augmentation()
        
        # Training state
        self.current_epoch_num = 0
        self.distill_start_epoch = 0
    
    def _load_backbone(self, backbone_name: str, pretrained: bool, img_size: int):
        """Load TResNet backbone."""
        try:
            if backbone_name == "tresnet_xl" and img_size == 448:
                model_name = "tresnet_xl.miil_in1k_448"
            elif backbone_name == "tresnet_l" and img_size == 448:
                model_name = "tresnet_l.miil_in1k_448"
            elif backbone_name == "tresnet_m" and img_size == 448:
                model_name = "tresnet_m.miil_in1k_448"
            else:
                model_name = backbone_name
            
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
            )
            print(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            print(f"Failed to load {backbone_name}: {e}")
            model = timm.create_model(
                "tresnet_xl",
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
            )
            return model
    
    def _load_teacher(self, teacher_path: str):
        """Load teacher model."""
        try:
            checkpoint = torch.load(teacher_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Create teacher network
            teacher_backbone = self._load_backbone(self.hparams.backbone_name, True, self.hparams.img_size)
            teacher = PMALNetwork(teacher_backbone, self.num_classes, self.hparams.img_size)
            
            # Load state dict
            teacher.load_state_dict(state_dict, strict=False)
            teacher.eval()
            
            # Freeze teacher
            for param in teacher.parameters():
                param.requires_grad = False
            
            print(f"Successfully loaded teacher from {teacher_path}")
            return teacher
            
        except Exception as e:
            print(f"Warning: Could not load teacher model from {teacher_path}: {e}")
            return None
    
    def _setup_noise_augmentation(self):
        """Setup noise augmentation."""
        self.noise_seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(
                loc=0, 
                scale=(0.0, self.noise_std), 
                per_channel=0.5
            )
        ])
    
    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise to input tensor."""
        if self.training:
            # Convert to numpy for imgaug
            x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
            x_noisy_np = self.noise_seq(images=x_np)
            x_noisy = torch.from_numpy(x_noisy_np).permute(0, 3, 1, 2).to(x.device, x.dtype)
            return torch.clamp(x_noisy, 0, 1)
        return x
    
    def on_fit_start(self):
        """Initialize training parameters."""
        self.distill_start_epoch = int(self.trainer.max_epochs * self.distill_phase_ratio)
        print(f"Training mode: {self.mode}")
        if self.mode == "pmd":
            print(f"PMD distillation will start from epoch {self.distill_start_epoch}")
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def _compute_pmal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PMAL loss with progressive learning (Algorithm 1).
        """
        # Add noise to inputs
        noisy_inputs = self._add_noise(inputs)
        
        # Extract features
        feat1, feat2, feat3 = self.network(noisy_inputs, return_features=True)
        
        total_loss = 0
        loss_dict = {}
        predictions = []
        
        # Progressive learning: K steps (k=1,2,3)
        features = [feat1, feat2, feat3]
        
        for k in range(3):  # K=3 stages
            if features[k] is None:
                continue
                
            # Forward through DRH k
            p_rec, denoised_img = self.network.forward_drh(k+1, features[k], noisy_inputs)
            
            # L_rec: Recognition loss
            L_rec = self.ce_loss(p_rec, targets)
            
            # L_den^mse: MSE loss between denoised and clean images
            L_den_mse = self.mse_loss(denoised_img, inputs)
            
            # L_den^softmax: Re-input denoised image for recognition
            p_den = self.network(denoised_img)
            L_den_softmax = self.ce_loss(p_den, targets)
            
            # Total DRH loss
            L_drh = L_rec + L_den_mse + L_den_softmax
            
            total_loss += L_drh
            loss_dict[f'L_rec_{k+1}'] = L_rec.item()
            loss_dict[f'L_den_mse_{k+1}'] = L_den_mse.item()
            loss_dict[f'L_den_softmax_{k+1}'] = L_den_softmax.item()
            
            predictions.append(p_rec)
        
        # Step K+1: Main classifier
        main_output = self.network(noisy_inputs)
        L_main = self.ce_loss(main_output, targets)
        total_loss += L_main
        loss_dict['L_main'] = L_main.item()
        predictions.append(main_output)
        
        # Final prediction (ensemble)
        if len(predictions) > 0:
            final_pred = torch.stack(predictions).mean(dim=0)
        else:
            final_pred = main_output
        
        return total_loss, final_pred, loss_dict
    
    def _compute_pmd_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PMD loss with progressive distillation (Algorithm 2).
        """
        if self.teacher is None:
            # Fallback to simple classification
            outputs = self.network(inputs)
            loss = self.ce_loss(outputs, targets)
            return loss, outputs, {'pmd_fallback': loss.item()}
        
        total_loss = 0
        loss_dict = {}
        
        # Get teacher features and predictions
        with torch.no_grad():
            teacher_feat1, teacher_feat2, teacher_feat3 = self.teacher(inputs, return_features=True)
            teacher_main = self.teacher(inputs)
        
        # Get student features and predictions
        student_feat1, student_feat2, student_feat3, student_main = self.network(inputs, return_features=True)
        
        # Progressive feature distillation (Steps 1-K)
        teacher_features = [teacher_feat1, teacher_feat2, teacher_feat3]
        student_features = [student_feat1, student_feat2, student_feat3]
        
        for k in range(3):  # K=3 stages
            if teacher_features[k] is not None and student_features[k] is not None:
                # Feature alignment loss
                # Adapt feature sizes if needed
                t_feat = teacher_features[k]
                s_feat = student_features[k]
                
                if t_feat.shape != s_feat.shape:
                    # Simple channel adaptation
                    if t_feat.shape[1] != s_feat.shape[1]:
                        s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                        if not hasattr(self, f'feature_adapter_{k}'):
                            adapter = nn.Conv2d(s_feat.shape[1], t_feat.shape[1], 1).to(self.device)
                            setattr(self, f'feature_adapter_{k}', adapter)
                        s_feat = getattr(self, f'feature_adapter_{k}')(s_feat)
                    
                    # Spatial adaptation
                    if t_feat.shape[2:] != s_feat.shape[2:]:
                        s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
                
                L_feature = self.alpha * self.mse_loss(s_feat, t_feat)
                total_loss += L_feature
                loss_dict[f'L_feature_{k+1}'] = L_feature.item()
        
        # Step K+1: Score distillation and classification
        # Teacher prediction scores (if teacher has multiple outputs, use main)
        if hasattr(self.teacher, 'forward') and len(teacher_main.shape) == 2:
            teacher_scores = teacher_main
        else:
            teacher_scores = teacher_main
        
        # Score distillation loss
        L_score_distill = self.mse_loss(student_main, teacher_scores.detach())
        
        # Classification loss
        L_classification = self.ce_loss(student_main, targets)
        
        # Combined loss for step K+1
        L_final = L_score_distill + L_classification
        total_loss += L_final
        
        loss_dict['L_score_distill'] = L_score_distill.item()
        loss_dict['L_classification'] = L_classification.item()
        
        return total_loss, student_main, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step with progressive learning."""
        inputs, targets = batch
        self.current_epoch_num = self.current_epoch
        
        # Choose loss computation based on mode and epoch
        if self.mode == "pmal":
            loss, predictions, loss_dict = self._compute_pmal_loss(inputs, targets)
        elif self.mode == "pmd":
            # PMD can start immediately or after warmup
            loss, predictions, loss_dict = self._compute_pmd_loss(inputs, targets)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Calculate accuracy
        preds = torch.argmax(predictions, dim=1)
        acc = self.train_accuracy(preds, targets)
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log detailed losses
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        inputs, targets = batch
        
        # Use main network output for validation
        if self.mode == "pmal":
            predictions = self.network(inputs)
        else:  # PMD
            predictions = self.network(inputs)
        
        loss = self.ce_loss(predictions, targets)
        preds = torch.argmax(predictions, dim=1)
        acc = self.val_accuracy(preds, targets)
        
        self.log_dict({
            "val/loss": loss,
            "val/acc": acc,
        }, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def configure_optimizers(self):
        """Configure optimizers with SAM support."""
        # Different learning rates for different components
        if self.mode == "pmal":
            # PMAL: Different rates for backbone vs heads
            backbone_params = list(self.network.feature_extractor.parameters())
            head_params = (
                list(self.network.drh1.parameters()) +
                list(self.network.drh2.parameters()) +
                list(self.network.drh3.parameters()) +
                list(self.network.main_classifier.parameters())
            )
            
            param_groups = [
                {"params": backbone_params, "lr": self.learning_rate * 0.1},
                {"params": head_params, "lr": self.learning_rate}
            ]
        else:  # PMD
            # PMD: All parameters with same rate
            param_groups = [{"params": self.network.parameters(), "lr": self.learning_rate}]
        
        # Create base optimizer
        base_optimizer = torch.optim.AdamW
        optimizer_kwargs = {"weight_decay": self.weight_decay}
        
        if self.use_sam:
            # Use SAM optimizer
            optimizer = SAM(
                param_groups, 
                base_optimizer, 
                rho=self.sam_rho, 
                **optimizer_kwargs
            )
        else:
            # Standard optimizer
            optimizer = base_optimizer(param_groups, **optimizer_kwargs)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step for SAM."""
        if self.use_sam and isinstance(optimizer, SAM):
            # SAM requires closure
            def closure():
                loss = optimizer_closure()
                return loss
            
            optimizer.step(closure)
        else:
            # Standard step
            if optimizer_closure is not None:
                optimizer_closure()
            optimizer.step()
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step."""
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        inputs, _ = batch
        
        if self.mode == "pmal":
            # PMAL: Use ensemble of all predictions
            try:
                feat1, feat2, feat3 = self.network(inputs, return_features=True)
                predictions = []
                
                # Get predictions from all DRHs
                for k in range(3):
                    features = [feat1, feat2, feat3]
                    if features[k] is not None:
                        p_rec, _ = self.network.forward_drh(k+1, features[k], inputs)
                        predictions.append(p_rec)
                
                # Main classifier
                main_pred = self.network(inputs)
                predictions.append(main_pred)
                
                # Ensemble
                if len(predictions) > 1:
                    final_pred = torch.stack(predictions).mean(dim=0)
                else:
                    final_pred = main_pred
                    
            except Exception as e:
                print(f"Ensemble prediction failed: {e}, using main classifier")
                final_pred = self.network(inputs)
        else:
            # PMD: Use student network
            final_pred = self.network(inputs)
        
        return final_pred
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.network.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.network.feature_extractor.parameters():
            param.requires_grad = True