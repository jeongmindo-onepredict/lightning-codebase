import os
import os.path as osp

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)


class RichCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # 기본 ModelCheckpoint (베스트 모델 저장용)
        parser.add_lightning_class_args(ModelCheckpoint, "model_ckpt")
        parser.set_defaults(
            {
                "model_ckpt.monitor": "val/loss",
                "model_ckpt.mode": "min", 
                "model_ckpt.save_last": True,
                "model_ckpt.filename": "best-epoch={epoch:02d}-val_loss={val/loss:.4f}",
                "model_ckpt.auto_insert_metric_name": False,
            }
        )

        # 매 10 에폭마다 저장하는 추가 ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "periodic_ckpt")
        parser.set_defaults(
            {
                "periodic_ckpt.every_n_epochs": 10,
                "periodic_ckpt.save_top_k": -1,  # 모든 체크포인트 저장
                "periodic_ckpt.filename": "epoch={epoch:02d}",
                "periodic_ckpt.auto_insert_metric_name": False,
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val/loss",
                "early_stopping.mode": "min",
                "early_stopping.strict": False,
            }
        )

        # add `-n` argument linked with trainer.logger.name for easy cmdline access
        parser.add_argument(
            "--name", "-n", dest="name", action="store", default="default_name"
        )
        parser.add_argument(
            "--version", "-v", dest="version", action="store", default="version_0"
        )

        # add `--incremental_version` for sweep versioning, e.g. version_0, version_1, ...
        # This disables resume feature
        parser.add_argument("--increment_version", action="store_true", default=False)

    def _increment_version(self, save_dir: str, name: str) -> str:
        subcommand = self.config["subcommand"]
        if subcommand is None:
            return

        i = 0
        while osp.exists(osp.join(save_dir, name, f"version_{i}")):
            i += 1

        return f"version_{i}"

    def _update_model_ckpt_dirpath(self, logger_log_dir):
        if "subcommand" not in self.config:
            # 기본 체크포인트 경로 설정
            ckpt_root_dirpath = self.config["model_ckpt"]["dirpath"]
            if ckpt_root_dirpath:
                self.config["model_ckpt"]["dirpath"] = osp.join(
                    ckpt_root_dirpath, logger_log_dir, "checkpoints"
                )
            else:
                self.config["model_ckpt"]["dirpath"] = osp.join(
                    logger_log_dir, "checkpoints"
                )
            
            # 주기적 체크포인트 경로 설정
            periodic_ckpt_root_dirpath = self.config["periodic_ckpt"]["dirpath"]
            if periodic_ckpt_root_dirpath:
                self.config["periodic_ckpt"]["dirpath"] = osp.join(
                    periodic_ckpt_root_dirpath, logger_log_dir, "periodic_checkpoints"
                )
            else:
                self.config["periodic_ckpt"]["dirpath"] = osp.join(
                    logger_log_dir, "periodic_checkpoints"
                )
            return

        subcommand = self.config["subcommand"]
        
        # 기본 체크포인트 경로 설정
        ckpt_root_dirpath = self.config[subcommand]["model_ckpt"]["dirpath"]
        if ckpt_root_dirpath:
            self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
                ckpt_root_dirpath, logger_log_dir, "checkpoints"
            )
        else:
            self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
                logger_log_dir, "checkpoints"
            )
        
        # 주기적 체크포인트 경로 설정
        periodic_ckpt_root_dirpath = self.config[subcommand]["periodic_ckpt"]["dirpath"]
        if periodic_ckpt_root_dirpath:
            self.config[subcommand]["periodic_ckpt"]["dirpath"] = osp.join(
                periodic_ckpt_root_dirpath, logger_log_dir, "periodic_checkpoints"
            )
        else:
            self.config[subcommand]["periodic_ckpt"]["dirpath"] = osp.join(
                logger_log_dir, "periodic_checkpoints"
            )