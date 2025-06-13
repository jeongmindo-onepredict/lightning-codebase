from datasets import load_dataset
from PIL import Image
import os

def save_dataset_by_class(dataset_name, output_dir="./images_by_class", split="fold1of5"):
    """
    클래스별로 폴더를 나누어 이미지 저장 (실제 클래스 이름 사용)
    """
    dataset = load_dataset(dataset_name, split)

    train = dataset['train'] if 'train' in dataset else dataset
    validation = dataset['validation'] if 'validation' in dataset else None

    if train:
        save_images(train, os.path.join(output_dir, 'train'), dataset_name)
    if validation:
        save_images(validation, os.path.join(output_dir, 'val'), dataset_name)
    
def save_images(dataset, output_dir, dataset_name):
    # 라벨 정보가 있는지 확인
    if 'label' not in dataset.column_names:
        print("라벨 정보가 없어 클래스별 저장이 불가능합니다.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 클래스 이름 정보 가져오기
    class_names = None
    if hasattr(dataset.features['label'], 'names'):
        # ClassLabel feature인 경우
        class_names = dataset.features['label'].names
        print(f"클래스 이름들: {class_names}")
    else:
        # 다른 방법으로 클래스 이름 정보 찾기
        try:
            # 데이터셋 정보에서 클래스 이름 찾기
            dataset_info = load_dataset(dataset_name, split='train')
            if hasattr(dataset_info.features['label'], 'names'):
                class_names = dataset_info.features['label'].names
                print(f"클래스 이름들: {class_names}")
        except:
            print("클래스 이름 정보를 찾을 수 없습니다. 숫자 라벨을 사용합니다.")
    
    # 클래스별 카운터
    class_counters = {}
    
    for example in dataset:
        image = example['image']
        label = example['label']
        
        # PIL Image로 변환
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 클래스 폴더명 결정
        if class_names and label < len(class_names):
            class_folder_name = class_names[label]
            # 폴더명에 사용할 수 없는 문자 제거/변경
            class_folder_name = sanitize_folder_name(class_folder_name)
        else:
            class_folder_name = f"class_{label}"
        
        # 클래스 폴더 생성
        class_dir = os.path.join(output_dir, class_folder_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 클래스별 카운터 업데이트
        if class_folder_name not in class_counters:
            class_counters[class_folder_name] = 0
        
        # 파일명 생성
        filename = f"{class_counters[class_folder_name]:06d}.jpg"
        image_path = os.path.join(class_dir, filename)
        
        # 이미지 저장
        image.save(image_path, 'JPEG', quality=95)
        class_counters[class_folder_name] += 1
    
    print(f"클래스별로 이미지를 {output_dir}에 저장했습니다.")
    for class_name, count in class_counters.items():
        print(f"{class_name}: {count}개 이미지")

def sanitize_folder_name(name):
    """
    폴더명에 사용할 수 없는 문자들을 제거하거나 변경
    """
    # Windows와 Unix 모두에서 사용할 수 없는 문자들
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # 공백을 언더스코어로 변경
    name = name.replace(' ', '_')
    
    # 연속된 언더스코어를 하나로 변경
    while '__' in name:
        name = name.replace('__', '_')
    
    # 앞뒤 언더스코어 제거
    name = name.strip('_')
    
    return name

# 사용 예시
if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 처리
    import argparse
    parser = argparse.ArgumentParser(description="클래스별로 이미지 저장 스크립트")
    parser.add_argument('--dataset', type=str, default="ohjoonhee/UsedCarsImageNetGamma",
                        help="저장할 데이터셋 이름 (예: ohjoonhee/UsedCarsImageNetGamma)")
    parser.add_argument('--output_dir', type=str, default="./images_by_class",
                        help="이미지를 저장할 디렉토리 경로")
    parser.add_argument('--split', type=str, default="fold1of5",
                        help="데이터셋의 분할 이름 (예: fold1of5)")


    # 클래스별로 이미지 저장
    args = parser.parse_args()
    save_dataset_by_class(args.dataset, args.output_dir, args.split)
    
    print("이미지 저장이 완료되었습니다!")