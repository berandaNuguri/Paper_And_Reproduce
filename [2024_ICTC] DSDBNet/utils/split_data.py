import os
import shutil
from tqdm import tqdm

def split_data_for_validation(train_txt_path, images_dir, output_dir, split_img_number=3901):
    # 디렉토리 구조 생성
    os.makedirs(os.path.join(output_dir, 'train', 'real'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'fake'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    
    # train.txt 파일 읽기
    with open(train_txt_path, 'r') as file:
        lines = file.readlines()
    
    # 각 줄을 파싱하여 데이터 분리
    for line in tqdm(lines):
        mat_path, label = line.strip().split(' ')
        image_name = os.path.splitext(os.path.basename(mat_path))[0] + '.png'
        image_number = int(image_name.split('_')[0])  # 이미지 파일 이름에서 숫자 추출
        
        # 원본 이미지 경로
        src_path = os.path.join(images_dir, image_name)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(src_path):
            print(f"이미지 파일을 찾을 수 없습니다: {src_path}")
            continue
        
        # 이미지 번호에 따라 train 또는 validation 설정
        if image_number < split_img_number:
            # train 데이터는 라벨에 따라 분류
            dst_dir = os.path.join(output_dir, 'train', 'fake' if label == '0' else 'real')
        else:
            # validation 데이터는 라벨 구분 없이 저장
            dst_dir = os.path.join(output_dir, 'validation')
        
        # 이미지 파일 복사
        dst_path = os.path.join(dst_dir, image_name)
        shutil.copy(src_path, dst_path)
    
    print('데이터 분리 완료!')

# 사용 예시
train_txt_path = './data/phase1/train.txt'
images_dir = './data/phase1/images/'
output_dir = './data/phase1/split/'

split_data_for_validation(train_txt_path, images_dir, output_dir)
