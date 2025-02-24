import os
import glob
from PIL import Image
import pandas as pd
import random

from tqdm import tqdm

if __name__ == '__main__':
    root_dir = r'../../../Data/ALASKA2/Competition/'

    # quantization table의 합을 통해 quality factor를 매핑 (예: 1858 -> 75, 736 -> 90, 369 -> 95)
    qf_dict = {1858: 75, 736: 90, 369: 95}

    cover_root = os.path.join(root_dir, 'Cover')
    cover_paths = glob.glob(os.path.join(cover_root, '*.jpg'))

    stego_algorithms = ['JMiPOD', 'UERD', 'JUNIWARD']
    
    # 알고리즘별, quality factor별로 CSV 데이터 저장을 위한 딕셔너리 생성
    csv_data = {alg: {qf: [] for qf in qf_dict.values()} for alg in stego_algorithms}

    for alg in stego_algorithms:
        for cover_path in tqdm(cover_paths, total=len(cover_paths), desc=f'Processing {alg}'):
            try:
                # Pillow를 이용하여 이미지 열기 (내부적으로 libjpeg 사용)
                image = Image.open(cover_path)
                image.load()  # quantization 테이블 읽기를 위해 이미지 데이터를 로드

                # Pillow의 quantization 정보는 딕셔너리 형태 (예: {0: [list of 64 ints], ...})
                quantization = image.quantization

                if 0 in quantization:
                    quant_tbl_sum = sum(quantization[0])
                else:
                    quant_tbl_sum = sum(next(iter(quantization.values())))
                
                if quant_tbl_sum in qf_dict:
                    qf = qf_dict[quant_tbl_sum]
                    basename = os.path.basename(cover_path).split('.')[0]
                    
                    # cover 이미지와 stego 이미지(알고리즘 이름을 label로)를 각각 추가
                    csv_data[alg][qf].append((f'{basename}.jpg', 'cover', qf))
                    csv_data[alg][qf].append((f'{basename}.jpg', alg, qf))
            except Exception as e:
                print(f"Error processing {cover_path}: {e}")

        # 각 알고리즘/quality factor별로 Train/Validation/Test 분할 후 CSV 저장 (6:1:3 비율)
        for qf in csv_data[alg]:
            df = pd.DataFrame(csv_data[alg][qf], columns=['image_name', 'algorithm', 'qf'])
            
            # 인덱스를 섞어서 한 번에 분할 (랜덤 시드를 고정하여 재현성 확보)
            indices = df.index.tolist()
            random.seed(42)
            random.shuffle(indices)
            n = len(indices)
            train_end = int(0.6 * n)
            val_end = int(0.7 * n)
            
            train_df = df.iloc[indices[:train_end]]
            val_df = df.iloc[indices[train_end:val_end]]
            test_df = df.iloc[indices[val_end:]]
            
            # 저장 경로 생성 및 CSV 저장
            os.makedirs('./csv/ALASKA2/', exist_ok=True)
            train_df.to_csv(f'./csv/ALASKA2/{alg}_QF_{qf}_train.csv', index=False)
            val_df.to_csv(f'./csv/ALASKA2/{alg}_QF_{qf}_val.csv', index=False)
            test_df.to_csv(f'./csv/ALASKA2/{alg}_QF_{qf}_test.csv', index=False)

            print(f"[INFO] Saved {alg}_QF_{qf}_train.csv ({len(train_df)} samples)")
            print(f"[INFO] Saved {alg}_QF_{qf}_val.csv ({len(val_df)} samples)")
            print(f"[INFO] Saved {alg}_QF_{qf}_test.csv ({len(test_df)} samples)")
