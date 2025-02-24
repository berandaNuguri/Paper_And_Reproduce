import os
import glob
from PIL import Image
import pandas as pd
import random

from tqdm import tqdm

if __name__ == '__main__':
    random.seed(42)
    
    root_dir = r'../../data/ALASKA2/ALASKA_v2_JPG_256_QF75_COLOR/'
    sample_num = 50000
    
    cover_root = os.path.join(root_dir, 'Cover')
    cover_paths = glob.glob(os.path.join(cover_root, '*.jpg'))
    cover_paths = random.sample(cover_paths, sample_num)
    
    stego_algorithms = ['nsf5_0.4']
    qf = root_dir.split('/')[-2].split('_')[4][2:]
    bpnzac = stego_algorithms[0].split('_')[1]
    
    csv_data = {alg: {qf: []} for alg in stego_algorithms}
    for alg in stego_algorithms:
        for cover_path in tqdm(cover_paths, total=len(cover_paths), desc=f'Processing {alg}'):
            basename = os.path.basename(cover_path).split('.')[0]
            try:
                csv_data[alg][qf].append((f'{basename}.jpg', qf, 'cover', '0'))
                csv_data[alg][qf].append((f'{basename}.jpg', qf, alg.split('_')[0], bpnzac))
            except Exception as e:
                print(f"Error processing {cover_path}: {e}")

        # 각 알고리즘/quality factor별로 Train/Validation/Test 분할 후 CSV 저장 (6:1:3 비율)
        for qf in csv_data[alg]:
            df = pd.DataFrame(csv_data[alg][qf], columns=['image_name', 'qf', 'algorithm', 'embedding_rate'])
            
            # 인덱스를 섞어서 한 번에 분할 (랜덤 시드를 고정하여 재현성 확보)
            indices = df.index.tolist()
           
            random.shuffle(indices)
            n = len(indices)
            train_end = int(0.6 * n)
            val_end = int(0.7 * n)
            
            train_df = df.iloc[indices[:train_end]]
            val_df = df.iloc[indices[train_end:val_end]]
            test_df = df.iloc[indices[val_end:]]
            
            # 저장 경로 생성 및 CSV 저장
            os.makedirs(f'./csv/ALASKA_v2_JPG_256_QF75_COLOR/{sample_num}/', exist_ok=True)
            train_df.to_csv(f'./csv/ALASKA_v2_JPG_256_QF75_COLOR/{sample_num}/{alg}_QF_{qf}_train.csv', index=False)
            val_df.to_csv(f'./csv/ALASKA_v2_JPG_256_QF75_COLOR/{sample_num}/{alg}_QF_{qf}_val.csv', index=False)
            test_df.to_csv(f'./csv/ALASKA_v2_JPG_256_QF75_COLOR/{sample_num}/{alg}_QF_{qf}_test.csv', index=False)

            print(f"[INFO] Saved {alg}_QF_{qf}_train.csv ({len(train_df)} samples)")
            print(f"[INFO] Saved {alg}_QF_{qf}_val.csv ({len(val_df)} samples)")
            print(f"[INFO] Saved {alg}_QF_{qf}_test.csv ({len(test_df)} samples)")
