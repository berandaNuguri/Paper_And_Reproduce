import os
import shutil
import glob
import jpegio as jio
import pandas as pd

from tqdm import tqdm

root_dir = r'../../data/ALASKA2/'

qf_dict = {1858: 75, 736: 90, 369: 95 }

cover_root = os.path.join(root_dir, 'Cover')
cover_paths = glob.glob(os.path.join(cover_root, '*.jpg'))

stego_algorithms = ['JMiPOD', 'UERD', 'JUNIWARD']

# Create a dictionary to store paths for each algorithm and quality factor
csv_data = {alg: {qf: [] for qf in qf_dict.values()} for alg in stego_algorithms}

for alg in stego_algorithms:
    for cover_path in tqdm(cover_paths, total=len(cover_paths), desc=f'Processing {alg}'):
        image = jio.read(cover_path)
        basename = os.path.basename(cover_path).split('.')[0]
        quant_tbl_sum = int(image.quant_tables[0].sum())
        
        if quant_tbl_sum in qf_dict:
            qf = qf_dict[quant_tbl_sum]
            
            stego_path = os.path.join(root_dir, 'Stego', alg, f'{basename}.jpg')
            csv_data[alg][qf].append((cover_path, 'cover', qf))
            csv_data[alg][qf].append((stego_path, alg, qf))

    for qf in csv_data[alg]:
        df = pd.DataFrame(csv_data[alg][qf], columns=['image_path', 'label', 'qf'])
        df.to_csv(f'./csv/ALASKA2/{alg}_QF_{qf}.csv', index=False)
