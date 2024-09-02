import os
import shutil

# 데이터셋의 기본 경로 설정
dataset_base_path = './data/MCIO/video/replay'

# train과 test 폴더에 대해 반복
for phase in ['train', 'test']:
    phase_path = os.path.join(dataset_base_path, phase)
    
    # fake 폴더의 경로 설정
    fake_path = os.path.join(phase_path, 'fake')
    
    # fixed와 hand 폴더의 경로 설정
    fixed_path = os.path.join(fake_path, 'fixed')
    hand_path = os.path.join(fake_path, 'hand')
    
    # fixed 폴더의 모든 파일을 fake 폴더로 이동
    for file in os.listdir(fixed_path):
        src_file_path = os.path.join(fixed_path, file)
        dst_file_path = os.path.join(fake_path, f'fixed_{file}')
        shutil.copy(src_file_path, dst_file_path)
    
    # hand 폴더의 모든 파일을 fake 폴더로 이동
    for file in os.listdir(hand_path):
        src_file_path = os.path.join(hand_path, file)
        dst_file_path = os.path.join(fake_path, f'hand_{file}')
        shutil.copy(src_file_path, dst_file_path)

