import os
import shutil

# real과 fake 파일을 분류하여 적절한 폴더로 이동하는 함수
def classify_and_move_files(src_path):
    # Real과 Fake 폴더 생성
    real_path = os.path.join(src_path, 'real')
    fake_path = os.path.join(src_path, 'fake')
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)
    
    # 각 ID 폴더에 대해 반복
    for id_folder in os.listdir(src_path):
        if 'real' not in id_folder and 'fake' not in id_folder:
            id_folder_path = os.path.join(src_path, id_folder)

            # ID 폴더가 실제 폴더인지 확인 (디렉터리인 경우만 처리)
            if not os.path.isdir(id_folder_path):
                continue
            
            # 폴더 내의 모든 파일에 대해 반복
            for file in os.listdir(id_folder_path):
                src_file_path = os.path.join(id_folder_path, file)
                file_prefix = '_' if 'HR' in file else '_NM_'
                
                # 파일이 real에 해당하는 경우
                if file.startswith('HR_1') or file.startswith('1') or file.startswith('2'):
                    dst_file_path = os.path.join(real_path, f"{id_folder}{file_prefix}{file}")
                # 파일이 fake에 해당하는 경우
                else:
                    dst_file_path = os.path.join(fake_path, f"{id_folder}{file_prefix}{file}")
        
                # 파일을 새 위치로 이동
                shutil.move(src_file_path, dst_file_path)
            
            # 원래 ID 폴더가 비어 있다면 삭제
            if not os.listdir(id_folder_path):
                os.rmdir(id_folder_path)

if __name__ =='__main__':
    # 데이터셋의 경로 설정
    dataset_path = './data/MCIO/video/casia'
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    # train 및 test 폴더에 대해 함수 호출
    classify_and_move_files(train_path)
    classify_and_move_files(test_path)
