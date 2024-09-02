import os
import shutil

def read_sub_list(sub_list_path):
    with open(sub_list_path, 'r') as file:
        ids = file.read().splitlines()
    return ids

def copy_files(sub_list, source_folder, dest_folder):
    for client_id in sub_list:
        # 파일 검색 및 복사
        for file in os.listdir(source_folder):
            if f"client0{client_id}_" in file:
                src_file_path = os.path.join(source_folder, file)
                dest_file_path = os.path.join(dest_folder, file)
                
                # 대상 폴더가 없으면 생성
                os.makedirs(dest_folder, exist_ok=True)
                # 파일 복사
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {src_file_path} -> {dest_file_path}")


if __name__ == '__main__':
    # 데이터셋의 루트 디렉토리 경로 설정
    dataset_root = "./data/MSU-MFSD(video)"

    # ID 목록 파일 경로
    train_sub_list_path = os.path.join(dataset_root, "information/train_sub_list.txt")
    test_sub_list_path = os.path.join(dataset_root, "information/test_sub_list.txt")

    # ID 목록 읽기
    train_ids = read_sub_list(train_sub_list_path)
    test_ids = read_sub_list(test_sub_list_path)

    # 'attack' 및 'real' 폴더에서 'train' 및 'test' 폴더로 파일 이동
    copy_files(train_ids, os.path.join(dataset_root, "attack"), os.path.join(dataset_root, "train/attack"))
    copy_files(train_ids, os.path.join(dataset_root, "real"), os.path.join(dataset_root, "train/real"))

    copy_files(test_ids, os.path.join(dataset_root, "attack"), os.path.join(dataset_root, "test/attack"))
    copy_files(test_ids, os.path.join(dataset_root, "real"), os.path.join(dataset_root, "test/real"))
