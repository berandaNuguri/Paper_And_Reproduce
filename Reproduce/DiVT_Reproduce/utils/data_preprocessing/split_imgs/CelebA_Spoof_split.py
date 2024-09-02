import os
import shutil

def reorganize_data(root_dir):
    for split in ['train', 'test']:
        for category in ['live', 'spoof']:
            new_dir = os.path.join(root_dir, split, category)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                
    for split in ['train', 'test']:
        split_dir = os.path.join(root_dir, split)
        for id_dir in os.listdir(split_dir):
            id_path = os.path.join(split_dir, id_dir)
            if os.path.isdir(id_path):
                for category in ['live', 'spoof']:
                    category_path = os.path.join(id_path, category)
                    if os.path.exists(category_path):
                        for file in os.listdir(category_path):
                            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.txt'):
                                new_file_name = f"{id_dir}_{file}"
                                src_path = os.path.join(category_path, file)
                                dest_path = os.path.join(root_dir, split, category, new_file_name)
                                shutil.move(src_path, dest_path)
                                
    # Clean up empty directories
    for split in ['train', 'test']:
        split_dir = os.path.join(root_dir, split)
        for id_dir in os.listdir(split_dir):
            id_path = os.path.join(split_dir, id_dir)
            if os.path.isdir(id_path):
                for category in ['live', 'spoof']:
                    category_path = os.path.join(id_path, category)
                    if os.path.exists(category_path) and not os.listdir(category_path):
                        os.rmdir(category_path)
                if not os.listdir(id_path):
                    os.rmdir(id_path)

# Example usage
root_dir = '../Datas/MCIO/frame/celeba'
reorganize_data(root_dir)
