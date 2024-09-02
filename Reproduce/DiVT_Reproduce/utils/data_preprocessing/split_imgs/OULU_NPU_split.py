import os
import shutil

def classify_videos_based_on_file_number(base_dir):
    """
    base_dir: Base directory where 'train' and 'test' directories are located.
    """
    for sub_dir in ['train', 'test']:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        videos = [f for f in os.listdir(sub_dir_path) if f.endswith('.txt')]

        for video in videos:
            # Extracting the File number from the video file name
            file_number = int(video.split('_')[-1].split('.')[0])
            
            # Determine the destination folder (real or fake) based on File number
            if file_number == 1:
                dest_folder = 'real'
            else:
                dest_folder = 'fake'
            
            # Construct source and destination paths
            src_path = os.path.join(sub_dir_path, video)
            dest_path = os.path.join(sub_dir_path, dest_folder, video)
            
            # Create destination folder if it does not exist
            if not os.path.exists(os.path.join(sub_dir_path, dest_folder)):
                os.makedirs(os.path.join(sub_dir_path, dest_folder))

            # Move the video file
            shutil.copy(src_path, dest_path)
            print(f'Moved {src_path} to {dest_path}')

# Example usage
base_dir = './data/MCIO/video/oulu'
classify_videos_based_on_file_number(base_dir)
