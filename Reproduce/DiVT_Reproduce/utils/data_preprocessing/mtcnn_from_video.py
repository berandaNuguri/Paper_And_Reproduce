import cv2
import os
import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch

def crop_faces_mtcnn(base_path, dest_path):
    benchmarks = ['casia', 'oulu', 'msu', 'replay']

    # Initialize a face detector with the default weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = MTCNN(keep_all=True, device=device)

    for bench in benchmarks:
        for phase in ['train', 'test']:
            for category in ['fake', 'real']:
                src_dir = os.path.join(base_path, bench, phase, category)
                dest_dir = os.path.join(dest_path, bench, phase, category)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                # Load all video files from the source directory
                video_files_avi = glob.glob(os.path.join(src_dir, '*.avi'))
                video_files_mov = glob.glob(os.path.join(src_dir, '*.mov'))
                video_files_mp4 = glob.glob(os.path.join(src_dir, '*.mp4'))
                files = video_files_avi + video_files_mov + video_files_mp4

                for file in tqdm(files, desc=f'{bench}[{phase} - {category}]'):
                    cap = cv2.VideoCapture(file)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

                    for frame_idx in range(total_frames):
                        # Read frame by frame
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Detect faces in the frame
                            boxes, _ = detector.detect(frame_rgb)
                            
                            if boxes is not None:
                                for i, box in enumerate(boxes):
                                    x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
                                    cropped_img = frame[int(y):int(y+height), int(x):int(x+width)]
                                    if cropped_img.size > 0:
                                        base_name = os.path.basename(file)
                                        file_name = os.path.splitext(base_name)[0]
                                        dest_img_path = os.path.join(dest_dir, f"{file_name}_frame{frame_idx}_crop{i}.png")
                                        cv2.imwrite(dest_img_path, cropped_img)

                    cap.release()

if __name__ == '__main__':
    base_path = '../Datas/MCIO/video'
    dest_path = '../Datas/MCIO/all_frame(MTCNN)'
    crop_faces_mtcnn(base_path, dest_path)
