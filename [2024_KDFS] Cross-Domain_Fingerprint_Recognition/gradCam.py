import os
import argparse

from tqdm import tqdm
from glob import glob
from torch_cam.scripts import cam_torch


parser = argparse.ArgumentParser(description="Args for choosing augmentation method and dataset")
parser.add_argument('--aug', default="None", type=str, help='choose augmentations')
parser.add_argument('--dataset', default='Dermalog', help='choose dataset')

parse = parser.parse_args()

dataset_name = parse.dataset
name_of_aug = parse.aug

# 이미지 경로 설정
root_path = "D:/Projects/AIM/Competition/LivDet2023/data"

cam_methods = ["GradCAMpp","CAM","GradCAM","SmoothGradCAMpp","LayerCAM"]

fake_image_list = []
live_image_list = []
for kind in (['FAKE', 'LIVE']):
    for path in glob(f'{root_path}/{dataset_name}/**/{kind}/*.jpg'):
        path = path.replace('\\', '/')
        name = path.split('/')
        identity = name[len(name)-3]
        finger = name[len(name)-1].split('.')[0]

        if kind == 'FAKE':
            fake_image_list.append({
                'path': path,
                'id': identity,
                'finger': finger
            })
        elif kind =='LIVE':
            live_image_list.append({
                'path': path,
                'id': identity,
                'finger': finger
            })

"""
Visualize Cam Function Description

arch: timm model name (default = efficientnet_b0)
img: image path to plot gradCame image (default = "")
class_idx: class index to plot gradCam image (default = 0)
device: cuda or cpu (default = None)
dataset: dataset name to plot gradCam image (default = "")
savefig: path to save image (default = "D:/Projects/AIM/Competition/LivDet2023/results/Dermalog/")
aug_method: choose augmentation method (default = "None")
method: choose gradCam method("CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "LayerCAM", ... ) (default = "CAM,GradCAM,GradCAMpp,SmoothGradCAMpp,LayerCAM")
target: choose target layer's name (default = "bn2")
alpha: strength of transperate (default = 0.5)
flag: FAKE or LIVE (default = "FAKE")
finger_id: information of id and finger (default = "")

"""

for method in tqdm(cam_methods):
    for idx, data in tqdm(enumerate(fake_image_list)):
        finger_id = f"{data['id']}_{data['finger']}"

        cam_torch.visualize_cam(arch="efficientnet_b0",
                                img=data['path'],
                                class_idx=0,
                                dataset=parse.dataset,
                                savefig="D:/Projects/AIM/Competition/LivDet2023/results/",
                                aug_method=parse.aug,
                                method=method,
                                # target='conv_head',
                                alpha=0.5,
                                flag="FAKE",
                                finger_id=finger_id)

    for idx, data in tqdm(enumerate(live_image_list)):
        finger_id = f"{data['id']}_{data['finger']}"

        cam_torch.visualize_cam(arch="efficientnet_b0",
                                img=data['path'],
                                class_idx=0,
                                dataset=parse.dataset,
                                savefig="D:/Projects/AIM/Competition/LivDet2023/results/",
                                aug_method=parse.aug,
                                method=method,
                                # target='conv_head',
                                alpha=0.5,
                                flag="LIVE",
                                finger_id=finger_id)
