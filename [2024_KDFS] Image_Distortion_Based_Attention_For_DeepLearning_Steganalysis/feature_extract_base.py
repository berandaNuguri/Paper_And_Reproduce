import os
import argparse
import random
import timm
import time
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arch.VIA.MobileViT_SIA import MobileViT_SIA
from utils.datasets import StegDataset

class MobileViT_FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model=None):
        super(MobileViT_FeatureExtractor, self).__init__()
        self.model = pretrained_model
        modules = list(self.model.children())
        
        self.stage0 = modules[0]

        stage1_modules = list(modules[1])
        self.stage1_0 = stage1_modules[0]
        self.stage1_1 = stage1_modules[1]
        self.stage1_2 = stage1_modules[2]
        self.stage1_3 = stage1_modules[3]
        self.stage1_4 = stage1_modules[4]
        
        self.stage2 = modules[2]
        self.stage3 = modules[3]
        
    def forward(self, x, return_features=False):
        # x Shape: (batch_size, 3, 512, 512)
        x_original = x.clone()  # 원본 이미지를 저장
        
        if return_features:
            feature_maps = []

        # Stage 0
        x = self.stage0(x)
        # if return_features:
        #     feature_maps.append(x)
        
        # Stage 1_0
        x = self.stage1_0(x)
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_1
        x = self.stage1_1(x)
        
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_2
        x = self.stage1_2(x)
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_3
            # print(f'SIA: mean={x_sia.mean().item():.4f}, std={x_sia.std().item():.4f}, min={x_sia.min().item():.4f}, max={x_sia.max().item():.4f}')
        x = self.stage1_3(x)
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_4
        x = self.stage1_4(x)
        # if return_features:
        #     feature_maps.append(x)

        # Stage 2
        x = self.stage2(x)
        if return_features:
            feature_maps.append(x)

        # Stage 3 - Classification Head
        x = self.stage3(x)

        if return_features:
            return x, feature_maps[0]
        else:
            return x
    
def visualize(features, labels, method="tsne", output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    features = features.reshape(features.shape[0],-1)


    print('Start Dimensional Reduction...')
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose between 'tsne' and 'umap'.")

    reduced_features = reducer.fit_transform(features)
    print('End Dimensional Reduction...')
    
    plt.figure()
    for label, color in zip([0, 1], ['lightblue', 'lightcoral']):
        indices = labels == label
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
                    s=10, alpha=0.3, label=f"{'Cover' if label == 0 else 'Stego'}", color=color)

    plt.title(f"{method.upper()} Visualization for Transformer Block")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.legend()

    save_path = os.path.join(output_dir, f"{method}_block.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)


def feature_extract(args, valid_loader, model, device):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, total=len(valid_loader), desc='Feature Extracting'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            _, features = model(inputs, return_features=True)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone.split('-')[0] == 'timm':
        model = timm.create_model(args.backbone.split('-')[1], pretrained=True, num_classes=2)

    if args.pretrained_path and os.path.isfile(args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path)['state_dict'], strict=True)
    model = MobileViT_FeatureExtractor(model)
    model.to(device)
    
    valid_csv = f"{args.csv_root}/StegoAppDB/{args.stego_method}_bpp_{args.min_embedding_rate}_{args.max_embedding_rate}_valid.csv"
    valid_df = pd.read_csv(valid_csv)
    dataset = StegDataset(valid_df, transform=test_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=False, drop_last=False)

    features, labels = feature_extract(args, loader, model, device)
    visualize(features, labels, method=args.dim_reduction, output_dir="visualizations")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2024 SCI Steganalysis Parser')
    # Model parameters
    parser.add_argument('--backbone', type=str, default='timm-mobilevit_s', help='Backbone architecture')
    parser.add_argument('--pretrained_path', type=str, default='./ckpt/timm-mobilevit_s_NIST_PixelKnot_bpp_0.1_0.15_w_32_0.0001.pth.tar', help='Path to pretrained model')

    # Dataset parameters
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to CSV files')
    parser.add_argument('--stego_method', type=str, default='PixelKnot', help='Stego method')
    parser.add_argument('--min_embedding_rate', type=float, default=0.1, help='Min embedding rate')
    parser.add_argument('--max_embedding_rate', type=float, default=0.15, help='Max embedding rate')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers_per_loader', type=int, default=4, help='Number of data loader workers')

    # Visualization parameters
    parser.add_argument('--dim_reduction', type=str, default='tsne', choices=['tsne', 'umap'], help='Dimensionality reduction method')

    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_transform = A.Compose([
        A.CenterCrop(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    main(args)
