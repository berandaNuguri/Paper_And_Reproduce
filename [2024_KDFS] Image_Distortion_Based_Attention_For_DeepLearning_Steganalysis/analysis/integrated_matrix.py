import sys
import os
import random
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
import timm
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arch.iformer.inception_transformer import iformer_small
from utils.datasets import StegDataset

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)

def plot_matrix(matrix, filename, title, target_devices):
    matrix = np.array(matrix, dtype=float)
    
    sns.color_palette('pastel')
    
    plt.figure(figsize=(13, 7))
    plt.subplots_adjust(left=0.13, right=1, top=0.95, bottom=0.13)
    
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm",
                     xticklabels=target_devices, yticklabels=[title],
                     vmin=0, vmax=100, annot_kws={"size": 13})
    ax.set_aspect(1)
    plt.title("Integrated-Device Matrix", fontsize=14)
    plt.xlabel("Target Devices", fontsize=12)
    plt.ylabel("Train Devices", fontsize=12)
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=45, fontsize=10, ha='right')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    
    plt.savefig(filename)
    plt.close()
def test(args, target_devices):
    results = pd.DataFrame(index=['integrated'], columns=target_devices)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone.split('-')[0] == 'timm':
        backbone = timm.create_model(args.backbone.split('-')[1], pretrained=True, num_classes=2)
    elif args.backbone == 'iformer-small':
        # Required timm==0.5.4
        backbone = iformer_small(pretrained=True)
        backbone.head = nn.Sequential(
            nn.Dropout(p=args.dropout_rate),
            nn.Linear(backbone.head.in_features, 2)
        )
    
    if args.stride is not None:
        if args.backbone.split('-')[1] == 'mobilevit_s':
            backbone.stem.conv.stride = (args.stride, args.stride)
        elif args.backbone.split('-')[1] == 'efficientnet_b0':
            backbone.conv_stem.stride = (args.stride, args.stride)
        elif args.backbone == 'iformer-small':
            backbone.patch_embed.proj1.conv = (args.stride , args.stride)
            
    model = backbone
    state_dict = torch.load(f'./ckpt/{args.backbone}_integrated_{args.total_data}_{"_".join(map(str, args.band_scale))}_{args.stego_method}_{args.batch_size}_0.0001_subtract_only_LL/model_best.pth.tar')['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    model.eval()
    for target_device in target_devices:
        test_df = pd.read_csv(f'{args.csv_root}/single/{args.cover_size}_{args.stego_method}/{target_device}_valid.csv')
        test_dataset = StegDataset(test_df, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
        
        correct = 0
        with torch.no_grad():
            for coeffs, labels in test_loader:
                coeffs, labels = coeffs.to(device), labels.to(device)
                
                outputs = []
                for idx in range(4):
                    output = model(coeffs[:, idx, :, :, :])
                    outputs.append(output)
                
                    probs = torch.softmax(output.data, dim=1)
                    _, predicted = torch.max(probs, 1)
                    correct += (predicted == labels).sum().item()

        test_accuracy = round(100 * correct / (len(test_loader.dataset)*4), 2)
        
        results.loc['integrated', target_device] = test_accuracy

    results.to_csv(f'./results/integrated_inter-device_results.csv', index_label='Device')
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='iformer-small')
    parser.add_argument('--stego_method', type=str, default='LSB_0.5')
    parser.add_argument('--cover_size', type=str, default='224')
    parser.add_argument('--total_data', type=str, default='100000')
    
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to the input data')
    parser.add_argument('--pretrained_path', type=str, default='./ckpt', help='Path to the pretrained model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--band_scale', type=float, nargs='+',default=[0.1, 0.2, 0.2, 0.5], help='Scaling factor for each sub-band([LL, LH, HL, HH])')
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--workers_per_loader', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use')
    args = parser.parse_args()
    
    target_devices = ['Galaxy_S10+', 'Galaxy_S20+', 'Galaxy_S20FE', 'Galaxy_S21', 'Galaxy_S21_Ultra', 'Galaxy_S22', 'Galaxy_Note9', \
                      'Galaxy_Fold3', 'Galaxy_Fold4', 'Galaxy_Flip3', 'Galaxy_Flip4', 'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing']
    if args.gpus != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_transform = A.Compose([
        A.CenterCrop(height=224, width=224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    if args.result_path == None:
        results = test(args, target_devices)
    else:
        results = pd.read_csv(args.result_path, index_col='Device')
        
    plot_matrix(results, './results/integrated_inter-device_matrix.png', f'Integrated_{args.total_data}', target_devices)