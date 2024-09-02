import os
import timm
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from arch.architecture import DSDBNet, SiameseNet
from utils.transform_configs import get_transforms_albu, analysis_transforms
from utils.datasets import SSIDataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def main(args, device):
    _, _, transform = get_transforms_albu(args)

    train_path = f'{args.data_root}/train.txt'
    valid_path = f'{args.data_root}/valid.txt'

    train_df = pd.read_csv(train_path, header=None, sep=" ", names=['path', 'label'])
    train_df, valid_df = train_test_split(train_df, test_size=0.3, stratify=train_df['label'], random_state=args.seed)

    test_df = pd.read_csv(valid_path, header=None, sep=" ", names=['path', 'label'])

    train_dataset = SSIDataset(df=train_df, root_path=args.data_root+'/train_valid/', mix_chans=args.mix_chans, transform_1=transform, transform_2=transform)
    valid_dataset = SSIDataset(df=valid_df, root_path=args.data_root+'/train_valid/', mix_chans=args.mix_chans, transform_1=transform, transform_2=transform)
    test_dataset = SSIDataset(df=test_df, root_path=args.data_root+'/train_valid/', mix_chans=args.mix_chans, transform_1=transform, transform_2=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=False,
        worker_init_fn=seed_worker,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        drop_last=False,
        shuffle=False,
        worker_init_fn=seed_worker,
    )
    
    if args.mix_chans is None:
        in_chans = 30
    else:
        in_chans = len(args.mix_chans)
    
    if args.data_loader == 'train':
        selected_loader = train_loader
    elif args.data_loader == 'valid':
        selected_loader = valid_loader
    elif args.data_loader == 'test':
        selected_loader = test_loader

    spect_encoder = timm.create_model(args.backbone, in_chans=in_chans, num_classes=2, pretrained=True)    
    png_encoder = timm.create_model(args.backbone, num_classes=2, pretrained=True)

    if args.arch == 'DSDBNet':
        model = DSDBNet(spect_encoder=spect_encoder, png_encoder=png_encoder, num_classes=2, feature_fusion=args.feature_fusion, global_pool=args.global_pool, spect_input_size=(1, in_chans, args.input_size, args.input_size)).to(device)
    elif args.arch == 'SiameseNet':
        model = SiameseNet(encoder=spect_encoder, num_classes=2, feature_fusion=args.feature_fusion, global_pool=args.global_pool, input_size=(1, in_chans, args.input_size, args.input_size)).to(device)
    elif args.arch == 'MatBackbone':
        model = spect_encoder.to(device)
    elif args.arch == 'PngBackbone':
        model = png_encoder.to(device)
    
    try:
        state = torch.load(args.pretrained_path)
        best_epoch = state['epoch']
        best_acer = state['best_hter']
        optimal_threshold = state['optimal_threshold']
        
        model.load_state_dict(state['state_dict'])
        print(f"{best_epoch} epoch's acer({best_acer*100:.2f}%) using optimal_threshold - {optimal_threshold}")
    except:
        print(f'Failed to load pretrained model from {args.pretrained_path}')
        exit()

    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        failure_cases = [] 
        for idx, (spect, png, label, filename) in enumerate(selected_loader): 
            spect, png, label = spect.to(device), png.to(device), label.to(device)

            logits, feature = model(spect, png, return_features=True)
            pred = torch.softmax(logits, dim=1)[:, 1]
            predicted_labels = (pred >= optimal_threshold).long()
            updated_labels = torch.where(predicted_labels != label, torch.tensor(2).to(device), label)
            failure_indices = (updated_labels == 2).nonzero(as_tuple=True)[0]

            failure_cases.extend([(filename[i], label[i].item()) for i in failure_indices.cpu().numpy()])
            features.append(feature.cpu().detach().numpy())
            labels.append(updated_labels.cpu().numpy())
            
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
    failure_cases_path = os.path.join(args.save_path, f'{args.arch}_{args.data_loader}_failure_cases.txt')
    with open(failure_cases_path, 'w') as f:
        for case, label in failure_cases:
            f.write(f'{case} {label}\n')

    print('Feature Transformation...')
    if args.visualize == 'tsne':
        tsne = TSNE(n_components=2, random_state=args.seed)
        features = tsne.fit_transform(features)
    elif args.visualize == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        features = reducer.fit_transform(features)
    print('Feature Transformation End!')

    os.makedirs(args.save_path, exist_ok=True)
    plt.figure(figsize=(10, 10))

    colors = ['lightcoral', 'skyblue', 'black']
    labels_names = ['Spoof', 'Live', 'Failure']
    for i, label_name in enumerate(labels_names):
        indices = np.where(labels == i)
        plt.scatter(features[indices, 0], features[indices, 1], label=label_name, alpha=0.5, color=colors[i])
    plt.legend()
    plt.title(f'Feature Visualization')
    plt.savefig(f'{args.save_path}/{args.arch}_{args.data_loader}_feature_visualization_{args.visualize}.png')
    plt.close()
    
        # features = []
        # labels = []
        # for spect, png, label, _ in selected_loader:
        #     spect, png, label = spect.to(device), png.to(device), label.to(device)

        #     _, feature = model(spect, png, return_features=True)
        #     features.append(feature.cpu().detach().numpy())
        #     labels.append(label.cpu().numpy())

        # features = np.concatenate(features, axis=0)
        # labels = np.concatenate(labels, axis=0)

        # print('Feature Transformation...')
        # if args.visualize == 'tsne':
        #     tsne = TSNE(n_components=2, random_state=args.seed)
        #     features = tsne.fit_transform(features)
        # elif args.visualize == 'umap':
        #     reducer = umap.UMAP(n_components=2, random_state=args.seed)
        #     features = reducer.fit_transform(features)
        # print('Feature Transformation End!')

        # os.makedirs(args.save_path, exist_ok=True)
        # plt.figure(figsize=(10, 10))
        # colors = ['lightcoral' if label == 0 else 'skyblue' for label in labels]
        # for label in np.unique(labels):
        #     indices = np.where(labels == label)
        #     label = 'Spoof' if label == 0 else 'Live'
        #     plt.scatter(features[indices, 0], features[indices, 1], label=label, alpha=0.5, color=colors[indices[0][0]])
        # plt.legend()
        # plt.title(f'Feature Space Visualization')
        # plt.savefig(f'{args.save_path}/{args.arch}_{args.data_loader}_feature_visualization_{args.visualize}.png')
        # plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model Settings
    parser.add_argument('--arch', type=str, default='MatBackbone', choices=['MatBackbone', 'PngBackbone', 'DSDBNet', 'SiameseNet'])
    parser.add_argument('--feature_fusion', type=str, default='concat', choices=['concat', 'sum'])
    parser.add_argument('--backbone', type=str, default='mobilevit_s.cvnets_in1k')
    parser.add_argument('--pretrained_path', type=str, default='./ckpt/train_to_valid/DSDBNet_concat_mobilevit_s.cvnets_in1k_MAT_0_29_6/model_best.pth.tar')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--input_size', type=int, default=224)

    # Training Settings 
    parser.add_argument('--data_loader', type=str, default='test')
    parser.add_argument('--mix_chans', nargs='+', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)

    # MISC Settings
    parser.add_argument('--data_root', type=str, default='./data/all/')
    parser.add_argument('--visualize', type=str, default='tsne', choices=['tsne', 'umap'])
    parser.add_argument('--save_path', type=str, default='./analysis/')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    main(args, device)