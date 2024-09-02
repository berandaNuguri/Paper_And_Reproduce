import os
import timm
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from arch.architecture import DSDBNet, SiameseNet
from utils.transform_configs import analysis_transforms
from utils.datasets import SSIDataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args, device):
    transform = analysis_transforms(args)

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
        # optimal_threshold = state['optimal_threshold']
        model.load_state_dict(state['state_dict'])
    except:
        print(f'Failed to load pretrained model from {args.pretrained_path}')
        exit()


    model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []

        for spect, png, labels in selected_loader:
            spect, png, labels = spect.to(device), png.to(device), labels.to(device)

            if args.arch == 'MatBackbone':
                outputs = model(spect)
            elif args.arch == 'PngBackbone':
                outputs = model(png)
            else:
                outputs = model(spect, png)

            preds = torch.softmax(outputs, dim=1)

            val_preds.extend(preds.detach().cpu().numpy())
            val_labels.extend(labels.detach().cpu().numpy().tolist())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        positive_preds = val_preds[:, 1]

        valid_auc = roc_auc_score(val_labels, positive_preds)
        _, _, thresholds = roc_curve(val_labels, [i for i in positive_preds])

        optimal_err_cand = []
        optimal_apcer = []
        optimal_bpcer = []
        optimal_thresholds = []
        for thresh in thresholds:
            _conf_mat = confusion_matrix(val_labels, [1 if i >= thresh else 0 for i in positive_preds])
            _tn, _fp, _fn, _tp = _conf_mat.ravel()
            _val_apcer = _fp / (_fp + _tn)
            _val_npcer = _fn / (_fn + _tp)
            _val_acer = (_val_apcer + _val_npcer) / 2
            optimal_err_cand.append(_val_acer)
            optimal_apcer.append(_val_apcer)
            optimal_bpcer.append(_val_npcer)
            optimal_thresholds.append(thresh)
            
        optimal_idx = np.argmin(optimal_err_cand)
        optimal_val_acer = optimal_err_cand[optimal_idx]
        optimal_val_apcer = optimal_apcer[optimal_idx]
        optimal_val_bpcer = optimal_bpcer[optimal_idx]
        optimal_val_threshold = optimal_thresholds[optimal_idx]

        print('optimal acer: ', optimal_val_acer)
        print('optimal threshold: ', optimal_val_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model Settings
    parser.add_argument('--arch', type=str, default='DSDBNet', choices=['MatBackbone', 'PngBackbone', 'DSDBNet', 'SiameseNet'])
    parser.add_argument('--feature_fusion', type=str, default='concat', choices=['concat', 'sum'])
    parser.add_argument('--backbone', type=str, default='mobilevit_s.cvnets_in1k')
    parser.add_argument('--pretrained_path', type=str, default='./ckpt/train_to_valid/DSDBNet_concat_mobilevit_s.cvnets_in1k_MAT_0_29_6/model_best.pth.tar')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--input_size', type=int, default=224)

    # Training Settings
    parser.add_argument('--data_loader', type=str, default='valid')
    parser.add_argument('--mix_chans', nargs='+', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
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