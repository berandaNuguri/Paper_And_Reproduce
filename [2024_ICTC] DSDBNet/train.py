import os
import timm
import argparse
import numpy as np
import pandas as pd
import random
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from arch.architecture import DSDBNet, SiameseNet
from utils.transform_configs import get_transforms_albu
from utils.datasets import SSIDataset
from utils.utils import save_checkpoint


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args, device):
    if args.use_wandb:
        if args.run_name != '':
            wandb.run.name = f'{args.run_name}'
        else:
            wandb.run.name = f'2024_KCC_protocol1'

        wandb.save(f'./utils/datasets.py', policy="now")
        wandb.save(f'./utils/transform_configs.py', policy="now")
        wandb.save(f'./train.py', policy="now")
        
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)

    train_transform_mat, train_transform_png, valid_transform = get_transforms_albu(args)

    train_path = f'{args.data_root}/train.txt'
    valid_path = f'{args.data_root}/valid.txt'

    train_df = pd.read_csv(train_path, header=None, sep=" ", names=['path', 'label'])
    train_df, valid_df = train_test_split(train_df, test_size=0.3, stratify=train_df['label'], random_state=args.seed)

    test_df = pd.read_csv(valid_path, header=None, sep=" ", names=['path', 'label'])

    class_sample_count = np.array([len(np.where(train_df['label'] == t)[0]) for t in np.unique(train_df['label'])])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_df['label']])

    train_dataset = SSIDataset(df=train_df, root_path=args.data_root+'/train_valid/', mix_chans=args.mix_chans, transform_1=train_transform_mat, transform_2=train_transform_png)
    valid_dataset = SSIDataset(df=valid_df, root_path=args.data_root+'/train_valid/', mix_chans=args.mix_chans, transform_1=valid_transform, transform_2=valid_transform)
    test_dataset = SSIDataset(df=test_df, root_path=args.data_root+'/train_valid/', mix_chans=args.mix_chans, transform_1=valid_transform, transform_2=valid_transform)
    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=16,
        drop_last=True,
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
        worker_init_fn=seed_worker,
    )

    if args.mix_chans is None:
        in_chans = 30
    else:
        in_chans = len(args.mix_chans)

    spect_encoder = timm.create_model(args.backbone, in_chans=in_chans, num_classes=2, pretrained=args.pretrained)    
    png_encoder = timm.create_model(args.backbone, num_classes=2, pretrained=args.pretrained)

    if args.arch == 'DSDBNet':
        model = DSDBNet(spect_encoder=spect_encoder, png_encoder=png_encoder, num_classes=2, feature_fusion=args.feature_fusion, global_pool=args.global_pool, spect_input_size=(1, in_chans, args.input_size, args.input_size)).to(device)
    elif args.arch == 'SiameseNet':
        model = SiameseNet(encoder=spect_encoder, num_classes=2, feature_fusion=args.feature_fusion, global_pool=args.global_pool, input_size=(1, in_chans, args.input_size, args.input_size)).to(device)
    elif args.arch == 'MatBackbone':
        model = spect_encoder.to(device)
    elif args.arch == 'PngBackbone':
        model = png_encoder.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = {'softmax': nn.CrossEntropyLoss()}

    print(f'Model: {args.backbone}, Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test {len(test_dataset)}')

    # Training
    best_val_acer = 1.0
    best_test_acer = 1.0
    for epoch in range(args.epochs):
        model.train()

        train_loss = []
        train_preds = []
        train_labels = []  
        running_loss = 0.0
        for spect, png, labels in train_loader:
            spect, png, labels = spect.to(device), png.to(device), labels.to(device)
            optimizer.zero_grad()

            if args.arch == 'MatBackbone':
                outputs = model(spect)
            elif args.arch == 'PngBackbone':
                outputs = model(png)
            else:
                outputs = model(spect, png)

            ce_loss = criterion['softmax'](outputs, labels)
            loss = ce_loss

            loss.backward()
            optimizer.step()

        model.eval()
        optimizer.zero_grad()
        with torch.no_grad():
            valid_loss = []
            val_preds = []
            val_labels = []
            running_loss = 0.0

            for spect, png, labels in valid_loader:
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

                ce_loss = criterion['softmax'](outputs, labels)
                
                loss = ce_loss
                running_loss += loss.item()

            valid_loss.append(running_loss / len(valid_loader))
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

            is_best = optimal_val_acer <= best_val_acer
            if is_best:
                best_val_acer = optimal_val_acer
                best_val_threshold = optimal_val_threshold
                if args.use_wandb:
                    wandb.log({'valid_auc': valid_auc, 'valid_acer': optimal_val_acer , 'valid_apcer': optimal_val_apcer, 'valid_bpcer': optimal_val_bpcer}, step=epoch)

        # print(f'Epoch [{epoch}/{args.epochs-1}], Valid - Loss: {valid_loss[-1]:.4f}, AUC: {valid_auc:.4f}, ACER: {optimal_val_acer:.4f}, APCER: {optimal_val_apcer:.4f}, BPCER: {optimal_val_bpcer:.4f}')
        
        if is_best:
            model.eval()
            test_loss = []
            test_preds = []
            test_labels = []
            with torch.no_grad():
                for spect, png, labels in test_loader:
                    spect, png, labels = spect.to(device), png.to(device), labels.to(device)

                    if args.arch == 'MatBackbone':
                        outputs = model(spect)
                    elif args.arch == 'PngBackbone':
                        outputs = model(png)
                    else:
                        outputs = model(spect, png)

                    preds = torch.softmax(outputs, dim=1)

                    test_preds.extend(preds.detach().cpu().numpy())
                    test_labels.extend(labels.detach().cpu().numpy().tolist())

                    ce_loss = criterion['softmax'](outputs, labels)
                    
                    loss = ce_loss
                    test_loss.append(loss.item())

                test_preds = np.array(test_preds)
                test_labels = np.array(test_labels)

                positive_preds = test_preds[:, 1]

                test_auc = roc_auc_score(test_labels, positive_preds)
                _conf_mat = confusion_matrix(test_labels, [1 if i >= best_val_threshold else 0 for i in positive_preds])
                _tn, _fp, _fn, _tp = _conf_mat.ravel()
                _test_apcer = _fp / (_fp + _tn)
                _test_npcer = _fn / (_fn + _tp)
                _test_acer = (_test_apcer + _test_npcer) / 2

                is_test_best = _test_acer <= best_test_acer
                if is_test_best:
                    best_test_auc = test_auc
                    best_test_acer = _test_acer
                    best_test_apcer = _test_apcer
                    best_test_bpcer = _test_npcer

                if args.use_wandb:
                    wandb.log({'test_auc': test_auc, 'test_acer': _test_acer, 'test_apcer': _test_apcer, 'test_bpcer': _test_npcer}, step=epoch)            

        print(f'Epoch [{epoch}/{args.epochs-1}], Test - AUC: {test_auc:.4f}, ACER: {_test_acer:.4f}, APCER: {_test_apcer:.4f}, BPCER: {_test_npcer:.4f}')

        if epoch == args.epochs - 1:
            print(f'===========================================================================')
            print(f'Final Best Scores - AUC: {best_test_auc:.4f}, ACER: {best_test_acer:.4f}, APCER: {best_test_apcer:.4f}, BPCER: {best_test_bpcer:.4f}')
            
        if args.save_model and is_test_best:
            save_checkpoint(
                state={
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_hter': best_test_acer,
                'optimal_threshold': best_val_threshold,
                },
                gpus=args.gpu,
                is_best=is_best,
                model_path=f'{args.ckpt_root}/{args.run_name}/',
                model_name=f'checkpoint_ep{epoch}.pth.tar')
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Settings
    parser.add_argument('--arch', type=str, default='MatBackbone', choices=['MatBackbone', 'PngBackbone', 'DSDBNet', 'SiameseNet'])
    parser.add_argument('--feature_fusion', type=str, default='concat', choices=['concat', 'sum'])
    parser.add_argument('--backbone', type=str, default='mobilevit_s.cvnets_in1k')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--save_model', action='store_true')

    # Training Settings
    parser.add_argument('--mix_chans', nargs='+', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)

    # MISC Settings
    parser.add_argument('--run_name', type=str, default='2024_KCC')
    parser.add_argument('--data_root', type=str, default='./data/all/')
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/train_to_valid/')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)

        
    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(project='2024_KCC_protocol1', entity='kumdingso')

    main(args, device)