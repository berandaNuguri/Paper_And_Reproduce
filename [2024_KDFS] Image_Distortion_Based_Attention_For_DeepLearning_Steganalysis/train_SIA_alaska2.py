import os
import argparse
import random
import timm
import wandb
import time
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from arch.iformer.inception_transformer import iformer_small, iformer_base
from arch.VIA.MobileViT_SIA import MobileViT_SIA
from arch.VIA.MobileViT_SIA_ import MobileViT_SIA_
from utils.datasets import StegDataset
from utils.utils import save_checkpoint, time_to_str

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)

def train(args, train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if args.return_features:
            outputs, feature_maps = model(inputs, apply_sia=args.apply_sia, attention_type=args.attention_type, return_features=True)
        elif args.collect_attention_features:
            outputs, feature_maps = model(inputs, apply_sia=args.apply_sia, attention_type=args.attention_type, collect_attention_features=True)
        else:
            outputs = model(inputs, apply_sia=args.apply_sia, attention_type=args.attention_type)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs.data, dim=1)
        _, predicted = torch.max(probs, 1)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)
    
    if args.return_features or args.collect_attention_features:
        return avg_loss, accuracy, feature_maps
    else:
        return avg_loss, accuracy

def validate(args, valid_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs, apply_sia=args.apply_sia, attention_type=args.attention_type)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs.data, dim=1)
            _, predicted = torch.max(probs, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / len(valid_loader.dataset)
    
    return avg_loss, accuracy

def main(args):
    start_time = time.time()
    if args.use_wandb:
        if args.run_name == 'auto':
            args.run_name = f'MobileViT_ALASKA2_{args.stego_method}_QF_{args.qf}_w_{args.batch_size}_{args.lr}'
            if args.apply_sia:
                args.run_name += f'_SIA_AfterViTBlock_{args.attention_type}_{args.rho_scaler}'
            if args.suffix != '':
                args.run_name += f'_{args.suffix}'
        wandb.run.name = args.run_name
        
        wandb.save(f'./utils/datasets.py', policy='now')
        wandb.save(f'./train_SIA_alaska2.py', policy='now')
        
    result_df = pd.DataFrame(columns=['epoch', 'train_acc', 'valid_acc'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = MobileViT_SIA(num_classes=2, rho_scaler=args.rho_scaler)

    model = backbone
    if args.pretrained_path is not None:
        state_dict = torch.load(args.pretrained_path)['state_dict']
        model.load_state_dict(state_dict)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    df = pd.read_csv(f'{args.csv_root}/ALASKA2/{args.stego_method}_QF_{args.qf}.csv')
    train_df, val_df = train_test_split(df, stratify=df['label'], test_size=0.3, random_state=args.seed)

    train_dataset = StegDataset(train_df, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
    
    valid_dataset = StegDataset(val_df, transform=test_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)

    start_time = timer()

    print(f'Train Data: {len(train_loader.dataset)}')
    print(f'Valid Data: {len(valid_loader.dataset)}')
    print(f'=================================== ALASKA2 {args.stego_method} QF_{args.qf} training ===================================')
    best_valid_acc = 0.0
    best_epoch = 0
    if args.use_wandb and args.apply_sia:
        wandb.log({'scale_factor': model.rho_scaler.item()}, step=0)
    for epoch in range(1, args.epochs+1):
        result_row = {'epoch': epoch}
        if args.return_features or args.collect_attention_features:
            train_loss, train_acc, feature_maps = train(args, train_loader, model, criterion, optimizer, device)
        else:
            train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, device)

        print(f'Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')
        result_row['train_acc'] = train_acc
        if args.use_wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc}, step=epoch)
            if args.apply_sia:
                wandb.log({'scale_factor': model.rho_scaler.item()}, step=epoch)
            
        valid_loss, valid_acc = validate(args, valid_loader, model, criterion, device)
        print(f'Epoch [{epoch}/{args.epochs}], Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')
        result_row['valid_acc'] = valid_acc
        if args.use_wandb:
            wandb.log({f'valid_loss': valid_loss, f'valid_acc': valid_acc}, step=epoch)
        
        if args.apply_sia:
            print(f'Epoch [{epoch}/{args.epochs}], {model.rho_scaler}')
        
        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
        
        is_best = valid_acc >= best_valid_acc
        if is_best:
            best_valid_acc = valid_acc
            best_epoch = epoch
            if args.use_wandb:
                wandb.log({'best_valid_acc': valid_acc}, step=epoch)

        if args.save_model:
            save_checkpoint(
                state={
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_valid_acc': best_valid_acc,
                },
                gpus=args.gpus,
                is_best=is_best,
                model_path=f'{args.ckpt_root}/{args.run_name}/',
                model_name=f'ep{epoch}.pth.tar')

    print(f'=================================== Training End ===================================')
    print(f'Best Result[{best_epoch}Epoch] - Acc: {best_valid_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')

    os.makedirs(f'./results/{args.run_name}', exist_ok=True)
    result_df.to_csv(f'./results/{args.run_name}/result.csv', index=False)
    wandb.finish()

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='2024 SCI Steganalysis Parser')
    # Model Parsers
    parser.add_argument('--apply_sia', action='store_true', help='Whether to apply SIA')
    parser.add_argument('--attention_type', type=str, default='add', choices=['add', 'mul'], help='Attention type')
    parser.add_argument('--collect_attention_features', action='store_true', help='Whether to collect attention features')
    parser.add_argument('--return_features', action='store_true', help='Whether to return feature maps')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')

    # Training Parsers
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers_per_loader', type=int, default=4, help='Number of workers per data loader')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--rho_scaler', type=float, default=0.5, help='Rho scaler for SIA')
    
    # MISC Parsers
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to the csv')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility') 
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/', help='Root directory for checkpoint saving')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--run_name', type=str, default='auto', help='Run name for checkpoint saving')

    parser.add_argument('--qf', type=int, default=90, help='Quality factor')
    parser.add_argument('--stego_method', type=str, default='UERD', help='Stego method')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for the run name')
    args = parser.parse_args()

    if args.gpus != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_transform = A.Compose([
        A.CenterCrop(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.CenterCrop(height=512, width=512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    

    
    if args.use_wandb:
        wandb.init(project='2024_SCI', entity='kumdingso')
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)
    main(args)

