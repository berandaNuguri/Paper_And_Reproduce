import os
import argparse
import random
import timm
import wandb
import time
import numpy as np
import pandas as pd
import albumentations as A
from timeit import default_timer as timer
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from arch.iformer.inception_transformer import iformer_small, iformer_base
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
    
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs.data, dim=1)
        _, predicted = torch.max(probs, 1)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)
    
    return avg_loss, accuracy

def validate(args, valid_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)

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
            if args.stride is not None or args.dropout_rate > 0:
                args.run_name = f'{args.backbone}_{args.train_device}_stride_{args.stride}_dropout_{args.dropout_rate}_{args.stego_method}'
            else:
                args.run_name = f'{args.backbone}_{args.train_device}_{args.stego_method}'
            if args.suffix != '':
                args.run_name += f'_{args.suffix}'
        wandb.run.name = args.run_name

    result_df = pd.DataFrame(columns=['epoch'] + args.val_devices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone.split('-')[0] == 'timm':
        backbone = timm.create_model(args.backbone.split('-')[1], pretrained=True, num_classes=2)
    elif args.backbone == 'iformer-small':
        backbone = iformer_small(pretrained=True)
        backbone.head = nn.Sequential(
            nn.Dropout(p=args.dropout_rate),
            nn.Linear(backbone.head.in_features, 2))
    elif args.backbone == 'iformer-base':
        backbone = iformer_base(pretrained=True)
        backbone.head = nn.Sequential(
            nn.Dropout(p=args.dropout_rate),
            nn.Linear(backbone.head.in_features, 2))

    if args.stride is not None:
        if args.backbone.split('-')[1] == 'mobilevit_s':
            backbone.stem.conv.stride = (args.stride, args.stride)
        elif args.backbone.split('-')[1] == 'efficientnet_b0':
            backbone.conv_stem.stride = (args.stride, args.stride)
        elif args.backbone.split('-')[0] == 'iformer':
            backbone.patch_embed.proj1.conv = (args.stride , args.stride)

    model = backbone
    if args.pretrained_path is not None:
        state_dict = torch.load(args.pretrained_path)['state_dict']
        model.load_state_dict(state_dict)
        
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_df = pd.read_csv(f'{args.csv_root}/single/{args.data_type}/{args.cover_size}_{args.stego_method}/{args.train_device}_train.csv')
    train_dataset = StegDataset(train_df, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

    valid_loaders = {}
    for val_device in args.val_devices:
        valid_df = pd.read_csv(f'{args.csv_root}/single/{args.data_type}/{args.cover_size}_{args.stego_method}/{val_device}_valid.csv')
        valid_dataset = StegDataset(valid_df, transform=test_transform)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)
        valid_loaders[val_device] = valid_loader

    # Print data num of each loader
    if args.stride is not None:
        print(f'Model: {args.backbone} with First conv stride({args.stride}, {args.stride})')
    else:
        print(f'Model: {args.backbone}')

    start_time = timer()

    print(f'Train Data: {len(train_loader.dataset)}')
    print(f'Valid Data: {len(valid_loader.dataset)}')
    print(f'=================================== {args.train_device} intra training ===================================')
    best_valid_acc = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs+1):
        valid_losses, valid_accs = {}, {}
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, device)
        print(f'Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')

        if args.use_wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc}, step=epoch)

        for val_device, valid_loader in valid_loaders.items():
            result_row = {'epoch': epoch}
            valid_loss, valid_acc = validate(args, valid_loader, model, criterion, device)
            valid_losses[val_device] = valid_loss
            valid_accs[val_device] = valid_acc

            result_row[val_device] = valid_accs[val_device]

            if args.use_wandb:
                wandb.log({f'{val_device}_valid_loss': valid_loss, f'{val_device}_valid_acc': valid_acc}, step=epoch)

        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
        mean_loss = sum(valid_losses.values()) / len(valid_losses)
        mean_acc = sum(valid_accs.values()) / len(valid_accs)
        
        print(f'Epoch [{epoch}/{args.epochs}] Valid Loss - ',  end = '')
        for val_device, valid_loss in valid_losses.items():
            print(f'{val_device}: {valid_loss:.4f}', end=' ')
        print(f'Mean Loss: {mean_loss:.4f} | {time_to_str(timer()-start_time, mode="min")}')
        
        print(f'Epoch [{epoch}/{args.epochs}] Valid Acc - ',  end = '')
        for val_device, valid_acc in valid_accs.items():
            print(f'{val_device}: {valid_acc:.2f}', end=' ')
        print(f'Mean Acc: {mean_acc:.2f} | {time_to_str(timer()-start_time, mode="min")}')

        is_best = mean_acc >= best_valid_acc
        if is_best:
            best_valid_acc = mean_acc
            best_epoch = epoch
            if args.use_wandb:
                wandb.log({'best_valid_acc': mean_acc}, step=epoch)

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
    print(f'Best Result[{best_epoch}Epoch]', end=' ')
    for val_device, valid_acc in valid_accs.items():
        print(f'{val_device}: {valid_acc:.2f}  ', end='')
    print()
    
    os.makedirs(f'./results/{args.run_name}', exist_ok=True)
    result_df.to_csv(f'./results/{args.run_name}/result.csv', index=False)
    wandb.finish()

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='2024 SCI Continual Learning Parser')
    # Model Parsers
    parser.add_argument('--backbone', type=str, default='iformer-small', help='Backbone name from timm library')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--stride', type=int, default=None, help='Stride of the backbone')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')

    # Training Parsers
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers_per_loader', type=int, default=4, help='90Number of workers per data loader')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--train_rate', type=float, default=0.7, help='Train split size')

    # MISC Parsers
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to the csv')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility') 
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/', help='Root directory for checkpoint saving')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--run_name', type=str, default='auto', help='Run name for checkpoint saving')

    parser.add_argument('--train_device', type=str, default='Galaxy_Flip3')
    parser.add_argument('--val_devices', type=list, nargs='+', default=['Galaxy_Flip3', 'Galaxy_S20+', 'iPhone12_ProMax', 'Huawei_P30', 'LG_Wing'])
    parser.add_argument('--data_type', type=str, default='PNG', choices=['JPEG', 'PNG'])
    parser.add_argument('--cover_size', type=str, default='224')
    parser.add_argument('--stego_method', type=str, default='LSB_0.5')
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
        A.CenterCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.CenterCrop(height=224, width=224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    if args.use_wandb:
        wandb.init(project='2024_SCI', entity='kumdingso')
        wandb.save(f'./utils/datasets.py', policy='now')
        wandb.save(f'./train_base.py', policy='now')
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)
    main(args)

