import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import random
import timm
import wandb
import time
import numpy as np
import pandas as pd
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import save_checkpoint, time_to_str, load_dataset, load_model

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

def test(args, test_loader, model, criterion, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs.data, dim=1)
            _, predicted = torch.max(probs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(test_loader.dataset)
    return accuracy

def main(args):
    overall_start_time = time.time()
    if args.use_wandb:
        if args.run_name == 'auto':
            args.run_name = f'{args.model_name}'
            if ('CBAM' in args.model_name or args.model_name == 'SFANet') and args.apply_sfa:
                args.run_name += f'_ResConnection_{args.residual_connection}_TrainableFilter_{args.trainable_filter}'
            elif args.model_name == 'UCNet_WOW'
                args.run_name += f'_TrainableFilter_{args.trainable_filter}'

            # if '0.0' not in (args.min_bpp, args.max_bpp) and args.min_bpp == args.max_bpp:
            #     args.run_name += f'_{args.data}_{args.stego_method}_QF{args.qf}_mbpp_{args.max_bpp}_w_{args.batch_size}_{args.lr}'
            # elif float(args.min_bpp) > 0.0 and float(args.max_bpp) > 0.0:
            #     args.run_name += f'_{args.data}_{args.stego_method}_QF{args.qf}_bpp_{args.min_bpp}_to_{args.max_bpp}_w_{args.batch_size}_{args.lr}'
            # else:
            args.run_name += f'_{args.data}_{args.data_size}_{args.stego_method}_QF{args.qf}_{args.sample_pair if int(args.sample_pair) > 0 else "All"}Pair_w_{args.batch_size}_{args.optimizer}_{args.lr}'
                
            if args.suffix != '':
                args.run_name += f'_{args.suffix}'
        wandb.run.name = args.run_name

    result_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'test_acc'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args, device)

    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'MultiStep':
        print("Learning rate will be decreased at epochs 70, 110, and 160.")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 110, 160], gamma=0.1)
 
    train_loader, valid_loader, test_loader = load_dataset(args)
    
    start_time = timer()
    print(f'Train Data: {len(train_loader.dataset)}')
    print(f'Validation Data: {len(valid_loader.dataset)}')
    print(f'=================================== ALASKA2 Training ===================================')
    
    best_valid_acc = 0.0
    best_epoch = 0
    best_train_loss = None
    best_train_acc = None
    best_valid_loss = None
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, device)
        print(f'Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')
        
        if args.use_wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc}, step=epoch)

        valid_loss, valid_acc = validate(args, valid_loader, model, criterion, device)
        print(f'Epoch [{epoch}/{args.epochs}], Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')
        
        if args.use_wandb:
            wandb.log({'valid_loss': valid_loss, 'valid_acc': valid_acc}, step=epoch)

        result_row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'valid_loss': round(valid_loss, 4),
            'valid_acc': round(valid_acc, 2),
            'test_acc': None
        }
        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
        
        is_best = valid_acc >= best_valid_acc
        if is_best:
            best_valid_acc = valid_acc
            best_epoch = epoch
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_valid_loss = valid_loss
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
                    model_name=f'ep{epoch}.pth.tar'
                )
    
    print(f'=================================== Training End ===================================')
    print(f'Best Valid[{best_epoch}Epoch] - Valid Acc: {best_valid_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')
    
    best_model_path = os.path.join(f'{args.ckpt_root}/{args.run_name}/', f'ep{best_epoch}.pth.tar')
    print(f"Loading best model from {best_model_path} for testing.")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    print('=================================== Test Start ===================================')
    test_acc = test(args, test_loader, model, criterion, device)
    print(f'Test Acc: {test_acc:.2f} - {time_to_str(timer()-start_time, mode="min")}')
    if args.use_wandb:
        wandb.log({'test_acc': test_acc})

    final_row = {
        'epoch': f'Best@{best_epoch}',
        'train_loss': round(best_train_loss, 4),
        'train_acc': round(best_train_acc, 2),
        'valid_loss': round(best_valid_loss, 4),
        'valid_acc': round(best_valid_acc, 2),
        'test_acc': round(test_acc, 2)
    }
    result_df = pd.concat([result_df, pd.DataFrame([final_row])], ignore_index=True)
    
    os.makedirs(f'{args.ckpt_root}/{args.run_name}/', exist_ok=True)
    result_df.to_csv(f'{args.ckpt_root}/{args.run_name}/log.csv', index=False)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2024 SCI Steganalysis Parser')
    # Model Parsers
    parser.add_argument('--model_name', type=str, default='UCNet', \
                        choices=['UCNet', 'UCNet_WOW', 'SFANet', 'MobileViT_S_CBAM_based_SFA'], help='Model name')
    parser.add_argument('--apply_sfa', action='store_true', help='Whether to apply SFA')
    parser.add_argument('--attention_after', nargs='+', type=int, default=[2,3,4], help='Attention after which block')
    parser.add_argument('--residual_connection', action='store_true', help='Whether to use residual connection')
    parser.add_argument('--trainable_filter', action='store_true', help='Whether to train filter')
    parser.add_argument('--return_features', action='store_true', help='Whether to return features')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')

    # Training Parsers
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default=None, choices=['MultiStep'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers_per_loader', type=int, default=4, help='Number of workers per data loader')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    
    # Dataset Parsers
    parser.add_argument('--data', type=str, default='ALASKA2_Compete', choices=['ALASKA2_Compete', 'ALASKA_v2'], help='Dataset name')
    parser.add_argument('--data_type', type=str, default='JPG', choices=['PNG', 'TIFF', 'JPG'], help='Data type')
    parser.add_argument('--data_size', type=str, default='512', choices=['256', '512'], help='Image size')
    parser.add_argument('--stego_method', type=str, default='nsf5_0.4', help='Stego method')
    parser.add_argument('--qf', type=str, default='75')
    parser.add_argument('--sample_pair', type=int, default=0)
    parser.add_argument('--normalize', type=str, default=None, choices=['default', 'imagenet'], help='Normalization method')

    # MISC Parsers
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to the csv')
    parser.add_argument('--dataset_root', type=str, default='../../../Data/ALASKA2/', help='Path to the dataset')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--ckpt_root', type=str, default='./results/', help='Root directory for checkpoint and log file saving')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')
    parser.add_argument('--run_name', type=str, default='auto', help='Run name for checkpoint saving')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for the run name')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')

    args = parser.parse_args()

    if args.gpus != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    if torch.cuda.is_available():
        print(f'Using GPU: {args.gpus}')
    else:
        print('Using CPU')
        
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.use_wandb:
        wandb.init(project='2025_SCI', entity='kumdingso')
        wandb.save(f'./utils/datasets.py', policy='now')
        wandb.save(f'./train.py', policy='now')
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)
    main(args)
