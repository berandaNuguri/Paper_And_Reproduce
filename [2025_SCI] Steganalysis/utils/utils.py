import os
import random
import shutil
import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from arch.iformer.inception_transformer import iformer_small, iformer_base
from arch.SFA.MobileViT_SFA import MobileViT_SFA
from arch.SFA.MobileViT_CBAM_based_SFA import MobileViT_CBAM_based_SFA
from arch.SFA.SFANet import SFANet
from arch.ucnet.ucnet import UCNet
from arch.ucnet.ucnet_wow import UCNet_WOW
from utils.datasets import SFADataset

def save_checkpoint(state, gpus, is_best=False,
                    model_path = f'./ckpt/',
                    model_name = f'checkpoint.pth.tar'):
    
    if(len(gpus) > 1):
        old_state_dict = state['state_dict']
        
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1):
                k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict

    os.makedirs(model_path, exist_ok=True)
    torch.save(state, f'{model_path}/{model_name}')
    if is_best:
        shutil.copyfile(f'{model_path}/{model_name}', f'{model_path}/model_best.pth.tar')

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)
    
def load_dataset(args):
    if args.normalize == 'imagenet':
        transform = {
            'train': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            'test': A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        }
    else:
        transform = {
            'train': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Lambda(image=lambda x, **kwargs: x.astype(np.float32)),
                ToTensorV2(),
            ]),
            'test': A.Compose([
                A.Lambda(image=lambda x, **kwargs: x.astype(np.float32)),
                ToTensorV2(),
            ])
        } 

    dataset_key = f'{args.data}_{args.data_type}_{args.data_size}'
    
    if args.data == 'ALASKA2_Compete':
        dataset_root = os.path.join(args.dataset_root, 'Competition')
        folder = 'ALASKA2_Compete'
        prefix = f'{args.stego_method}_QF_{args.qf}'
    elif 'ALASKA_v2_TIFF' in dataset_key or 'ALASKA_v2_JPG' in dataset_key:
        folder = f'ALASKA_v2_{args.data_type}_{args.data_size}_QF{args.qf}_COLOR'
        dataset_root = os.path.join(args.dataset_root, folder)
        prefix = f'{args.stego_method}_QF_{args.qf}'

    else:
        raise ValueError("Unsupported dataset configuration.")

    if args.sample_pair > 0:
        train_csv = f'{args.csv_root}/{folder}/{args.sample_pair}/{prefix}_train.csv'
        valid_csv = f'{args.csv_root}/{folder}/{args.sample_pair}/{prefix}_val.csv'
        test_csv  = f'{args.csv_root}/{folder}/{args.sample_pair}/{prefix}_test.csv'
    else:
        train_csv = f'{args.csv_root}/{folder}/{prefix}_train.csv'
        valid_csv = f'{args.csv_root}/{folder}/{prefix}_val.csv'
        test_csv  = f'{args.csv_root}/{folder}/{prefix}_test.csv'
        
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df  = pd.read_csv(test_csv)

    train_dataset = SFADataset(train_df, dataset_root, transform=transform['train'])
    valid_dataset = SFADataset(valid_df, dataset_root, transform=transform['test'])
    test_dataset  = SFADataset(test_df, dataset_root, transform=transform['test'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, 
                              shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, 
                              shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)    
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers_per_loader, 
                              shuffle=False, drop_last=False, worker_init_fn=worker_init_fn)

    return train_loader, valid_loader, test_loader

def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)

    if type(module) == nn.GroupNorm:
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)
        
def load_model(args, device):
    if args.model_name == 'MobileViT_S_SFA':
        print('Load MobileViT_S_SFA using following arguments')
        print(args)
        model = MobileViT_SFA(num_classes=2, apply_sfa=args.apply_sfa, attention_type=args.attention_type, \
                              rho_scaler=args.rho_scaler, return_features=args.return_features, attention_after=args.attention_after)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)
            
    elif args.model_name == 'MobileViT_S_CBAM_based_SFA':
        print('Load MobileViT_S_CBAM_based_SFA using following arguments')
        print(args)
        model = MobileViT_CBAM_based_SFA(num_classes=2, apply_sfa=args.apply_sfa, return_features=args.return_features, \
                                         attention_after=args.attention_after, residual_connection=args.residual_connection, trainable_filter=args.trainable_filter)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif args.model_name == 'SFANet':
        print('Load SFANet using following arguments')
        print(args)
        model = SFANet(num_classes=2, residual_connection=args.residual_connection, trainable_filter=args.trainable_filter)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'UCNet' in args.model_name:
        print('Load UCNet_WOW using following arguments')
        print(args)
        if args.model_name.split('_')[-1] == 'UCNet':
            model = UCNet(num_classes=2)
        elif args.model_name.split('_')[-1] == 'WOW':
            model = UCNet_WOW(num_classes=2, trainable_filter=args.trainable_filter)
        model.apply(initWeights)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)
    return model.to(device) 