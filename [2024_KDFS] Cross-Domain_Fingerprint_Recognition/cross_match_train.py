import torch
import torch.nn.functional as F
import os
import time
import random
import cv2
import pandas as pd
import numpy as np
import wandb
import timm
import shutil
import warnings
import torchvision
import argparse
from glob import glob

from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchmetrics

from arch.architectures import SiameseNetwork, AFRNet, IAESNet, CASAResnet
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint, cal_accuracy, cal_auc, cal_f1score
from utils.losses import ArcMarginProduct, AddMarginProduct
from utils.datasets import DatasetRetriever
from utils.transforms_config import get_transforms

def train(train_loader, model, arc_fc, optimizer, criterion, epoch, device, args):
    model.train()

    with torch.enable_grad():
        train_acc, train_auc, train_f1, train_loss = iteration('train', train_loader, model, arc_fc, optimizer, criterion, epoch, device, args)

    return train_acc, train_auc, train_f1, train_loss

def validate(val_loader, model, arc_fc, optimizer, criterion, epoch, device, args):
    model.eval()

    with torch.no_grad():
        val_acc, val_auc, val_f1, val_loss = iteration('val', val_loader, model, arc_fc, optimizer,  criterion, epoch, device, args)

    return val_acc, val_auc, val_f1, val_loss

def iteration(mode, data_loader, model, arc_fc, optimizer, criterion, epoch, device, args):
    am_batch_time = AverageMeter()
    am_data_time = AverageMeter()
    am_loss = AverageMeter()
    am_acc = AverageMeter()
    am_auc = AverageMeter()
    am_f1 = AverageMeter()
    end = time.time()
    num_batch = np.ceil(len(data_loader)).astype(np.int32)

    for i, (input_img1, input_img2, label) in enumerate(data_loader):
        am_data_time.update(time.time() - end)
        input_img1 = input_img1.to(device, dtype=torch.float)
        input_img2 = input_img2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        if args.loss == 'arcface' or args.loss == 'addmargin':
            _, features = model(input_img1, input_img2)           
            logits = arc_fc(features, label)
            outputs = torch.softmax(logits, dim=1)
        else:
            logits, features = model(input_img1, input_img2)
            outputs = torch.softmax(logits, dim=1)

        loss = criterion(logits, label)

        am_loss.update(loss.item(), input_img1.size(0))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        outputs = (outputs[:, 1] > 0.5).float()

        class_acc = cal_accuracy(outputs, label)
        am_acc.update(class_acc, input_img1.size(0))
        class_auc = cal_auc(outputs, label.squeeze())
        am_auc.update(class_auc, input_img1.size(0))
        class_f1 = cal_f1score(outputs, label.squeeze())
        am_f1.update(class_f1, input_img1.size(0))

        am_batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.log_step == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Accuracy {acc.val:.2f} ({acc.avg:.2f}) '
                  'AUC {auc.val:.2f} ({auc.avg:.2f}) '
                  'F1 {f1.val:.2f} ({f1.avg:.2f}) '
                  .format(epoch + 1, args.num_epochs, i + 1, num_batch,
                          batch_time=am_batch_time, data_time=am_data_time,
                          loss=am_loss, acc=am_acc, auc=am_auc, f1=am_f1))

    return am_acc.avg, am_auc.avg, am_f1.avg, am_loss.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for choosing augmentation method and dataset")
    # Model Settings
    parser.add_argument('--backbone', default='efficientnet_b0', help='choose model name')
    parser.add_argument('--pretrained_path', default=None, help='pretrained model name')
    parser.add_argument('--global_pool', action='store_true', help='whether use global pool')
    
    # Training Settings
    parser.add_argument('--aug', default='Base', help='choose augmentation name')
    parser.add_argument('--dataset', default='Dermalog', help='choose dataset')
    parser.add_argument('--loss', default='ce', help='choose loss function')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--image_size', default=500, type=int, help='image size')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--cross', action='store_true', help='whether process cross match')
    parser.add_argument('--gpu', default='0', type=str, help='gpu number')
    
    # MISC Settings
    parser.add_argument('--dataset_root', default='./data/', help='Root foloder of dataset')
    parser.add_argument('--txt_root', default='./data/train_txt/', help='text root folder')
    parser.add_argument('--ckpt_root', default='./ckpt/', help='checkpoint directory')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--log_step', default=20, type=int, help='log step')
    parser.add_argument('--model_save_step', default=0, type=int, help='model save step')
    parser.add_argument('--use_wandb', action='store_true', help='whether use wandb')
    
    args = parser.parse_args()

    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    warnings.filterwarnings(action='ignore')
       
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.use_wandb:
        wandb.init(project="2023_KCI_FP_CrossMatch", entity="kumdingso", config=args)
    
    if args.cross:
        if args.dataset == 'Dermalog':
            dataset = args.dataset
            cross_dataset = 'Greenbit'
        elif args.dataset == 'Greenbit':
            dataset = args.dataset
            cross_dataset = 'Dermalog'
    else:
        if args.dataset == 'Dermalog':
            dataset = args.dataset
            cross_dataset = 'Dermalog'
        elif args.dataset == 'Greenbit':
            dataset = args.dataset
            cross_dataset = 'Greenbit'
    
    train_probe_path = f'{args.txt_root}' + dataset + '/' + dataset + '_probe_train.txt'
    train_template_path = f'{args.txt_root}' + dataset + '/' + dataset + '_template_train.txt'
    valid_probe_path = f'{args.txt_root}' + cross_dataset + '/' + cross_dataset + '_probe_train.txt'
    valid_template_path = f'{args.txt_root}' + cross_dataset + '/' + cross_dataset + '_template_train.txt'

    if args.use_wandb:
        wandb.run.name = f'SiameseNet_{args.backbone}_{dataset}_to_{cross_dataset}_{args.aug}'
        wandb.save('./cross_match_train.py')
        wandb.save('./utils/transforms_config.py')
        for arg in vars(args):
            wandb.config[arg] = getattr(args, arg)

    train_probeset = []
    with open(train_probe_path, 'r', encoding='UTF-8') as train_probe_text:
        text = train_probe_text.readlines()
        for t in text:
            train_probeset.append({
                'image_path':t.split('\t')[0],
                'probe_label':t.split('\t')[1].strip(),
                'template_label':t.split('\t')[2].strip()
            })

    train_templateset = []
    with open(train_template_path, 'r', encoding='UTF-8') as train_template_text:
        text = train_template_text.readlines()
        for t in text:
            train_templateset.append({
                'image_path':t.split('\t')[0],
                'template_label':t.split('\t')[1].strip()
            })

    valid_probeset = []
    with open(valid_probe_path, 'r', encoding='UTF-8') as valid_probe_text:
        text = valid_probe_text.readlines()
        for t in text:
            valid_probeset.append({
                'image_path':t.split('\t')[0],
                'probe_label':t.split('\t')[1].strip(),
                'template_label':t.split('\t')[2].strip()
            })

    valid_templateset = []
    with open(valid_template_path, 'r', encoding='UTF-8') as valid_template_text:
        text = valid_template_text.readlines()
        for t in text:
            valid_templateset.append({
                'image_path':t.split('\t')[0],
                'template_label':t.split('\t')[1].strip()
            })

    train_probeset = pd.DataFrame(train_probeset)
    train_templateset = pd.DataFrame(train_templateset)
    valid_probeset = pd.DataFrame(valid_probeset)
    valid_templateset = pd.DataFrame(valid_templateset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device : ', device)

    train_transform, valid_transform = get_transforms(args.aug)

    train_dataset = DatasetRetriever(
        image_names1=train_probeset[:].image_path.values,
        image_names2=train_templateset[:].image_path.values,
        labels1=train_probeset[:].template_label.values,
        labels2=train_templateset[:].template_label.values,
        transforms=train_transform,
    )

    validation_dataset = DatasetRetriever(
        image_names1=valid_probeset[:].image_path.values,
        image_names2=valid_templateset[:].image_path.values,
        labels1=valid_probeset[:].template_label.values,
        labels2=valid_templateset[:].template_label.values,
        transforms=valid_transform,
    )

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
    )
    
    
    if args.backbone == 'afr_net':
        backbone = AFRNet(backbone=timm.create_model('resnet50', in_chans=1, pretrained=True), in_chans=1, num_classes=args.num_classes)
    else:
        backbone = timm.create_model(f'{args.model_name}', num_classes=args.num_classes, in_chans=1, pretrained=True)
    # model = AFRNet(backbone=timm.create_model('resnet50', in_chans=1, pretrained=True), in_chans=1, num_classes=args.num_classes).to(device)
    model = SiameseNetwork(backbone=backbone, num_classes=args.num_classes, global_pool=args.global_pool).to(device)


    # criterion = nn.BCELoss()
    criterion = CrossEntropyLoss()
    if args.loss == 'arcface':
        metric_fc = ArcMarginProduct(in_features=384, out_features=args.num_classes, s=30.0, m=0.5, easy_margin=False).to(device)
        optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                            lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)
    elif args.loss == 'addmargin':
        metric_fc = AddMarginProduct(in_features=384, out_features=args.num_classes, s=30.0, m=0.5).to(device)
        optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=args.lr, betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)
    else:
        metric_fc = None
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr, betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)

    if args.pretrained_path is not None:
        pretrained_path = os.path.join(args.pretrained_path)
        model, optimizer, best_acc, initial_epoch = load_checkpoint(model, optimizer, pretrained_path, device)
    else:
        initial_epoch = 0
        best_acc = 0

    initial_epoch = 0

    min_loss = 100000
    best_train_acc = 0
    best_val_acc = 0
    best_train_auc = 0
    best_val_auc = 0
    best_train_f1 = 0
    best_val_f1 = 0

    for epoch in range(initial_epoch, args.num_epochs):
        print('# Training')
        train_acc, train_auc, train_f1, train_loss = train(train_loader, model, metric_fc, optimizer, criterion, epoch, device, args)
        if args.use_wandb:
            wandb.log({'Train_Acc': train_acc, 'Train_Auc': train_auc, 'Train_F1': train_f1,'Train_loss': train_loss, 'Epoch': epoch+1})
        
        # evaluate on validation set
        print('# Validation')
        val_acc, val_auc, val_f1, val_loss = validate(val_loader, model, metric_fc, optimizer, criterion, epoch, device, args)
        if args.use_wandb:
            wandb.log({'Valid_Acc': val_acc, 'Valid_Auc' : val_auc, 'Valid_F1':val_f1, 'Valid_loss': val_loss, 'Epoch': epoch+1})

        is_best = val_acc > best_val_acc

        best_train_acc = max(train_acc, best_train_acc)
        best_val_acc = max(val_acc, best_val_acc)
        best_train_auc = max(train_auc, best_train_auc)
        best_val_auc = max(val_auc, best_val_auc)
        best_train_f1 = max(train_f1, best_train_f1)
        best_val_f1 = max(val_f1, best_val_f1)
        min_loss = min(val_loss, min_loss)

        if args.use_wandb:
            wandb.log({'Best_Train_Acc': best_train_acc, 'Best_Valid_Acc': best_val_acc,
                    'Best_Train_AUC': best_train_auc, 'Best_Valid_AUC': best_val_auc,
                    'Best_Train_F1': best_train_f1, 'Best_Valid_F1': best_val_f1})

        if args.model_save_step != 0:
            if (epoch + 1) % args.model_save_step == 0:
                save_checkpoint(model, optimizer, best_val_acc, is_best, epoch + 1, cross_dataset, args)
        if is_best:
            save_checkpoint(model, optimizer, best_val_acc, is_best, epoch + 1, cross_dataset, args)

        torch.cuda.empty_cache()
      
    if args.use_wandb:  
        wandb.finish()
