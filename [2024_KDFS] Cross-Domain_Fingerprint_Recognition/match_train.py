import torch
import torch.nn.functional as F
import os
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import wandb
import timm
import shutil
import warnings
import torchvision
import argparse

from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch import nn
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy
from datetime import datetime
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from catalyst.data.sampler import BalanceClassSampler

class DatasetRetriever(Dataset):
    def __init__(self, image_names1, image_names2, labels1, labels2 , transforms=None):
        super().__init__()
        self.image_names1 = image_names1
        self.image_names2 = image_names2
        self.labels1 = labels1
        self.labels2 = labels2
        self.transforms = transforms

    def __getitem__(self, index: int):
        if 0.5 > random.random():
            index2 = index
        else : 
            index2 = random.randint(0,len(self.image_names2)-1)
        image_name1, image_name2,  label1, label2 = self.image_names1[index], self.image_names2[index2], self.labels1[index], self.labels2[index2]
        image_path1 = image_name1#f'{TRAIN_DATA_ROOT_PATH}/{kind}/{image_name1}'.replace('\\', '/')
        image_path2 = image_name2#f'{TRAIN_DATA_ROOT_PATH}/{kind}/{image_name2}'.replace('\\', '/')
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label = label1==label2
        #print(label1, label2, label)
        image1 /= 255.0
        image2 /= 255.0

        if self.transforms:
            image1 = self.transforms(image=image1)['image']
            image2 = self.transforms(image=image2)['image']
            
            # sample = {'image': image1}
            # sample = self.transforms(**sample)
            # image1 = sample['image']
            
            # sample = {'image' : image2}
            # sample = self.transforms(**sample)
            # image2 = sample['image'] 

        return image1, image2, label

    def __len__(self) -> int:
        return self.image_names1.shape[0]

    #def get_labels(self):
    #    return list(self.labels1), list(self.labels2)

def get_train_transforms():
    return A.Compose([
            A.Resize(500,500),
            # A.RandomBrightness(limit=0.5, p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
           A.Resize(500,500),
           ToTensorV2(p=1.0),
        ], p=1.0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_accuracy(output, label):
    acc = accuracy_score(label.to('cpu'), output)*100

    return acc

def cal_auc(output, label):
    fpr, tpr, thresholds = roc_curve(label.to('cpu'), output.to('cpu').detach().numpy(), pos_label=1)
    auc_score = auc(fpr, tpr)*100
    return auc_score

def to_np(x):
    return x.detach().cpu().data.numpy()

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def get_net():
    net = timm.create_model('efficientnet_b0', num_classes=args['num_classes'], in_chans = 1, pretrained=args['pretrained'])
    # net = nn.DataParallel(net) 

    return net

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = get_net()
        
        self.n_features = self.model.classifier.in_features

        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        self.classifier = nn.Sequential(
            nn.Linear(self.n_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.model.apply(self.init_weights)
        self.classifier.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.cat((output1, output2), 1)

        output = self.classifier(output)

        output = self.sigmoid(output)
        
        return output   


def save_checkpoint(model, optimizer, best_acc, checkpoint_path, model_name, model_save_step, is_best, epoch):
    state = {
        'model': model.state_dict(),
        'model_name':model_name,
        'optimizer': optimizer.state_dict(),
        'batch_size': args['batch_size'],
        'best_acc': best_acc,
        'epoch': epoch
    }

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    if model_save_step != 0 and not is_best:
        filename = os.path.join(checkpoint_path, f'{dataset_name}_{model_name}_epoch_{state["epoch"]}.pth')
        torch.save(state, filename)
    elif is_best:
        filename = os.path.join(checkpoint_path, f'Match_{dataset_name}_{model_name}_best.pth')
        torch.save(state, filename)


def load_checkpoint(model, optimizer, pretrained_path, device):
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state['model'])
    batch_size = state['batch_size']
    best_acc = state['best_acc']
    epoch = state['epoch']

    print(f'\t## loaded trained models (epoch: {epoch})\n')
    return model, optimizer, batch_size, best_acc, epoch


def train(train_loader, model, args):
    model.train()

    with torch.enable_grad():
        train_acc, train_auc, train_loss = iteration('train', train_loader, model, *args)

    return train_acc, train_auc, train_loss

def validate(val_loader, model, args):
    model.eval()

    with torch.no_grad():
        val_acc, val_auc, val_loss = iteration('val', val_loader, model, args)

    return val_acc, val_auc, val_loss

def iteration(mode, data_loader, model, optimizer, loss_func, epoch, args):
    am_batch_time = AverageMeter()
    am_data_time = AverageMeter()
    am_loss = AverageMeter()
    am_acc = AverageMeter()
    am_auc = AverageMeter()

    end = time.time()
    num_batch = np.ceil(len(data_loader)).astype(np.int32)

    for i, (input_img1, input_img2, target) in enumerate(data_loader):
        am_data_time.update(time.time() - end)

        input_img1 = input_img1.to(device)
        input_img2 = input_img2.to(device)
        # target = target.unsqueeze(1)
        # target = target.float()
        target = target.to(device)

        output = model(input_img1, input_img2)

        loss = loss_func(output, target)
        am_loss.update(loss.item(), input_img1.size(0))

        class_prob = output
        probs = []
        for prob in class_prob:
            if prob > 0.5:
                probs.append(1)
            elif prob < 0.5:
                probs.append(0)
        class_acc = cal_accuracy(probs, target)
        am_acc.update(class_acc, input_img1.size(0))

        class_auc = cal_auc(class_prob, target)
        am_auc.update(class_auc, input_img1.size(0))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        am_batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args['log_step'] == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Accuracy {acc.val:.2f} ({acc.avg:.2f})  '
                  'AUC {auc.val:.2f} ({auc.avg:.2f})  '
                  .format(epoch + 1, args['num_epochs'], i + 1, num_batch, batch_time=am_batch_time,
                          data_time=am_data_time, loss=am_loss, acc=am_acc, auc=am_auc))

    return am_acc.avg, am_auc.avg, am_loss.avg

if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "1"

    def now_time():
        return datetime.today().strftime("%y%m%d%H%M")

    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser(description="Args for choosing augmentation method and dataset")
    parser.add_argument('--aug', default="None", type=str, help='choose augmentations')
    parser.add_argument('--dataset', default='Dermalog', help='choose dataset')
    parser.add_argument('--cross', action='store_true', help='Whether to train cross test the image(Default: False)')
    parser.add_argument('--dataset_root', default='./data/train_txt/')

    parse = parser.parse_args()

    dataset_name = parse.dataset

    name_of_aug = parse.aug

    args = {
        'type': 'Match',
        'model_name': 'efficientnet_b0',  # 신경망 구조
        'lr': 1e-4,  # 학습률
        'weight_decay': 1e-4,  # 가중치 감쇠
        'image_size': 500,  # 이미지 크기
        'num_epochs': 100,  # 학습 반복수
        'batch_size': 48,  # 미니배치 크기
        'num_classes': 1,  # 판별할 클래스 개수
        'num_folds': 5,  # 데이터셋 분할 fold 개수
        'val_fold': 1,  # 검증용 fold 선택
        'seed': 10,  # 랜덤 seed 설정
        'log_step': 3,  # log 남길 iteration 반복 수
        'model_save_step': 0,  # model 저장할 epoch 반복 수, 0 은 저장안함
        'probe_path': './data/train_txt/' + dataset_name + '/' + dataset_name + '_probe_train.txt',  # probe text 위치
        'template_path': './data/train_txt/' + dataset_name + '/' + dataset_name + '_template_train.txt',  # template text 위치
        'checkpoint_dir': 'checkpoints',  # 모델 저장 디렉토리
        # 'pretrained_name': 'Match_' + dataset_name + '_' + name_of_model + '_best.pth',  # 학습한 모델 파일이름 (.pth까지 붙이기)
        'pretrained' : True
    }

    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    seed_everything(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device : ', device)

    wandb_config = {
        'Type' : args['type'],
        'Dataset' : dataset_name,
        'Model' : args['model_name'],
        'LR' : args['lr'],
        'WeightDecay' : args['weight_decay'],
        'BatchSize' : args['batch_size'],
        # 'DropRate' : args['drop_rate'],
        'Seed' : args['seed'],
        'Pretrained' : args['pretrained']
    }

    # wandb.init(project="LivDet2023", entity="kumdingso", config=wandb_config)

    # wandb.run.name = str(now_time()) + '_' + str(dataset_name)

    probeset = []
    with open(args['probe_path'], 'r', encoding='UTF-8') as probe_text:
        text = probe_text.readlines()

        for t in text:
            probeset.append({
                'image_path':t.split('\t')[0],
                'probe_label':t.split('\t')[1].strip(),
                'template_label':t.split('\t')[2].strip()
            })

    templateset = []
    with open(args['template_path'], 'r', encoding='UTF-8') as template_text:
        text = template_text.readlines()

        for t in text:
            templateset.append({
                'image_path':t.split('\t')[0],
                'template_label':t.split('\t')[1].strip()
            })

    probeset = pd.DataFrame(probeset)
    templateset = pd.DataFrame(templateset)

    kf = KFold(n_splits=args['num_folds'])

    probeset.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(kf.split(X=probeset.index, groups=probeset['image_path'])):
        probeset.loc[probeset.iloc[val_index].index, 'fold'] = fold_number

    templateset.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(kf.split(X=templateset.index, groups=templateset['image_path'])):
        templateset.loc[templateset.iloc[val_index].index, 'fold'] = fold_number

    print(probeset)
    print(templateset)

    train_dataset = DatasetRetriever(
        image_names1=probeset[probeset['fold'] != args['val_fold']].image_path.values,
        image_names2=templateset[templateset['fold'] != args['val_fold']].image_path.values,
        labels1=probeset[probeset['fold'] != args['val_fold']].template_label.values,
        labels2=templateset[templateset['fold'] != args['val_fold']].template_label.values,
        transforms=get_train_transforms(),
    )

    validation_dataset = DatasetRetriever(
        image_names1=probeset[probeset['fold'] == args['val_fold']].image_path.values,
        image_names2=templateset[templateset['fold'] == args['val_fold']].image_path.values,
        labels1=probeset[probeset['fold'] == args['val_fold']].template_label.values,
        labels2=templateset[templateset['fold'] == args['val_fold']].template_label.values,
        transforms=get_valid_transforms(),
    )

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            #sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
            batch_size=args['batch_size'],
            num_workers=16,
            pin_memory=False,
            drop_last=False,
        )
    val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=args['batch_size'],
            num_workers=16,
            shuffle=False,
            #sampler=SequentialSampler(validation_dataset),
            pin_memory=False,
        )
    
    # model = get_net().to(device)
    model = SiameseNetwork().to(device)
    # model = nn.DataParallel(model).to(device)
    loss_func = nn.BCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                              lr=args['lr'], betas=(0.9, 0.999),
                              weight_decay=args['weight_decay'])

    if args['pretrained_name']:
        pretrained_path = os.path.join(args['checkpoint_dir'], args['pretrained_name'])
        model, optimizer, best_acc, initial_epoch = load_checkpoint(model, optimizer, pretrained_path, device)
    else:
        initial_epoch = 0
        best_acc = 0

    min_loss = 1000
    for epoch in range(initial_epoch, args['num_epochs']):
        print('# Training')
        train_acc, train_auc, train_loss = train(train_loader, model, optimizer, loss_func, epoch, args)
        # wandb.log({'Train_Acc': train_acc, 'Train_Auc': train_auc, 'Train_loss': train_loss, 'Epoch': epoch+1})

        print('# Validation')
        val_acc, val_auc, val_loss = validate(val_loader, model, optimizer, loss_func, epoch, args)
        # wandb.log({'Valid_Acc': val_acc, 'Valid_Auc' : val_auc, 'Valid_loss': val_loss, 'Epoch': epoch+1})

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        min_loss = min(val_loss, min_loss)
        
        # if args['model_save_step'] != 0:
        #     if (epoch + 1) % args['model_save_step'] == 0:
        #         save_checkpoint(model, optimizer, best_acc, args['checkpoint_dir'], args['model_name'], args['model_save_step'], is_best, epoch + 1)
        # if is_best:
        #     save_checkpoint(model, optimizer, best_acc, args['checkpoint_dir'], args['model_name'], args['model_save_step'], is_best, epoch + 1)
