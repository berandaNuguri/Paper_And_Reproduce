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

from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
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


def now_time():
    return datetime.today().strftime("%y%m%d%H%M")

warnings.filterwarnings(action='ignore')

dataset_name = 'Dermalog'
name_of_model = 'efficientnet_b0'
args = {
    'type': 'Probe',
    'model_name': name_of_model,  # 신경망 구조
    'lr': 1e-4,  # 학습률
    'weight_decay': 1e-4,  # 가중치 감쇠
    'drop_rate': 0,  # 학습 시 dropout 비율
    'image_size': 500,  # 이미지 크기
    'num_epochs': 15,  # 학습 반복수
    'batch_size': 64,  # 미니배치 크기
    'num_classes': 1,  # 판별할 클래스 개수
    'num_folds': 5,  # 데이터셋 분할 fold 개수
    'val_fold': 1,  # 검증용 fold 선택
    'seed': 10,  # 랜덤 seed 설정
    'log_step': 3,  # log 남길 iteration 반복 수
    'model_save_step': 0,  # model 저장할 epoch 반복 수, 0 은 저장안함
    'workspace_path': '../data/dataset_FL/' + dataset_name + '_FL',  # 작업 위치
    'checkpoint_dir': 'checkpoints',  # 모델 저장 디렉토리
    'pretrained_name': dataset_name + '_' + name_of_model + '_best.pth',  # 학습한 모델 파일이름 (.pth까지 붙이기)
    'pretrained': True
}

print(args['workspace_path'])

TRAIN_DATA_ROOT_PATH = os.path.join(args['workspace_path'])
# TEST_DATA_ROOT_PATH = os.path.join(args['workspace_path'], 'test')

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
    'DropRate' : args['drop_rate'],
    'Seed' : args['seed'],
    'Pretrained' : args['pretrained']
}

wandb.init(project="LivDet2023", entity="kumdingso", config=wandb_config)
wandb.run.name = str(now_time()) + '_' + str(dataset_name)

dataset = []

# 0: Fake, 1: Live
for label, kind in enumerate(['Fake', 'Live']):
    for path in glob(f'{TRAIN_DATA_ROOT_PATH}/{kind}/*.jpg', recursive=True):
        dataset.append({
            'kind': kind,
            'image_name': path.replace('\\', '/').split('/')[-1],
            'label': label
        })

random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

skf = StratifiedKFold(n_splits=args['num_folds'])

dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(skf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number

print(dataset.head())

class DatasetRetriever(Dataset):
    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image_path = f'{TRAIN_DATA_ROOT_PATH}/{kind}/{image_name}'.replace('\\', '/')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image, label

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)

def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            ToTensorV2(p=1.0),
        ], p=1.0)

train_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] != args['val_fold']].kind.values,
    image_names=dataset[dataset['fold'] != args['val_fold']].image_name.values,
    labels=dataset[dataset['fold'] != args['val_fold']].label.values,
    transforms=get_train_transforms(),
)

validation_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] == args['val_fold']].kind.values,
    image_names=dataset[dataset['fold'] == args['val_fold']].image_name.values,
    labels=dataset[dataset['fold'] == args['val_fold']].label.values,
    transforms=get_valid_transforms(),
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=args['batch_size'],
        pin_memory=False,
        drop_last=False,
    )
val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )

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

loss_func = nn.BCEWithLogitsLoss().to(device)

def get_net():
    net = timm.create_model(args['model_name'], num_classes=args['num_classes'], in_chans = 1, pretrained=args['pretrained'])
    return net

model = get_net().to(device)

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
        filename = os.path.join(checkpoint_path, f'{dataset_name}_{model_name}_best.pth')
        torch.save(state, filename)


def load_checkpoint(model, optimizer, pretrained_path, device):
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state['model'])
    batch_size = state['batch_size']
    best_acc = state['best_acc']
    epoch = state['epoch']

    print(f'\t## loaded trained models (epoch: {epoch})\n')
    return model, optimizer, batch_size, best_acc, epoch

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=args['lr'], betas=(0.9, 0.999),
                              weight_decay=args['weight_decay'])

# if args['pretrained_name']:
#     pretrained_path = os.path.join(args['checkpoint_dir'], args['pretrained_name'])
#     model, optimizer, best_acc, initial_epoch = load_checkpoint(model, optimizer, pretrained_path, device)
# else:
#     initial_epoch = 0
#     best_acc = 0
initial_epoch = 0
best_acc = 0

def train(train_loader, model, *args):
    # switch to train mode
    model.train()

    with torch.enable_grad():
        train_acc, train_auc, train_loss = iteration('train', train_loader, model, *args)

    return train_acc, train_auc, train_loss

def validate(val_loader, model, *args):
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        val_acc, val_auc, val_loss = iteration('val', val_loader, model, *args)

    return val_acc, val_auc, val_loss

def iteration(mode, data_loader, model, optimizer, loss_func, epoch):
    am_batch_time = AverageMeter()
    am_data_time = AverageMeter()
    am_loss = AverageMeter()
    am_acc = AverageMeter()
    am_auc = AverageMeter()

    end = time.time()
    num_batch = np.ceil(len(data_loader)).astype(np.int32)

    for i, (input_img, target) in enumerate(data_loader):
        # measure data loading time
        am_data_time.update(time.time() - end)

        input_img = input_img.to(device)
        target = target.unsqueeze(1)
        target = target.float()
        target = target.to(device)
        
        # feed-forward
        output = model(input_img)   # two output
        # calculate loss
        output = torch.nan_to_num(output)
        loss = loss_func(output, target)
        am_loss.update(loss.item(), input_img.size(0))

        # calculate accuracy
        class_prob = F.sigmoid(output)
        probs = []
        for prob in class_prob:
            if prob > 0.5:
                probs.append(1)
            elif prob < 0.5:
                probs.append(0)
        class_acc = cal_accuracy(probs, target)
        am_acc.update(class_acc, input_img.size(0))
        
        # calculate AUC
        class_auc = cal_auc(class_prob, target)
        am_auc.update(class_auc, input_img.size(0))

        # compute gradient and do SGD step
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # measure elapsed time
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

min_loss = 1000
for epoch in range(initial_epoch, args['num_epochs']):
    # train for one epoch
    print('# Training')
    train_acc, train_auc, train_loss = train(train_loader, model, optimizer, loss_func, epoch)
    wandb.log({'Train_Acc': train_acc, 'Train_Auc': train_auc, 'Train_loss': train_loss, 'Epoch': epoch+1})

    # evaluate on validation set
    print('# Validation')
    val_acc, val_auc, val_loss = validate(val_loader, model, optimizer, loss_func, epoch)
    wandb.log({'Valid_Acc': val_acc, 'Valid_Auc' : val_auc, 'Valid_loss': val_loss, 'Epoch': epoch+1})

    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)
    min_loss = min(val_loss, min_loss)
    
    if args['model_save_step'] != 0:
        if (epoch + 1) % args['model_save_step'] == 0:
            save_checkpoint(model, optimizer, best_acc, args['checkpoint_dir'], args['model_name'], args['model_save_step'], is_best, epoch + 1)
    if is_best:
        save_checkpoint(model, optimizer, best_acc, args['checkpoint_dir'], args['model_name'], args['model_save_step'], is_best, epoch + 1)
