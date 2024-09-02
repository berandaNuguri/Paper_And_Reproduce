import os
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score

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

def cal_accuracy(outputs, label):
    # output = torch.argmax(output, dim=1)
    # output = (output >= 0.5).float()
    acc = accuracy_score(label.to('cpu'), outputs.to('cpu').detach().numpy()) * 100
    return acc

def cal_auc(outputs, label):
    # positive_preds = output[:, 1]
    # positive_preds = (output >= 0.5).float()

    auc_score = roc_auc_score(label.to('cpu'), outputs.to('cpu').detach().numpy()) * 100

    return auc_score

def cal_f1score(outputs, label):
    # output = torch.argmax(output, dim=1)
    # output = (output >= 0.5).float()

    f1_score_value = f1_score(label.to('cpu'), outputs.to('cpu').detach().numpy(), average='binary') * 100
    return f1_score_value

def to_np(x):
    return x.detach().cpu().data.numpy()

def save_checkpoint(model, optimizer, best_acc, is_best, epoch, cross_dataset, args):
    state = {
        'model': model.state_dict(),
        'model_name': f'SiameseNet_{args.backbone}',
        'optimizer': optimizer.state_dict(),
        'batch_size': args.batch_size,
        'best_acc': best_acc,
        'epoch': epoch
    }

    os.makedirs(f'{args.ckpt_root}/SiameseNet_{args.backbone}_{args.loss}_Cross_{args.cross}_{args.dataset}_to_{cross_dataset}_{args.aug}', exist_ok=True)
    
    if args.model_save_step != 0 and not is_best:
        filename = os.path.join(args.ckpt_root, f'SiameseNet_{args.backbone}_{args.loss}_Cross_{args.cross}_{args.dataset}_to_{cross_dataset}_{args.aug}/epoch_{state["epoch"]}.pth')
        torch.save(state, filename)
    elif is_best:
        filename = os.path.join(args.ckpt_root, f'SiameseNet_{args.backbone}_{args.loss}_Cross_{args.cross}_{args.dataset}_to_{cross_dataset}_{args.aug}/model_best.pth')
        torch.save(state, filename)

def load_checkpoint(model, optimizer, pretrained_path, device):
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state['model'])
    batch_size = state['batch_size']
    best_acc = state['best_acc']
    epoch = state['epoch']

    print(f'\t## loaded trained models (epoch: {epoch})\n')
    return model, optimizer, batch_size, best_acc, epoch