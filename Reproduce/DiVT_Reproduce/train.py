import sys
# sys.path.append('../../')
# sys.path.append('./')
 
import os
import argparse
import wandb
import time
import timm
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from arch.DiViT import DiViT
from utils.utils import save_checkpoint, AverageMeter, Logger, OvRScore, time_to_str
from utils.get_loader import get_dataset
from utils.losses import DiCLoss
from timeit import default_timer as timer
from config import configC, configM, configI, configO

def train(config, args, device):
    if args.use_wandb:
        wandb.init(project='DiViT_Reproduce', entity='kumdingso')

        if args.run_name != '':
            wandb.run.name = f'{args.run_name}_to_{args.config}'
        else:
            wandb.run.name = f'DiViT_Reproduce_to_{args.config}'
    else:
        pass

    os.makedirs(args.logs, exist_ok=True)
    
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    tgt_dataloader = get_dataset(config.src1_data, config.src1_train_num_frames, config.src2_data, config.src2_train_num_frames,
                                    config.src3_data, config.src3_train_num_frames, config.tgt_data, config.tgt_test_num_frames, batch_size=10)
    
    train_dia_loss = AverageMeter()
    train_dic_loss = AverageMeter()
    train_total_loss = AverageMeter()
    
    log = Logger()
    log.open(args.logs + config.tgt_data + '_log_DiViT.txt', mode='w')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write(
        '--------------|----------------- TRAIN -----------------|-------------- TEST -------------|-------- BEST -------|--------------|\n')
    log.write(
        '     ITER     |       DiA         DiC         Loss      |     DiA        HTER       AUC   |    ITER      HTER   |     time     |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    
    encoder = torch.load(f'./ckpt/mobilevit_s/mobilevit_s_structure.pt')
    encoder.load_state_dict(torch.load(f'./ckpt/mobilevit_s/mobilevit_s.pt'), strict=True)
    net = DiViT(encoder=encoder, num_classes=3).to(device)
    
    criterion = {
        'DiA': nn.CrossEntropyLoss().cuda(),
        'DiC': DiCLoss().cuda(),
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": args.init_lr},
    ]
    optimizer = optim.Adam(optimizer_dict, lr=args.init_lr, weight_decay=args.weight_decay)

    if(len(args.gpu) > 1):
        net = torch.nn.DataParallel(net).cuda()

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)

    best_hter = 1.0
    best_auc = 0.0
    best_iter = 0
    
    max_iter = args.max_iter
    epoch = 1
    iter_per_epoch = 10
    
    for iter_num in range(max_iter+1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
            
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        # Training Phase
        net.train(True)
        optimizer.zero_grad()
        
        src1_img_real, src1_label_real = next(src1_train_iter_real)
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        input1_real_shape = src1_img_real.shape[0]

        src2_img_real, src2_label_real = next(src2_train_iter_real)
        src2_img_real = src2_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        input2_real_shape = src2_img_real.shape[0]

        src3_img_real, src3_label_real = next(src3_train_iter_real)
        src3_img_real = src3_img_real.cuda()
        src3_label_real = src3_label_real.cuda()
        input3_real_shape = src3_img_real.shape[0]

        src1_img_fake, src1_label_fake = next(src1_train_iter_fake)
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        src2_img_fake, src2_label_fake = next(src2_train_iter_fake)
        src2_img_fake = src2_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_fake, src3_label_fake = next(src3_train_iter_fake)
        src3_img_fake = src3_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
        input3_fake_shape = src3_img_fake.shape[0]
        
        input_data = torch.cat([src1_img_real, src1_img_fake, src2_img_real, src2_img_fake, src3_img_real, src3_img_fake], dim=0)

        source_label = torch.cat([src1_label_real, src1_label_fake,
                                  src2_label_real, src2_label_fake,
                                  src3_label_real, src3_label_fake], dim=0)
        ######### forward #########
        outputs, features = net(input_data)
        
        ######### loss #########
        dia_loss = criterion["DiA"](outputs, source_label)
        dic_loss = criterion["DiC"](features, source_label)
        total_loss = dia_loss + args.dic_rate * dic_loss
        
        train_dia_loss.update(dia_loss.item())
        train_dic_loss.update(dic_loss.item())
        train_total_loss.update(total_loss.item())
        
        ######### backward #########
        total_loss.backward()
        optimizer.step()

        # Test Phase
        net.eval()

        test_probs = []
        test_labels = []
        test_dia_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tgt_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs, _ = net(inputs)
                
                dia_loss = criterion['DiA'](outputs, labels)
                test_dia_loss += dia_loss.item()
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                test_probs.extend(probs)
                test_labels.extend(labels.detach().cpu().numpy())

        test_dia_loss /= len(tgt_dataloader)
        test_probs = np.array(test_probs)
        test_labels = np.array(test_labels)
        
        test_auc, test_hter = OvRScore(test_probs, test_labels)

        is_best= best_hter >= test_hter
        if is_best:
            best_hter = test_hter
            best_iter = iter_num

        log.write('     %4i     |     %5.3f      %7.3f      %7.3f     |   %7.3f    %6.2f    %6.2f   |    %4i     %6.2f  | %s' % (
            iter_num,
            train_dia_loss.avg,
            train_dic_loss.avg,
            train_total_loss.avg,
            test_dia_loss,
            test_hter*100,
            test_auc*100,
            best_iter,
            best_hter*100,
            time_to_str(timer() - start, 'min')))
        log.write('\n') 
        
        if is_best and args.save_models:
            save_checkpoint(state={
                'iter': iter_num,
                'state_dict': net.state_dict(),
                'best_hter': best_hter
                }, gpus=args.gpu, is_best=is_best,
                model_path=args.ckpt_root + f'/{args.run_name}_to_{args.config}/',
                model_name=f'iter{iter_num}.pth.tar')

        if args.use_wandb:
            wandb.log({
                "train_dia_loss": train_dia_loss.avg,
                "train_dic_loss": train_dic_loss.avg,
                "train_loss": train_total_loss.avg,
                "test_dia_loss": test_dia_loss,
                'test_hter': test_hter,
                'test_auc': test_auc,
                'lr':optimizer.param_groups[0]["lr"]},
                step=epoch)
            
        time.sleep(0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Settings
    parser.add_argument('--encoder', type=str, default='./ckpt/cvnets/mobilevit_s.pt')

    # Training Settings
    parser.add_argument('--max_iter', type=int, default=4000)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay for SGD or Adam, ...')
    parser.add_argument('--dic_rate', type=float, default=0.2, help='DiC Loss rate for backward')
    parser.add_argument('--config', type=str)
    # MISC Settings
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/')
    parser.add_argument('--logs', type=str, default='./logs/')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--save_models', action='store_true')
    parser.add_argument('--run_name', type=str, default='DiViT_Reproduce')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
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

    # If you select I, Trainig Data is OULU, MSU, CASIA and Test Data is IDIAP
    if args.config == 'I':
        config = configI
    if args.config == 'C':
        config = configC
    if args.config == 'M':
        config = configM
    if args.config == 'O':
        config = configO

    train(config, args, device)
