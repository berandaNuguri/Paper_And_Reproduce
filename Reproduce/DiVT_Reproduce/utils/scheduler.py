import math

def CosineSchedule(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.init_lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.init_lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def StepLR(optimizer, epoch, init_param_lr, step_1, step_2):
    """Decay the learning rate with StepLR Scheduler"""
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]
        i += 1
        if(epoch <= step_1):
            param_group['lr'] = init_lr * 0.1 ** 0
        elif(epoch <= step_2):
            param_group['lr'] = init_lr * 0.1 ** 1
        else:
            param_group['lr'] = init_lr * 0.1 ** 2