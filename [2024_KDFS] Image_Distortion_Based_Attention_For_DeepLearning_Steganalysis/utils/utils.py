import os
import torch
import shutil
import numpy as np
from collections import OrderedDict

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

def task_mapping(task):
    task_mapping = {
        1: "iPhone7",                   # 2016-10-21
        2: "Galaxy_Note9",              # 2018-11-30
        3: "Galaxy_S10+",               # 2019-03-08
        4: "Huawei_P30",                # 2019-03-26
        5: "Galaxy_Note10",             # 2019-08-23
        6: "iPhone11_Pro",              # 2019-09-20
        7: "Galaxy_S20+",               # 2020-03-06
        8: "LG_Wing",                   # 2020-10-15
        9: "iPhone12",                  # 2020-10-23
        10: "iPhone12_ProMax",          # 2020-11-13
        11: "Galaxy_S21_Ultra",         # 2021-01-29
        12: "Galaxy_Fold3",             # 2021-08-27
        13: "Galaxy_Flip3",             # 2021-08-27
        14: "iPhone13_Mini",            # 2021-09-24
        15: "Galaxy_S22",               # 2022-02-25
        16: "Galaxy_S22_Ultra",         # 2022-02-25
        17: "Galaxy_Quantum3",          # 2022-04-29
        18: "Galaxy_Fold4",             # 2022-08-25
        19: "Galaxy_Flip4"              # 2022-08-25
    }
    return task_mapping.get(task, "Unknown Task")