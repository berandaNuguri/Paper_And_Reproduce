import os
import torch
import shutil
import numpy as np

def save_checkpoint(state, gpus, is_best,
                    model_path = f'./ckpt/',
                    model_name = f'checkpoint.pth.tar'):
    
    if(len(gpus) > 1):
        old_state_dict = state['state_dict']
        from collections import OrderedDict
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
