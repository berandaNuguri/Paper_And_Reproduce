import pandas as pd
import os
import random

from pprint import pprint
from tqdm import tqdm
from glob import glob

seed = 100
random.seed(seed)

root_path = f'D:/AIM/Competition/2023/LivDet2023/utils/data/text_file'
dataset_name = 'Dermalog'

probe_list = []
for kind in (['FAKE', 'LIVE']):
    for path in glob(f'{root_path}/{dataset_name}/**/{kind}/*.jpg'):
        path = path.replace('\\', '/')
        name = path.split('/')
        identity = name[len(name)-3]
        finger = name[len(name)-1].split('.')[0].split(' ')[0]

        if kind == 'FAKE':
            probe_list.append({
                'path': path,
                'id': identity,
                'label': 0,
                'finger': finger
            })

        elif kind == 'LIVE':
            probe_list.append({
                'path': path,
                'id': identity,
                'label': 1,
                'finger': finger
            })

random.shuffle(probe_list)
probeset = pd.DataFrame(probe_list)

template_list = []
for kind in (['LIVE']):
    for path in glob(f'{root_path}/{dataset_name}/**/{kind}/*.jpg'):
        path = path.replace('\\', '/')
        name = path.split('/')
        identity = name[len(name)-3]
        finger = name[len(name)-1].split('.')[0].split(' ')[0]

        if kind == 'FAKE':
            template_list.append({
                'path': path,
                'id': identity,
                'label': 0,
                'finger': finger
            })
        if kind == 'LIVE':
            template_list.append({
                'path': path,
                'id': identity,
                'label': 1,
                'finger': finger
            })


templateset = pd.DataFrame(template_list)

probe_images = []
template_images = []

for i in range(len(probeset)):
    probe_images.append({
        'path': probeset['path'][i]
    })

    same_id = templateset.loc[templateset['id'] == probeset['id'][i]]
    same_finger = pd.DataFrame(same_id[same_id['finger'] == probeset['finger'][i]])
    path = pd.DataFrame(same_finger['path'].sample(n=1, random_state=seed))
    
    template_images.append({
        'path': path['path'].iloc[0]
    })


probe = pd.DataFrame(probe_images)
template = pd.DataFrame(template_images)

with open(f'{root_path}/{dataset_name}/{dataset_name}_probe_original.txt', 'w', encoding='UTF-8') as probe_text:
    for probe_path in probe['path']:
        probe_text.write(probe_path+'\n')

with open(f'{root_path}/{dataset_name}/{dataset_name}_template_original.txt', 'w', encoding='UTF-8') as template_text:
    for template_path in template['path']:
        template_text.write(template_path + '\n')
