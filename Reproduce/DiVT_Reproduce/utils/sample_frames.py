import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

"""
torch Weighted Sampler 사용 시 데이터 샘플링 함수(구 버전)
"""
# def sample_frames(bench, flag, num_frames):
#     if flag == 'train':
#         txt_path = '../Datas/MCIO/labels/all_frame(2_attackType)/' + f'{bench}/{bench}_train.txt'
#     else:
#         txt_path = '../Datas/MCIO/labels/all_frame(2_attackType)/' + f'{bench}/{bench}_test.txt'
    
#     df = pd.read_csv(txt_path, delimiter=' ', header=None, names=['photo_path', 'photo_label'])

#     df['video_id'] = df['photo_path'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-2]))

#     sampled_df = pd.DataFrame()
#     real_samples = 0
#     fake_samples = 0
#     for video_id, group in df.groupby('video_id'):
#         if len(group) >= num_frames:
#             sampled_frames = group.sample(n=num_frames, random_state=42)
#         else:
#             sampled_frames = group
#         sampled_df = pd.concat([sampled_df, sampled_frames])
        
#         real_samples += (sampled_frames['photo_label'] == 0).sum()
#         fake_samples += (sampled_frames['photo_label'] != 0).sum()
        
#     sampled_df = sampled_df.drop('video_id', axis=1).reset_index(drop=True)

#     class_num = sampled_df['photo_label'].nunique()
#     unique_classes = sampled_df['photo_label'].unique()
    
#     print(f'{bench.upper()} Domain')
#     print(f'Class: {class_num}')
#     return sampled_df

def sample_frames(bench, mode, flag, num_frames):
    txt_path = f'../Datas/MCIO/labels/all_frame(2_attackType)/{bench}/{bench}_{mode}.txt'
    df = pd.read_csv(txt_path, delimiter=' ', header=None, names=['photo_path', 'photo_label'])
    df['video_id'] = df['photo_path'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-2]))
    sampled_df = pd.DataFrame()
    
    for video_id, group in df.groupby('video_id'):
        sampled_frames = group if len(group) < num_frames else group.sample(n=num_frames, random_state=42)
        if flag == 0:
            sampled_df = pd.concat([sampled_df, sampled_frames[sampled_frames['photo_label'] == 0]])
        elif flag == 1:
            sampled_df = pd.concat([sampled_df, sampled_frames[sampled_frames['photo_label'] != 0]])
        else:
            sampled_df = pd.concat([sampled_df, sampled_frames])
    
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df