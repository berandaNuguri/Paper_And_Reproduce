import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.io import loadmat
from skimage.transform import resize
from torchvision.transforms import ToTensor

class SSIDataset(Dataset):
    def __init__(self, df, root_path,  mix_chans, is_train=True, transform_1=None, transform_2=None):
        self.df = df
        self.root_path = root_path
        self.is_train = is_train
        self.mix_chans = mix_chans
        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.channel_min_max = pd.read_csv('./data/all/train_valid_mat_min_max.csv', index_col="Channel")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name, label = self.df.iloc[idx]
        file_path, file_ext = os.path.splitext(file_name)
        mat_path = os.path.join(self.root_path, file_name.replace(file_ext, '.mat'))
        png_path = os.path.join(self.root_path, file_name.replace(file_ext, '.png'))

        view1 = self.process_mat_file(mat_path)
        view2 = self.process_png_file(png_path)

        
        view1 = self.transform_1(image=np.array(view1, dtype=np.float32))['image']
        view2 = self.transform_2(image=np.array(view2, dtype=np.float32))['image']

        return view1, view2, label, file_name

    def process_mat_file(self, file_path):
        mat_data = loadmat(file_path)['var']
        
        img = []
        if self.mix_chans is None:
            channels_to_process = range(30)
        else:
            channels_to_process = [int(chan) for chan in self.mix_chans]

        for chan_index in channels_to_process:
            channel_data = mat_data[:, :, chan_index]
            min_val = self.channel_min_max.loc[chan_index+1, "Min Value"]
            max_val = self.channel_min_max.loc[chan_index+1, "Max Value"]

            scaled_data = (channel_data - min_val) / (max_val - min_val)
            img.append(scaled_data)

        img = np.stack(img, axis=-1)

        return img
    def process_png_file(self, file_path):
        img = Image.open(file_path).convert('RGB')

        return img
