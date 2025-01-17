import os
import cv2
from torch.utils.data import Dataset

class StegDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path']
        image_path = image_path.replace('../../', 'C:/Personal/Hanbat_Univ/AIMLab/Paper/2024/SCI/')

        # 0: Cover, 1: Stego
        label = 0 if self.df.iloc[idx]['label'] == 'cover' else 1

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label
