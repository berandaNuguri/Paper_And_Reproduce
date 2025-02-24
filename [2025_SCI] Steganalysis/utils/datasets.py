import os
import cv2
from torch.utils.data import Dataset

class NistDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path']
        # 0: Cover, 1: Stego
        label = 0 if self.df.iloc[idx]['label'] == 'cover' else 1

        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label

class SFADataset(Dataset):
    def __init__(self, df, root_dir='',transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['image_name']
        algorithm = self.df.iloc[idx]['algorithm']
        embedding_rate = self.df.iloc[idx]['embedding_rate']
        # 0: Cover, 1: Stego
        if algorithm == 'cover':
            image_path = os.path.join(self.root_dir, 'Cover', image_name)
            label = 0       
        else:
            image_path = os.path.join(self.root_dir, 'Stego', f'{algorithm}_{embedding_rate}', image_name)
            label = 1   

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label
