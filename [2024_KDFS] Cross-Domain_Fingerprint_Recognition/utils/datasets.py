import cv2
import random
import numpy as np
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)

class DatasetRetriever(Dataset):
    def __init__(self, image_names1, image_names2, labels1, labels2 , transforms=None):
        super().__init__()
        self.image_names1 = image_names1
        self.image_names2 = image_names2
        self.labels1 = labels1
        self.labels2 = labels2
        self.transforms = transforms

    def __getitem__(self, index: int):
        if 0.5 > random.random():
            index2 = index
        else : 
            index2 = random.randint(0,len(self.image_names2)-1)

        image_name1, image_name2,  label1, label2 = self.image_names1[index], self.image_names2[index2], self.labels1[index], self.labels2[index2]
        image_path1 = image_name1 #f'{TRAIN_DATA_ROOT_PATH}/{kind}/{image_name1}'.replace('\\', '/')
        image_path2 = image_name2  #f'{TRAIN_DATA_ROOT_PATH}/{kind}/{image_name2}'.replace('\\', '/')
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label = int(label1==label2)

        # image1 /= 255.0
        # image2 /= 255.0

        if self.transforms:
            image1 = self.transforms(image=image1)['image']
            image2 = self.transforms(image=image2)['image']
            
        return image1, image2, label

    def __len__(self) -> int:
        return self.image_names1.shape[0]