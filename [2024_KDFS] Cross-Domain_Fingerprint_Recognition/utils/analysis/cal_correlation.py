import os
import torch
import torch.nn as nn
import timm
import numpy as np
import random
import cv2
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.spatial import distance
from collections import OrderedDict
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.nn.functional import cosine_similarity

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_net():
    net = timm.create_model('efficientnet_b0', num_classes=1, in_chans=1, pretrained=True).to(device)

    return net


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = get_net()

        self.n_features = self.model.classifier.in_features

        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        self.classifier = nn.Sequential(
            nn.Linear(self.n_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'classifier': self.classifier.state_dict(),
        }
        
    def load_state_dict(self, state_dict):
        model_state = {k: v for k, v in state_dict.items() if k.startswith('model.')}
        classifier_state = {k.replace('classifier.', ''): v for k, v in state_dict.items() if k.startswith('classifier.')}

        self.model.load_state_dict(model_state)
        self.classifier.load_state_dict(classifier_state)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def eval(self):
        self.model.eval()

    def forward_once(self, x):
        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        output = torch.cat((output1, output2), 1)
        output = self.classifier(output)
        
        output = self.sigmoid(output)

        cos_sim = cosine_similarity(output1, output2)
        
        return output, cos_sim

def print_score(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    image1 = image1 / 255.0
    image2 = image2 / 255.0
    
    transform = A.Compose([
        A.Resize(500, 500),
        # A.RandomBrightness(limit=0.3, p=0.5),
        # A.RandomContrast(limit=0.3, p=0.5),
        # A.Sharpen(alpha=(0.2, 0.5), lightness=(1.0, 1.0), p=0.5),
        # A.MedianBlur(blur_limit=(3, 7), p=1),
        ToTensorV2(p=1.0)
    ])
    
    # Transform images and add an extra dimension
    image1 = transform(image=image1)['image'].unsqueeze(1).to(device)
    image2 = transform(image=image2)['image'].unsqueeze(1).to(device)
    
    # Forward images through the model
    with torch.no_grad():
        match_score, cos_sim = model(image1, image2)

    # Print Similarity
    # print("IMG 1: ", image1_path.split('/')[-4:])
    # print("IMG 2: ", image2_path.split('/')[-4:])
    print(f"Match Score: {match_score.item():.4f}")
    print(f"Cosine Similarity: {cos_sim.item():.4f}")
    print("")


if __name__ == "__main__":
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork().to(device)

    augmentations = ['Base', 'Sharpen', 'RandomBrightness', 'RandomContrast', 'MedianBlur']

    train_dataset='Dermalog'
    cross_dataset='Dermalog'
    # 인물 A: 11_0_19
    # 인물 B: 9_0_18
    # 인물 C: 25_0_31
    # 인물 D: 17_1_18


    image1_path = f'D:/AIM/Competition/2023/LivDet2023/data/{cross_dataset}/25_0_31/LIVE/RIGHT_RING (3).jpg'

    image2_path_list = [f'D:/AIM/Competition/2023/LivDet2023/data/{cross_dataset}/25_0_31/LIVE/RIGHT_RING.jpg',
                        f'D:/AIM/Competition/2023/LivDet2023/data/{cross_dataset}/17_1_18/LIVE/RIGHT_RING (4).jpg',
                        f'D:/AIM/Competition/2023/LivDet2023/data/{cross_dataset}/17_1_18/LIVE/LEFT_THUMB (3).jpg']
    
    for aug in augmentations:
        print(f'======================{aug}======================')

        loaded_state = torch.load(f'D:/AIM/Competition/2023/LivDet2023/checkpoints/CrossMatch/CrossMatch_{train_dataset}2{cross_dataset}_{aug}_best.pth')

        model_state_dict = {k.replace('model.', ''): v for k, v in loaded_state['model'].items() if k.startswith('model.')}
        classifier_state_dict = {k.replace('classifier.', ''): v for k, v in loaded_state['model'].items() if k.startswith('classifier.')}

        model.model.load_state_dict(model_state_dict)
        model.classifier.load_state_dict(classifier_state_dict)

        model.eval()
        for path in image2_path_list:
            print_score(image1_path, path)
        break