import torch
import torchvision
import torch.nn.functional as F
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import torch, gc
import segmentation_models_pytorch as smp
import shutil

from glob import glob
from tqdm import trange, tqdm
from torch import nn, optim
from torchvision import transforms, datasets
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = {
    'root_path': f'../data/Train_LivDet2023',
    'lr': 1e-2,  # 학습률
    'weight_decay': 1e-4,  # 가중치 감쇠
    'drop_rate': 0,  # 학습 시 dropout 비율
    'image_size': 500,  # 이미지 크기
    'num_epochs': 10,  # 학습 반복수
    'batch_size': 16,  # 미니배치 크기
    'num_classes': 1,  # 판별할 클래스 개수
    'num_folds': 5,  # 데이터셋 분할 fold 개수
    'val_fold': 1,  # 검증용 fold 선택
    'seed': 10,  # 랜덤 seed 설정
    'ratio': [0.25, 0.25, 0.5]
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f'Seed fixed to {seed}...')

seed_everything(args['seed'])

class DatasetRetriever(Dataset):
    def __init__(self, kinds, encoder_names, id, image_names, transforms=None):
        super().__init__()
        self.root_path = args['root_path']
        self.kinds = kinds
        self.encoder_names = encoder_names
        self.id = id
        self.image_names = image_names
        # self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        path, kind, encoder_name, id, image_name, = self.root_path, self.kinds[index], self.encoder_names[index], self.id[index], self.image_names[index]
        image_path = f'{self.root_path}/{encoder_name}/{id}/{kind}/{image_name}'.replace('\\', '/')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image

    def __len__(self) -> int:
        return self.image_names.shape[0]

"""
First DataLoader for train autoencoder make content images.
"""
root_path = args['root_path']
content_name = 'Unknown1'

contentset = []
for kind in (['LIVE']):
    for path in glob(f'{root_path}/{content_name}/**/{kind}/*.jpg', recursive=True):
        path = path.replace('\\', '/')
        name = path.split('/')
        identity = name[len(name) - 3]
        contentset.append({
            'content_name': content_name,
            'id': identity,
            'kind': kind,
            'image_name': path.replace('\\', '/').split('/')[-1]
        })

contentset = pd.DataFrame(contentset)

kf = KFold(n_splits=args['num_folds'])

contentset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(kf.split(X=contentset.index, groups=contentset['image_name'])):
    contentset.loc[contentset.iloc[val_index].index, 'fold'] = fold_number

transform = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(p=1.0)
], p=1.0)

content_train_dataset = DatasetRetriever(
    kinds=contentset[contentset['fold'] != args['val_fold']].kind.values,
    encoder_names=contentset[contentset['fold'] != args['val_fold']].content_name.values,
    id=contentset[contentset['fold'] != args['val_fold']].id.values,
    image_names=contentset[contentset['fold'] != args['val_fold']].image_name.values,
    # labels=contentset[contentset['fold'] != args['val_fold']].label.values,
    transforms=transform,
)

content_validation_dataset = DatasetRetriever(
    kinds=contentset[contentset['fold'] == args['val_fold']].kind.values,
    encoder_names=contentset[contentset['fold'] == args['val_fold']].content_name.values,
    id=contentset[contentset['fold'] == args['val_fold']].id.values,
    image_names=contentset[contentset['fold'] == args['val_fold']].image_name.values,
    # labels=contentset[contentset['fold'] == args['val_fold']].label.values,
    transforms=transform,
)

content_train_loader = torch.utils.data.DataLoader(
        content_train_dataset,
        batch_size=args['batch_size'],
        pin_memory=False,
        drop_last=False,
    )

content_val_loader = torch.utils.data.DataLoader(
        content_validation_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

"""
Second DataLoader for train autoencoder make Dm,Gb Fake Images.
"""
root_path = args['root_path']
texture_name = 'Dermalog'

textureset = []
for kind in (['FAKE']):
    for path in glob(f'{root_path}/{texture_name}/**/{kind}/*.jpg', recursive=True):
        path = path.replace('\\', '/')
        name = path.split('/')
        identity = name[len(name) - 3]
        textureset.append({
            'texture_name': texture_name,
            'id': identity,
            'kind': kind,
            'image_name': path.replace('\\', '/').split('/')[-1]
        })

textureset = pd.DataFrame(textureset)

kf = KFold(n_splits=args['num_folds'])

textureset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(kf.split(X=textureset.index, groups=textureset['image_name'])):
    textureset.loc[textureset.iloc[val_index].index, 'fold'] = fold_number

transform = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(p=1.0)
], p=1.0)

texture_train_dataset = DatasetRetriever(
    kinds=textureset[textureset['fold'] != args['val_fold']].kind.values,
    encoder_names=textureset[textureset['fold'] != args['val_fold']].texture_name.values,
    id=textureset[textureset['fold'] != args['val_fold']].id.values,
    image_names=textureset[textureset['fold'] != args['val_fold']].image_name.values,
    # labels=contentset[contentset['fold'] != args['val_fold']].label.values,
    transforms=transform,
)

texture_validation_dataset = DatasetRetriever(
    kinds=textureset[textureset['fold'] == args['val_fold']].kind.values,
    encoder_names=textureset[textureset['fold'] == args['val_fold']].texture_name.values,
    id=textureset[textureset['fold'] == args['val_fold']].id.values,
    image_names=textureset[textureset['fold'] == args['val_fold']].image_name.values,
    # labels=contentset[contentset['fold'] == args['val_fold']].label.values,
    transforms=transform,
)

texture_train_loader = torch.utils.data.DataLoader(
        texture_train_dataset,
        batch_size=args['batch_size'],
        pin_memory=False,
        drop_last=False,
    )

texture_val_loader = torch.utils.data.DataLoader(
        texture_validation_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

"""
Third Process for train autoencoder create unknown fake images
"""
root_path = args['root_path']
live_name = 'Unknown1'

model = smp.Unet(
    encoder_name="timm-efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="noisy-student",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

EPOCHS = args['num_epochs']
LR = args['lr']

gc.collect()
torch.cuda.empty_cache()

# PerceptualLoss For Train Content, Texture Features
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval()) # Content
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval()) # Texture
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def save_checkpoint(model, optimizer, best_loss, checkpoint_path, model_name, is_best, epoch):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss,
        'epoch': epoch
    }

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # filename = os.path.join(checkpoint_path, f'{model_name}_model_epoch_{state["epoch"]}.pt')
    # torch.save(state['model'], filename)
    if is_best:
        torch.save(state['model'], os.path.join(checkpoint_path, f'{model_name}_model_best.pt'))

class Trainer():
    def __init__(self, model, optimizer, content_train_loader, content_val_loader, texture_train_loader, texture_val_loader, scheduler, device):
        self.content_model = model
        self.texture_model = model
        self.create_fake_model = model
        self.optimizer = optimizer
        self.content_train_loader = content_train_loader
        self.content_val_loader = content_val_loader
        self.texture_train_loader = texture_train_loader
        self.texture_val_loader = texture_val_loader
        self.scheduler = scheduler
        self.device = device
        # Loss Function
        self.criterion = VGGPerceptualLoss().to(self.device)

    def fit(self):
        self.content_model.to(self.device)
        self.texture_model.to(self.device)
        self.create_fake_model.to(self.device)
        best_loss = 1000000000
        for epoch in range(EPOCHS):
            self.content_model.train()
            self.texture_model.train()
            self.create_fake_model.train()
            content_train_loss = []
            texture_train_loss = []
            create_fake_train_loss = []
            mixed_train_loss = []
            for content_input, texture_input in zip(self.content_train_loader, self.texture_train_loader):
                content_images = content_input
                texture_images = texture_input
                live_images = content_input
                content_input = content_images.to(self.device)
                texture_input = texture_images.to(self.device)
                live_input = live_images.to(self.device)
                
                content_output = self.content_model(content_input)
                content_loss = self.criterion.forward(content_input, content_output, feature_layers=[0])
                content_train_loss.append(content_loss)

                texture_output = self.texture_model(texture_input)
                texture_loss = self.criterion.forward(texture_input, texture_output, feature_layers=[3])
                texture_train_loss.append(texture_loss)

                fake_output = self.create_fake_model(live_input)
                create_fake_loss = torch.nn.functional.l1_loss(live_input, fake_output)
                create_fake_train_loss.append(create_fake_loss)

                # ratio[0] = 0.25, ratio[1] = 0.25, ratio[2] = 0.5 
                self.optimizer.zero_grad()
                mixed_loss = content_loss * args['ratio'][0] + texture_loss * args['ratio'][1] + create_fake_loss * args['ratio'][2]
                mixed_train_loss.append(mixed_loss)
                mixed_loss.backward()
                self.optimizer.step()


            content_valid_loss = self.validation(self.content_model, self.content_val_loader, 'Content')
            texture_valid_loss = self.validation(self.texture_model, self.texture_val_loader, 'Texture')
            create_fake_valid_loss = self.validation(self.create_fake_model, self.content_val_loader, 'Fake')
            mixed_valid_loss = content_loss * args['ratio'][0] + texture_loss * args['ratio'][1] + create_fake_loss * args['ratio'][2]

            mean_content_train_loss = torch.mean(torch.stack(content_train_loss))
            mean_content_valid_loss = torch.mean(torch.stack(content_valid_loss))
            mean_texture_train_loss = torch.mean(torch.stack(texture_train_loss))
            mean_texture_valid_loss = torch.mean(torch.stack(texture_valid_loss))
            mean_create_fake_train_loss = torch.mean(torch.stack(create_fake_train_loss))
            mean_create_fake_valid_loss = torch.mean(torch.stack(create_fake_valid_loss))
            
            mean_mixed_train_loss = torch.mean(torch.stack(mixed_train_loss))
            #mean_mixed_valid_loss = mean_content_valid_loss * args['ratio'][0] + mean_texture_valid_loss * args['ratio'][1] + mean_create_fake_valid_loss * args['ratio'][2]
            mean_mixed_valid_loss = torch.mean(mixed_valid_loss)
            
            
            is_best = mean_mixed_valid_loss < best_loss
            best_loss = min(mean_mixed_valid_loss, best_loss)

            if is_best:
                save_checkpoint(model, self.optimizer, best_loss, './checkpoints/', 'AutoEncoder', is_best, epoch)
            

            print(f'Epoch[{epoch+1}/{EPOCHS}] Content Train Loss: {mean_content_train_loss:.6} Content Valid Loss: {mean_content_valid_loss:.6}')
            print(f'Epoch[{epoch+1}/{EPOCHS}] Texture Train Loss: {mean_texture_train_loss:.6} Texture Valid Loss: {mean_texture_valid_loss:.6}')
            print(f'Epoch[{epoch+1}/{EPOCHS}] Create Train Loss: {mean_create_fake_train_loss:.6} Create Valid Loss: {mean_create_fake_valid_loss:.6}')
            print(f'Epoch[{epoch+1}/{EPOCHS}] Mixed Train Loss: {mean_mixed_train_loss:.6} Mixed Valid Loss: {mean_mixed_valid_loss:.6}')

            # if self.scheduler is not None:
            #     self.scheduler.step(valid_loss)

            #if epoch % 10 == 0:
            #    torch.save(model.state_dict(), './model_' + str(epoch) + '.pth')

    def validation(self, eval_model, val_loader, type):
        eval_model.eval()

        valid_loss=[]
        with torch.no_grad():
            for input in val_loader:
                # x = input, _x = output
                images = input
                input = images.to(self.device)

                if type == 'Content':
                    output = self.content_model(input)
                    loss = self.criterion.forward(input, output, feature_layers=[0])
                elif type == 'Texture':
                    output = self.texture_model(input)
                    loss = self.criterion.forward(input, output, feature_layers=[3])
                elif type == 'Fake':
                    output = self.create_fake_model(input)
                    loss = torch.nn.functional.l1_loss(input, output)
                valid_loss.append(loss)

        return valid_loss


optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
print('=======Start Training=======')
trainer = Trainer(model, optimizer, content_train_loader, content_val_loader, texture_train_loader, texture_val_loader, None, device)
trainer.fit()



