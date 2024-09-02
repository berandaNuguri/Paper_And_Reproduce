import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_transforms(aug):
    if aug == 'Base':
        train_transform = A.Compose([
            A.Resize(500, 500),
            A.Normalize(mean=[0.449], std=[0.226]),
            ToTensorV2(p=1.0),
        ])
    elif aug == 'RandBrightness':
        train_transform = A.Compose([
            A.Resize(500, 500),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, brightness_by_max=True, p=0.5),
            A.Normalize(mean=[0.449], std=[0.226]),
            ToTensorV2(p=1.0),
        ])
    elif aug == 'RandContrast':
        train_transform = A.Compose([
            A.Resize(500, 500),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, brightness_by_max=True, p=0.5),
            A.Normalize(mean=[0.449], std=[0.226]),
            ToTensorV2(p=1.0),
        ])
    elif aug == 'Sharpen':
        train_transform = A.Compose([
            A.Resize(500, 500),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(1.0, 1.0), p=0.5),
            A.Normalize(mean=[0.449], std=[0.226]),
            ToTensorV2(p=1.0),
        ])
    elif aug == 'MedianBlur':
        train_transform = A.Compose([
            A.Resize(500, 500),
            A.MedianBlur(blur_limit=(3, 5), p=0.5),
            A.Normalize(mean=[0.449], std=[0.226]),
            ToTensorV2(p=1.0),
        ])


    valid_transform =  A.Compose([
           A.Resize(500,500),
           A.Normalize(mean=[0.449], std=[0.226]),
           ToTensorV2(p=1.0),
        ], p=1.0)
    
    return train_transform, valid_transform
