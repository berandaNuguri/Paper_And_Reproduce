import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms_albu(args):
        train_transform_1 = A.Compose([
                A.Resize(height=args.input_size, width=args.input_size, p=1.0),
                # A.ChannelShuffle(0.5),
                # A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomResizedCrop(height=args.input_size, width=args.input_size, scale=(0.1, 0.7), interpolation=3, p=1.0),
                # A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 2.0), p=0.5),
                # A.ToGray(p=0.5),
                # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05, p=0.5),
                # A.RandomGridShuffle(grid=(3, 3), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
        ])
        train_transform_2 = A.Compose([
                A.Resize(height=args.input_size, width=args.input_size, p=1.0),
                # A.ChannelShuffle(p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomResizedCrop(height=args.input_size, width=args.input_size, scale=(0.1, 0.7), interpolation=3, p=1.0),
                # A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 2.0), p=0.5),
                # A.ToGray(p=0.5),
                # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05, p=0.5),
                # A.RandomGridShuffle(grid=(3, 3), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
        ])

        valid_transform = A.Compose([
                A.Resize(height=args.input_size, width=args.input_size, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
        ])

        return train_transform_1, train_transform_2, valid_transform

def analysis_transforms(args):
        transform = A.Compose([
                A.Resize(args.input_size, args.input_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
        ])

        return transform