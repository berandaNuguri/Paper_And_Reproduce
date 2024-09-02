import cv2
import matplotlib.pyplot as plt
from albumentations import (
    Compose, Blur, RandomBrightness, Sharpen, ColorJitter, Downscale
)


# 이미지 로드
# 2_1_23/LIVE/LEFT_INDEX - 논문에 사용한 이미지
image = cv2.imread('D:/Projects/AIM/Competition/LivDet2023/data/Dermalog/11_0_19/LIVE/LEFT_INDEX (3).jpg')


augmentations = Compose([
    # Blur(p=1.0, blur_limit=(1, 1)),  # Blur 적용
    # RandomBrightness(p=1.0, limit=(-0.7, -0.7)),  # RandomBrightness 적용
    Sharpen(p=1.0, alpha=(0.2, 0.2), lightness=(1.0, 1.0)),  # Sharpen 적용
    # ColorJitter(p=1.0, hue=120)
    # Downscale(p=1.0, scale_min=0.2, scale_max=0.2)
])
augmented1_image = augmentations(image=image)['image']


augmentations = Compose([
    # Blur(p=1.0, blur_limit=(9, 9)),  # Blur 적용
    # RandomBrightness(p=1.0, limit=(0.7, 0.7)),  # RandomBrightness 적용
    Sharpen(p=1.0, alpha=(0.7, 0.7), lightness=(1.0, 1.0)),  # Sharpen 적용
    # ColorJitter(p=1.0, hue=120)
    # Downscale(p=1.0, scale_min=0.7, scale_max=0.7)
])
augmented2_image = augmentations(image=image)['image']


cv2.imwrite('aug_results/original.png', image)
cv2.imwrite('aug_results/sharpen_aug1_image.png', augmented1_image)
cv2.imwrite('aug_results/sharpen_aug2_image.png', augmented2_image)
# 원본 이미지와 증대된 이미지 시각화
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Augmented Image')
# plt.imshow(augmented_image)
# plt.axis('off')

# plt.show()
