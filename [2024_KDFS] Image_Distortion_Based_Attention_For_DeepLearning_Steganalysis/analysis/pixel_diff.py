import os
import cv2
import numpy as np

def main(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    diff = img1 - img2
    
    # diff[diff != 0] = 255
    cv2.imwrite('./results/LL_diff.png', diff)

if __name__ == "__main__":
    img1 = './results/wavelet_vis/cover_LL.png'
    img2 = './results/wavelet_vis/stego_LL.png'
    # img1 = '../../data/2024_NSR_Steganalysis/Galaxy_S20+/cover/224/Galaxy_S20+_39644_back_wide_pro.png'
    # img2 = '../../data/2024_NSR_Steganalysis/Galaxy_S20+/stego/224_LSB_0.5/Galaxy_S20+_39644_back_wide_pro_LSB_0.5.png'
    
    main(img1, img2)