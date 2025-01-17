import os
import random
import cv2
import pywt
import matplotlib.pyplot as plt
import numpy as np
import shutil
from natsort import natsorted


def perform_wavelet_transform(image_channels):
    coeffs = [pywt.swt2(channel, 'haar', level=1, trim_approx=True, norm=False) for channel in image_channels]

    LL = [coeff[0] for coeff in coeffs]
    LH = [coeff[1][0] for coeff in coeffs]
    HL = [coeff[1][1] for coeff in coeffs]
    HH = [coeff[1][2] for coeff in coeffs]
    
    return LL, LH, HL, HH

def save_subband_images(output_root, prefix, image, LL, LH, HL, HH):
    os.makedirs(output_root, exist_ok=True)
    cv2.imwrite(f"{output_root}/{prefix}_LL.png", image + cv2.merge(LL))
    cv2.imwrite(f"{output_root}/{prefix}_LH.png", image + cv2.merge(LH))
    cv2.imwrite(f"{output_root}/{prefix}_HL.png", image + cv2.merge(HL))
    cv2.imwrite(f"{output_root}/{prefix}_HH.png", image + cv2.merge(HH))

def wavelet_fig(image_root, output_root):
    cover_root = os.path.join(image_root, 'cover', '224')
    stego_root = os.path.join(image_root, 'stego', '224_LSB_0.5')
    
    img_index = random.randint(0, len(os.listdir(cover_root)) - 1)
    cover_path = os.path.join(cover_root, natsorted(os.listdir(cover_root))[img_index])
    stego_path = os.path.join(stego_root, natsorted(os.listdir(stego_root))[img_index])
    basename = os.path.basename(cover_path)
    filename = os.path.splitext(basename)[0]
    shutil.copy(cover_path, output_root + basename)
    
    cover_image = cv2.imread(cover_path)
    stego_image = cv2.imread(stego_path)
    cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB)
    
    # Split the image into R, G, B channels
    cover_channels = cv2.split(cover_image)
    stego_channels = cv2.split(stego_image)

    # Perform 2D wavelet transform on each channel
    cover_LL, cover_LH, cover_HL, cover_HH = perform_wavelet_transform(cover_channels)
    stego_LL, stego_LH, stego_HL, stego_HH = perform_wavelet_transform(stego_channels)
    
    # Save the subband images
    save_subband_images(output_root, 'cover', cover_image, cover_LL, cover_LH, cover_HL, cover_HH)
    save_subband_images(output_root, 'stego', stego_image, stego_LL, stego_LH, stego_HL, stego_HH)

def main():
    image_root = '../../data/2024_NSR_Steganalysis/Galaxy_S20+/'
    output_root = './results/wavelet_vis/'
    wavelet_fig(image_root, output_root)

if __name__ == "__main__":
    main()