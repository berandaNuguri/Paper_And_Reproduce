import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arch.SIA.SIA import StegoImpactAttention

def normalize_channel(channel):
    channel_min = channel.min()
    channel_max = channel.max()
    norm = (channel - channel_min) / (channel_max - channel_min + 1e-8) * 255
    return np.uint8(norm)

if __name__ == "__main__":
    data_root = '../../../Data/'
    dataset = 'ALASKA2'
    algorithm = 'UERD'
    payload = 0.4

    sia_projector = StegoImpactAttention()
    
    data_list = os.listdir(os.path.join(data_root, dataset, 'Cover'))
    random.shuffle(data_list)
    
    for data_name in data_list:
        cover = cv2.imread(os.path.join(data_root, dataset, 'Cover', data_name))
        stego = cv2.imread(os.path.join(data_root, dataset, algorithm, data_name))
        cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
        stego = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)

        cover_orig = cover.copy()
        stego_orig = stego.copy()
        
        cover_tensor = torch.tensor(cover, dtype=torch.float32).permute(2, 0, 1)
        stego_tensor = torch.tensor(stego, dtype=torch.float32).permute(2, 0, 1)

        rho_cover = sia_projector.get_rho(cover_tensor).detach().cpu().numpy()
        rho_stego = sia_projector.get_rho(stego_tensor).detach().cpu().numpy()

        cover_ch0 = normalize_channel(cover_orig[:, :, 0])
        cover_ch1 = normalize_channel(cover_orig[:, :, 1])
        cover_ch2 = normalize_channel(cover_orig[:, :, 2])

        stego_ch0 = normalize_channel(stego_orig[:, :, 0])
        stego_ch1 = normalize_channel(stego_orig[:, :, 1])
        stego_ch2 = normalize_channel(stego_orig[:, :, 2])

        rho_cover_ch0 = normalize_channel(rho_cover[0])
        rho_cover_ch1 = normalize_channel(rho_cover[1])
        rho_cover_ch2 = normalize_channel(rho_cover[2])

        rho_stego_ch0 = normalize_channel(rho_stego[0])
        rho_stego_ch1 = normalize_channel(rho_stego[1])
        rho_stego_ch2 = normalize_channel(rho_stego[2])

        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        # ----------------------
        # Row 0: Cover`
        axes[0, 0].imshow(cover_ch0, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title(f'Cover R Channel')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(rho_cover_ch0, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title(f'Cover Rho R Channel')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(cover_ch1, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title(f'Cover G Channel')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(rho_cover_ch1, cmap='gray', vmin=0, vmax=255)
        axes[0, 3].set_title(f'Cover Rho G Channel')
        axes[0, 3].axis('off')

        axes[0, 4].imshow(cover_ch2, cmap='gray', vmin=0, vmax=255)
        axes[0, 4].set_title(f'Cover B Channel')
        axes[0, 4].axis('off')

        axes[0, 5].imshow(rho_cover_ch2, cmap='gray', vmin=0, vmax=255)
        axes[0, 5].set_title(f'Cover Rho B Channel')
        axes[0, 5].axis('off')

        # ----------------------
        # Row 1: Stego
        axes[1, 0].imshow(stego_ch0, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f'Stego R Channel')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(rho_stego_ch0, cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title(f'Stego Rho R Channel')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(stego_ch1, cmap='gray', vmin=0, vmax=255)
        axes[1, 2].set_title(f'Stego G Channel')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(rho_stego_ch1, cmap='gray', vmin=0, vmax=255)
        axes[1, 3].set_title(f'Stego Rho G Channel')
        axes[1, 3].axis('off')

        axes[1, 4].imshow(stego_ch2, cmap='gray', vmin=0, vmax=255)
        axes[1, 4].set_title(f'Stego B Channel')
        axes[1, 4].axis('off')

        axes[1, 5].imshow(rho_stego_ch2, cmap='gray', vmin=0, vmax=255)
        axes[1, 5].set_title(f'Stego Rho B Channel')
        axes[1, 5].axis('off')

        plt.tight_layout()
        plt.show()
