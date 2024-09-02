import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio

from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from skimage.transform import resize


def plot_mat_spectrum_dist(df, data_root, save_path):
    num_channels = 30
    num_rows, num_cols = 5, 6
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
    
    live_spectra = []
    spoof_spectra = []
    
    for _, row in df.iterrows():
        mat_path = row['mat_path']
        label = row['label']
        
        mat_data = loadmat(f'{data_root}/{mat_path}')
        spectrum = mat_data['var']
        
        if label == 0:  # Live
            live_spectra.append(spectrum)
        else:  # Spoof
            spoof_spectra.append(spectrum)

    for channel in range(num_channels):
        row = channel // num_cols
        col = channel % num_cols
        ax = axes[row, col]
        
        live_channel_values = np.concatenate([spectrum[:, :, channel].flatten() for spectrum in live_spectra])
        spoof_channel_values = np.concatenate([spectrum[:, :, channel].flatten() for spectrum in spoof_spectra])
        
        live_hist, bins = np.histogram(live_channel_values, bins=50, density=True)
        spoof_hist, _ = np.histogram(spoof_channel_values, bins=bins, density=True)
        

        live_hist = live_hist / np.sum(live_hist)
        spoof_hist = spoof_hist / np.sum(spoof_hist)
        
        ax.plot(bins[:-1], live_hist, color='green', alpha=0.5, label='Live')
        ax.plot(bins[:-1], spoof_hist, color='red', alpha=0.5, label='Spoof')
        
        ax.set_title(f'Channel {channel+1}')
        ax.set_xlabel('Spectrum Value')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path+'/spectrum_dist.png')
    plt.close()

def plot_png_spectrum_dist(df, data_root, save_path):
    num_channels = 3
    num_rows, num_cols = 3, 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))

    live_spectra = []
    spoof_spectra = []

    for _, row in df.iterrows():
        png_path = row['mat_path'].replace('.mat', '.png')
        label = row['label']

        png_image = Image.open(f'{data_root}/{png_path}')
        png_image = png_image.resize((224, 224))
        png_image = np.array(png_image)

        if label == 0:  # Live
            live_spectra.append(png_image)
        else:  # Spoof
            spoof_spectra.append(png_image)

    channel_names = ['Red', 'Green', 'Blue']
    for channel in range(num_channels):
        ax = axes[channel]

        live_channel_values = np.concatenate([image[:, :, channel].flatten() for image in live_spectra])
        spoof_channel_values = np.concatenate([image[:, :, channel].flatten() for image in spoof_spectra])

        live_hist, bins = np.histogram(live_channel_values, bins=50, density=True)
        spoof_hist, _ = np.histogram(spoof_channel_values, bins=bins, density=True)

        live_hist = live_hist / np.sum(live_hist)
        spoof_hist = spoof_hist / np.sum(spoof_hist)

        ax.plot(bins[:-1], live_hist, color='green', alpha=0.5, label='Live')
        ax.plot(bins[:-1], spoof_hist, color='red', alpha=0.5, label='Spoof')

        ax.set_title(f'{channel_names[channel]} Channel')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path+'/png_spectrum_dist.png')
    plt.close()

def plot_average_images(df, data_root, save_path, target_size=(224, 224)):
    num_channels = 30
    live_avg_image = np.zeros(target_size + (num_channels,))
    spoof_avg_image = np.zeros(target_size + (num_channels,))
    live_count = 0
    spoof_count = 0

    for _, row in df.iterrows():
        mat_path = row['mat_path']
        label = row['label']
        mat_data = loadmat(f'{data_root}/{mat_path}')
        spectrum = mat_data['var']
        spectrum = resize(spectrum, target_size + (num_channels,), mode='reflect', preserve_range=True)
        if label == 0:
            live_avg_image += spectrum
            live_count += 1
        else:
            spoof_avg_image += spectrum
            spoof_count += 1

    live_avg_image /= live_count
    spoof_avg_image /= spoof_count

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(live_avg_image.mean(axis=-1), cmap='gray')  # 여기를 수정했습니다.
    axes[0].set_title('Live Average Image')
    axes[0].axis('off')
    axes[1].imshow(spoof_avg_image.mean(axis=-1), cmap='gray')  # 여기를 수정했습니다.
    axes[1].set_title('Spoof Average Image')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path+'/average_images.png')
    
def channel_wise_avg_imgs(df, data_root, save_path, target_size=(224, 224)):
    num_channels = 30
    live_avg_images = np.zeros(target_size + (num_channels,))
    spoof_avg_images = np.zeros(target_size + (num_channels,))
    live_count = 0
    spoof_count = 0

    for _, row in df.iterrows():
        mat_path = row['mat_path']
        label = row['label']
        mat_data = loadmat(f'{data_root}/{mat_path}')
        spectrum = mat_data['var']
        spectrum = resize(spectrum, target_size + (num_channels,), mode='reflect', preserve_range=True)
        if label == 0:
            live_avg_images += spectrum
            live_count += 1
        else:
            spoof_avg_images += spectrum
            spoof_count += 1

    live_avg_images /= live_count
    spoof_avg_images /= spoof_count

    # Live 이미지 저장
    fig, axes = plt.subplots(5, 6, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(live_avg_images[:, :, i], cmap='gray')
        ax.set_title(f'Live Channel {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_path}/live.png')
    plt.close()

    # Spoof 이미지 저장
    fig, axes = plt.subplots(5, 6, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(spoof_avg_images[:, :, i], cmap='gray')
        ax.set_title(f'Spoof Channel {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_path}/spoof.png')
    plt.close()

def channel_wise_diff(df, data_root, target_size=(224, 224), num_channels=30):
    live_df = df[df['label'] == 0]
    spoof_df = df[df['label'] == 1]
    
    live_avg_images = np.zeros(target_size + (num_channels,))
    spoof_avg_images = np.zeros(target_size + (num_channels,))
    live_count = 0
    spoof_count = 0
    
    for _, row in live_df.iterrows():
        mat_path = row['mat_path']
        mat_data = loadmat(f'{data_root}/{mat_path}')
        spectrum = mat_data['var']
        spectrum = resize(spectrum, target_size + (num_channels,), mode='reflect', preserve_range=True)
        live_avg_images += spectrum
        live_count += 1
        
    for _, row in spoof_df.iterrows():
        mat_path = row['mat_path']
        mat_data = loadmat(f'{data_root}/{mat_path}')
        spectrum = mat_data['var']
        spectrum = resize(spectrum, target_size + (num_channels,), mode='reflect', preserve_range=True)
        spoof_avg_images += spectrum
        spoof_count += 1
        
    live_avg_images /= live_count
    spoof_avg_images /= spoof_count
    
    channel_diffs = np.abs(live_avg_images - spoof_avg_images).mean(axis=(0, 1))
    
    print("Channel-wise Differences:")
    for i in range(num_channels):
        print(f"Channel {i}: {channel_diffs[i]:.4f}")

def plot_spectrum_distributions(img_root, mat_file_list, save_path):
    df = pd.read_csv(mat_file_list, sep=' ', header=None, names=['mat_path', 'label'])

    # Initialize lists to store spectra for all images
    rgb_spectra = {'R': [], 'G': [], 'B': []}
    mat_spectra = [[] for _ in range(30)]  # Assuming 30 channels in MAT files

    # Process PNG images
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PNG images"):
        img_path = os.path.join(img_root, row['mat_path'].replace('.mat', '.png'))
        img = Image.open(img_path)
        img_array = np.array(img)
        for i, channel in enumerate(['R', 'G', 'B']):
            rgb_spectra[channel].extend(img_array[:, :, i].ravel())

    # Process MAT files
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MAT files"):
        mat_path = os.path.join(img_root, row['mat_path'])
        mat_data = loadmat(mat_path)['var']
        for i in range(30):
            mat_spectra[i].extend(mat_data[:, :, i].ravel())

    # Plot comparisons for each RGB channel
    for channel in ['R', 'G', 'B']:
        fig, axs = plt.subplots(5, 6, figsize=(20, 15), sharex=True, sharey=True)
        fig.suptitle(f'Spectrum Distribution Comparison for {channel} Channel', fontsize=16)
        
        # Plot RGB channel distribution
        for i, ax in tqdm(enumerate(axs.flat), total=30):
            if i < 30:
                sns.histplot(rgb_spectra[channel], bins=100, color=channel.lower(), alpha=0.5, ax=ax, kde=True, label=f"{channel} Channel")
                sns.histplot(mat_spectra[i], bins=100, color='skyblue', alpha=0.5, ax=ax, kde=True, label=f"MAT Channel {i+1}")
                ax.legend()
            else:
                ax.axis('off')
            break

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, f"{channel}_channel_spectrum_distribution_comparison.png"))
        plt.close()

def calculate_channel_diff(img_root, mat_file_list, save_path):
    df = pd.read_csv(mat_file_list, sep=' ', header=None, names=['mat_path', 'label'])
    channel_diffs = {'Channel': [], 'R_diff': [], 'G_diff': [], 'B_diff': []}

    # 초기 차이값을 저장할 배열 준비
    diffs = np.zeros((30, 3))  # 30 MAT 채널, RGB 각각에 대한 차이

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
        img_path = os.path.join(img_root, row['mat_path'].replace('.mat', '.png'))
        mat_path = os.path.join(img_root, row['mat_path'])

        # 이미지 로드 및 정규화
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img) / 255.0

        # MAT 파일 로드
        mat_data = loadmat(mat_path)['var']

        # RGB 채널과 MAT 채널의 차이 계산
        for i in range(30):  # MAT 채널 반복
            mat_channel_mean = np.mean(mat_data[:, :, i])
            for j, color in enumerate(['R', 'G', 'B']):  # RGB 채널 반복
                rgb_channel_mean = np.mean(img_array[:, :, j])
                diffs[i, j] += np.abs(mat_channel_mean - rgb_channel_mean)

    # 평균 차이 계산 및 저장
    diffs /= len(df)
    for i in range(30):
        channel_diffs['Channel'].append(f'MAT Channel {i+1}')
        channel_diffs['R_diff'].append(diffs[i, 0])
        channel_diffs['G_diff'].append(diffs[i, 1])
        channel_diffs['B_diff'].append(diffs[i, 2])

    # 데이터 프레임 생성 및 CSV 파일로 저장
    diff_df = pd.DataFrame(channel_diffs)
    diff_df.to_csv(os.path.join(save_path, 'channel_diffs.csv'), index=False)

def find_min_diff_channels_and_values(csv_path):
    # CSV 파일 로드
    df = pd.read_csv(csv_path)

    # R, G, B에 대해 차이값이 가장 작은 3개의 MAT 채널과 그 값을 찾음
    results = {}
    for color in ['R_diff', 'G_diff', 'B_diff']:
        sorted_df = df[['Channel', color]].sort_values(by=color).head(3)
        results[color] = sorted_df
    
    return results

def calc_min_max(file_path, mat_file_list, output_csv_path):
    channel_min_values = np.inf * np.ones((30,))
    channel_max_values = -np.inf * np.ones((30,))

    with open(mat_file_list, 'r') as f:
        lines = f.readlines()

    for line in lines:
        file_name, _ = line.strip().split()
        mat_file_path = os.path.join(file_path, file_name)
        mat_contents = sio.loadmat(mat_file_path)

        data = mat_contents['var']

        for i in range(data.shape[2]):
            channel_data = data[:, i]
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            channel_min_values[i] = min(channel_min_values[i], min_val)
            channel_max_values[i] = max(channel_max_values[i], max_val)

    df = pd.DataFrame({
        'Channel': range(1, 31),
        'Min Value': channel_min_values,
        'Max Value': channel_max_values
    })

    df.to_csv(output_csv_path, index=False)

def visualize_difference(df, img_root, idx, r, g, b, input_size=(224, 224), save_path=None):
    mat_path = df.iloc[idx]['mat_path']
    print(mat_path.split('/')[-1])

    mat_data = loadmat(os.path.join(img_root, mat_path))

    channel1 = mat_data['var'][:, :, r]
    r_chan = resize(channel1, input_size, mode='reflect', preserve_range=True)
    channel2 = mat_data['var'][:, :, g]
    g_chan = resize(channel2, input_size, mode='reflect', preserve_range=True)
    channel3 = mat_data['var'][:, :, b]
    b_chan = resize(channel3, input_size, mode='reflect', preserve_range=True)

    img_mat = np.stack((r_chan, g_chan, b_chan), axis=-1)
    img_mat = (img_mat * 255).astype(np.uint8)

    img_name = os.path.splitext(mat_path.split('/')[-1])[0] + '.png'
    img_png_path = os.path.join(img_root, img_name)
    img_png = Image.open(img_png_path)
    img_png = img_png.resize(input_size)
    img_png = np.array(img_png)

    diff_img = np.abs(img_mat - img_png)
    diff_img = np.mean(diff_img, axis=-1)

    if save_path is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(img_mat)
        axs[0].set_title('MAT Channels Mixed')
        axs[1].imshow(img_png)
        axs[1].set_title('PNG Image')
        im = axs[2].imshow(diff_img, cmap='viridis')
        axs[2].set_title('Difference Image')
        fig.colorbar(im, ax=axs[2])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        print("저장 경로가 지정되지 않았습니다.")

def calculate_channel_correlations(df, data_root, target_size=(224, 224), num_channels=30):
    num_samples = len(df)
    mat_channels = np.zeros((num_samples, num_channels))
    rgb_channels = np.zeros((num_samples, 3))

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mat_path = row['mat_path']
        mat_data = loadmat(f'{data_root}/{mat_path}')
        spectrum = mat_data['var']
        spectrum = resize(spectrum, target_size + (num_channels,), mode='reflect', preserve_range=True)
        mat_channels[idx] = spectrum.mean(axis=(0, 1))

        png_path = os.path.splitext(mat_path)[0] + '.png'
        png_image = Image.open(f'{data_root}/{png_path}')
        png_image = png_image.resize(target_size)
        rgb_image = np.array(png_image)
        rgb_channels[idx] = rgb_image.mean(axis=(0, 1))
        # if idx == 100:
        #     break
    correlations = np.zeros((num_channels, 3))
    for i in range(num_channels):
        for j in range(3):
            correlations[i, j] = np.corrcoef(mat_channels[:, i], rgb_channels[:, j])[0, 1]

    return correlations

def channel_correlations(correlations, save_path):
    df_correlations = pd.DataFrame(correlations, columns=['R', 'G', 'B'])
    df_correlations.index = np.arange(0, len(df_correlations))
    df_correlations.index.name = 'MAT Channel'
    df_correlations.to_csv(save_path+'/channel_correlations.csv')

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(correlations, cmap='coolwarm', aspect='auto')
    # im = ax.imshow(correlations, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['R', 'G', 'B'])
    ax.set_yticks(range(0, len(correlations), 1))
    ax.set_yticklabels(range(0, len(correlations), 1))
    ax.set_xlabel('RGB Channels')
    ax.set_ylabel('MAT Channels')
    ax.set_title('Correlations')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'{save_path}/channel_correlations.png')
    plt.close()

if __name__ == '__main__':
    img_root = './data/phase1/images/'
    mat_file_list = f'./data/phase1/train.txt'
    save_path = './analysis/'
    csv_path = './analysis/'

    df = pd.read_csv(mat_file_list, sep=' ', header=None, names=['mat_path', 'label'])

    idx = random.randint(0, len(df))

    # 채널 별 스펙트럼 분포 시각화
    # plot_spectrum_dist(df, img_root, save_path)
    
    # 채널 별 분포 차이 계산
    # channel_wise_diff(df, img_root)    

    # Live, Spoof의 평균 이미지 시각화
    # plot_average_images(df, img_root, save_path)
    
    # 채널 별 Live, Spoof의 평균 이미지 시각화
    # channel_wise_avg_imgs(df, img_root, save_path)
    
    # PNG 이미지의 채널 별 스펙트럼 분포 시각화
    # plot_png_spectrum_dist(df, img_root, save_path)
    
    # PNG 이미지의 채널 별 분포와 MAT 파일의 채널 별 분포 시각화
    # plot_spectrum_distributions(img_root, mat_file_list, save_path)

    # PNG 이미지의 채널 별 분포와 MAT 파일의 채널 별 분포 차이 분석
    # calculate_channel_diff(img_root, mat_file_list, save_path)
    # min_diff_channels_and_values = find_min_diff_channels_and_values(csv_path)

    # print("R, G, B 채널과 차이값이 가장 작은 3개의 MAT 채널과 그 값:")
    # for color, data in min_diff_channels_and_values.items():
    #     print(f"{color[:-5]} 채널 비교:")
    #     for index, row in data.iterrows():
    #         print(f"  {row['Channel']}: {row[color]:.2f}")
    #     print()

    # MAT 파일의 채널 별 최소값과 최대값 계산
    # calc_min_max(img_root, mat_file_list, csv_path)

    # MAT와 PNG 차이 이미지 시각화
    # visualize_difference(df, img_root, idx, 10, 9, 21, input_size=(112, 112), save_path=save_path)

    channel_correlations(calculate_channel_correlations(df, img_root, target_size=(224, 224), num_channels=30), save_path)