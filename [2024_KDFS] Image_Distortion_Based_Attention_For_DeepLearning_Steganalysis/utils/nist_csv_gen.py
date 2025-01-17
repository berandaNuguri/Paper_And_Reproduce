import argparse
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_cover_stego_lists(args):
    df = pd.read_csv(args.csv_path)
    
    image_format = df['image_format'].to_numpy()
    label = df['image_type'].to_numpy()
    algorithm = df['embedding_method'].to_numpy()
    embedding_rate = df['embedding_rate'].to_numpy()
    
    mask = (args.min_embedding_rate <= embedding_rate) & (embedding_rate <= args.max_embedding_rate) & (image_format == args.data_format) \
            & ((label == 'cover') | (algorithm == args.stego_method))
    filtered_df = df[mask]
    
    cover_list = []
    stego_list = []
    
    cover_image_filenames = filtered_df['cover_image_filename'].to_numpy()
    stego_image_filenames = filtered_df['image_filename'].to_numpy()
    filtered_embedding_rates = filtered_df['embedding_rate'].to_numpy()

    print(f'Starting make {args.stego_method} cover and stego list...')
    for cover_image_filename, stego_image_filename, emb_rate in zip(cover_image_filenames, stego_image_filenames, filtered_embedding_rates):
        cover_image_path = os.path.join(args.base_folder, 'cover', cover_image_filename)
        stego_image_path = os.path.join(args.base_folder, 'stego', stego_image_filename)
        
        cover_image = cv2.imread(cover_image_path)
        stego_image = cv2.imread(stego_image_path)
        
        if cover_image is not None and cover_image.shape[2] == 3:
            cover_list.append((cover_image_path, 'cover', 0))
        if stego_image is not None and stego_image.shape[2] == 3:
            stego_list.append((stego_image_path, args.stego_method, emb_rate))

    return cover_list, stego_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='../../data/StegoAppDB_stegos_20210520-082343_stego_directory.csv')
    parser.add_argument('--base_folder', type=str, default='../../data/NIST_StegoAppDB/')
    parser.add_argument('--data_format', type=str, default='JPG')
    parser.add_argument('--min_embedding_rate', type=float, default=0.1)
    parser.add_argument('--max_embedding_rate', type=float, default=0.15)
    parser.add_argument('--stego_method', type=str, default='PixelKnot')
    parser.add_argument('--train_csv', type=str, default='./csv/StegoAppDB/PixelKnot_bpp_0.1_0.15_train.csv')
    parser.add_argument('--valid_csv', type=str, default='./csv/StegoAppDB/PixelKnot_bpp_0.1_0.15_valid.csv')
    args = parser.parse_args()
    
    cover_list, stego_list = load_cover_stego_lists(args)
    df = pd.DataFrame(cover_list + stego_list, columns=['image_path', 'label', 'embedding_rate'])

    train_df, valid_df= train_test_split(df, test_size=0.3, random_state=42)

    os.makedirs('./csv/StegoAppDB/', exist_ok=True)
    train_df.to_csv(args.train_csv, index=False)
    valid_df.to_csv(args.valid_csv, index=False)
    
    print(f'Train Num: {len(train_df)}')
    print(f'Valid Num: {len(valid_df)}')
    
    args.stego_method= 'Passlok'
    args.train_csv = './csv/StegoAppDB/Passlok_bpp_0.1_0.15_train.csv'
    args.valid_csv = './csv/StegoAppDB/Passlok_bpp_0.1_0.15_valid.csv'
    cover_list, stego_list = load_cover_stego_lists(args)
    df = pd.DataFrame(cover_list + stego_list, columns=['image_path', 'label', 'embedding_rate'])

    train_df, valid_df= train_test_split(df, test_size=0.3, random_state=42)

    os.makedirs('./csv/StegoAppDB/', exist_ok=True)
    train_df.to_csv(args.train_csv, index=False)
    valid_df.to_csv(args.valid_csv, index=False)
    
    print(f'Train Num: {len(train_df)}')
    print(f'Valid Num: {len(valid_df)}')