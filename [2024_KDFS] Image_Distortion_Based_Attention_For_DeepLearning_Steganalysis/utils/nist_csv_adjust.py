import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def check_bpp(args, train_df=None, valid_df=None):
    df = pd.read_csv(args.csv_path)
    if train_df is None:
        train_df = pd.read_csv(args.train_csv)
    if valid_df is None:
        valid_df = pd.read_csv(args.valid_csv)
        
    train_df['embedding_rate'] = ''
    valid_df['embedding_rate'] = ''

    stego_dict = df.set_index('image_filename')['embedding_rate'].to_dict()

    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_name = row['image_path'].split('/')[-1]
        label = row['label']

        if label == 'cover':
            train_df.loc[i, 'embedding_rate'] = 0
        elif label == 'stego':
            train_df.loc[i, 'embedding_rate'] = stego_dict.get(image_name, '')
    
    train_df.to_csv(args.train_csv, index=False)

    for i, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        image_name = row['image_path'].split('/')[-1]
        label = row['label']
        
        if label == 'cover':
            valid_df.loc[i, 'embedding_rate'] = 0
        elif label == 'stego':
            valid_df.loc[i, 'embedding_rate'] = stego_dict.get(image_name, '')
    
    valid_df.to_csv(args.valid_csv, index=False)

    return train_df, valid_df

def check_algorithm(args, train_df=None, valid_df=None):
    df = pd.read_csv(args.csv_path)
    if train_df is None:
        train_df = pd.read_csv(args.train_csv)
    if valid_df is None:
        valid_df = pd.read_csv(args.valid_csv)

    stego_dict = df.set_index('image_filename')['embedding_method'].to_dict()

    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_name = row['image_path'].split('/')[-1]
        label = row['label']

        if label == 'stego':
            train_df.loc[i, 'label'] = stego_dict.get(image_name, '')
            
    train_df.to_csv(args.train_csv, index=False)
    
    for i, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        image_name = row['image_path'].split('/')[-1]
        label = row['label']

        if label == 'stego':
            valid_df.loc[i, 'label'] = stego_dict.get(image_name, '')
    
    
    valid_df.to_csv(args.valid_csv, index=False)

    return train_df, valid_df

def make_csv(args, train_df=None, valid_df=None):
    if train_df is None:
        train_df = pd.read_csv(args.train_csv)
    if valid_df is None:
        valid_df = pd.read_csv(args.valid_csv)
    df = pd.concat([train_df, valid_df])

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='../../data/StegoAppDB_stegos_20210520-082343_stego_directory.csv')
    parser.add_argument('--base_folder', type=str, default='../../data/NIST_StegoAppDB/')
    parser.add_argument('--data_format', type=str, default='JPEG')
    parser.add_argument('--train_csv', type=str, default='./csv/StegoAppDB/JPEG_mbpp_0.4_train.csv')
    parser.add_argument('--valid_csv', type=str, default='./csv/StegoAppDB/JPEG_mbpp_0.4_valid.csv')
    args = parser.parse_args()
    
    # train_df, valid_df = check_bpp(args)
    # train_df, valid_df = check_algorithm(args, train_df, valid_df)
    make_csv(args)