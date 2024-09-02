import os
import json
import sys
import glob

def oulu_process(label_save_root, dataset_root):
    label_save_dir = label_save_root + '/oulu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
        
    train_dataset_path = dataset_root + '/oulu/train/'
    test_dataset_path = dataset_root + '/oulu/test/'
    train_label_file = label_save_dir + 'oulu_train.txt'
    test_label_file = label_save_dir + 'oulu_test.txt'

    train_path_list = glob.glob(train_dataset_path + '**/*.png', recursive=True)
    train_path_list.sort()
    test_path_list = glob.glob(test_dataset_path + '**/*.png', recursive=True)
    test_path_list.sort()

    with open(train_label_file, 'w') as f_train, open(test_label_file, 'w') as f_test:
        # 1: print, 2: replay
        for path in train_path_list:
            if 'real' in path:
                label = '0'
                f_train.write(f"{path} {label}\n")
            elif 'fake' in path:
                # Print, Replay
                label = path.split('_')[-3]
                label = '1' if label in ['2', '3'] else '2'
                f_train.write(f"{path} {label}\n")
                
        for path in test_path_list:
            if 'real' in path:
                label = '0'
                f_test.write(f"{path} {label}\n")
            elif 'fake' in path:
                # Print, Replay
                label = path.split('_')[-3]
                label = '1' if label in ['2', '3'] else '2'
                f_test.write(f"{path} {label}\n")


def casia_process(label_save_root, dataset_root):
    label_save_dir = label_save_root + '/casia/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    train_dataset_path = dataset_root + '/casia/train/'
    test_dataset_path = dataset_root + '/casia/test/'
    train_label_file = label_save_dir + 'casia_train.txt'
    test_label_file = label_save_dir + 'casia_test.txt'

    train_path_list = glob.glob(train_dataset_path + '**/*.png', recursive=True)
    train_path_list.sort()
    test_path_list = glob.glob(test_dataset_path + '**/*.png', recursive=True)
    test_path_list.sort()

    with open(train_label_file, 'w') as f_train, open(test_label_file, 'w') as f_test:
        # 1: print, 2: replay
        for path in train_path_list:
            if 'real' in path:
                label = '0'
                f_train.write(f"{path} {label}\n")
            elif 'fake' in path:
                resolution = path.split('_')[-4]
                label = path.split('_')[-3]
                
                if resolution == 'NM':
                    # Print(Cut)
                    if label in ['5','6']:
                        label = '1'
                    # Print(Warped)
                    elif label in ['3', '4']:
                        label = '1'
                    # Replay
                    elif label in ['7', '8', '8(1)']:
                        label = '2'
                else:
                    # Print(Cut)
                    if label in ['3']:
                        label = '1'
                    # Print(Warped)
                    elif label in ['2']:
                        label = '1'
                    # Replay
                    elif label in ['4']:
                        label = '2'
                f_train.write(f"{path} {label}\n")
                
        for path in test_path_list:
            if 'real' in path:
                label = '0'
                f_test.write(f"{path} {label}\n")
            elif 'fake' in path:
                resolution = path.split('_')[-4]
                label = path.split('_')[-3]
                
                if resolution == 'NM':
                    # Print(Cut)
                    if label in ['5','6','5(1)']:
                        label = '1'
                    # Print(Warped)
                    elif label in ['3', '4']:
                        label = '1'
                    # Replay
                    elif label in ['7', '8', '7(1)', '8(1)']:
                        label = '2'
                else:
                    # Print(Cut)
                    if label in ['3']:
                        label = '1'
                    # Print(Warped)
                    elif label in ['2']:
                        label = '1'
                    # Replay
                    elif label in ['4']:
                        label = '2'
                f_test.write(f"{path} {label}\n")

    
def replay_process(label_save_root, dataset_root):
    label_save_dir = label_save_root + '/replay/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    train_dataset_path = dataset_root + '/replay/train/'
    test_dataset_path = dataset_root + '/replay/test/'
    train_label_file = label_save_dir + 'replay_train.txt'
    test_label_file = label_save_dir + 'replay_test.txt'

    train_path_list = glob.glob(train_dataset_path + '**/*.png', recursive=True)
    train_path_list.sort()
    test_path_list = glob.glob(test_dataset_path + '**/*.png', recursive=True)
    test_path_list.sort()


    with open(train_label_file, 'w') as f_train, open(test_label_file, 'w') as f_test:
        # 1: print, 2: cut, 3: warped, 4: replay
        for path in train_path_list:
            if 'real' in path:
                label = '0'
                f_train.write(f"{path} {label}\n")
            elif 'fake' in path:
                label = path.split('_')[-4]
                # Print
                if label in ['photo']:
                    label = '1'
                # Replay
                elif label in ['video']:
                    label = '2'

                f_train.write(f"{path} {label}\n")

        for path in test_path_list:
            if 'real' in path:
                label = '0'
                f_test.write(f"{path} {label}\n")
            elif 'fake' in path:
                label = path.split('_')[-4]
                # Print
                if label in ['photo']:
                    label = '1'
                # Replay
                elif label in ['video']:
                    label = '2'

                f_test.write(f"{path} {label}\n")

def msu_process(label_save_root, dataset_root):
    label_save_dir = label_save_root + '/msu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    train_dataset_path = dataset_root + '/msu/train/'
    test_dataset_path = dataset_root + '/msu/test/'
    train_label_file = label_save_dir + '/msu_train.txt'
    test_label_file = label_save_dir + '/msu_test.txt'

    train_path_list = glob.glob(train_dataset_path + '**/*.png', recursive=True)
    train_path_list.sort()
    test_path_list = glob.glob(test_dataset_path + '**/*.png', recursive=True)
    test_path_list.sort()


    with open(train_label_file, 'w') as f_train, open(test_label_file, 'w') as f_test:
        # 1: print, 2: cut, 3: warped, 4: replay
        for path in train_path_list:
            if 'real' in path:
                label = '0'
                f_train.write(f"{path} {label}\n")
            elif 'fake' in path:
                label = path.split('_')[-4]
                # Print
                if label in ['photo']:
                    label = '1'
                # Replay
                elif label in ['video']:
                    label = '2'

                f_train.write(f"{path} {label}\n")

        for path in test_path_list:
            if 'real' in path:
                label = '0'
                f_test.write(f"{path} {label}\n")
            elif 'fake' in path:
                label = path.split('_')[-4]
                # Print
                if label in ['photo']:
                    label = '1'
                # Replay
                elif label in ['video']:
                    label = '2'

                f_test.write(f"{path} {label}\n")

if __name__ == '__main__':
    # Label consists of two parts
    # 0: Live
    # 1~: Spoof via AttackTypes
    # 1: print, 2: replay
    
    label_save_root = '../Datas/MCIO/labels/all_frame(2_attackType)'
    dataset_root = '../Datas/MCIO/all_frame(MTCNN)/'
    
    oulu_process(label_save_root, dataset_root)
    casia_process(label_save_root, dataset_root)
    replay_process(label_save_root, dataset_root)
    msu_process(label_save_root, dataset_root)
    
    