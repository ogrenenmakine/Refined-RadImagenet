import os
import pandas as pd
import cv2
import tarfile
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# untar file
def untar_file(tar_path, extract_path):
    print(f"Extracting {tar_path} to {extract_path}")
    print("Extracting more then 1M png files... This may take a while... ")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

# read csv file
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return list(df['filename'])

# get label
def get_label(file_list):
    label_mapping = {}
    for file in file_list:
        label = file.split('/')[1]
        if label == 'CT' or label == 'MR':
            file_ctg = '__'.join(file.split('/')[1:4])
        elif label == 'US':
            file_ctg = '__'.join(file.split('/')[1:3])
        label_mapping[file] = file_ctg + '/' + file.split('/')[-1]
    return label_mapping

# get target
def get_target(file_list, label_mapping, set_split):
    mapping = []
    for file in file_list:
        target_name = 'rRadImagenet1L' + '/' + set_split + '/' + label_mapping[file]
        mapping.append([file, target_name])
    return mapping

# process image
def process_image(source_path, target_path):
    if 'US' in source_path:
        # create mask_path
        mask_path = os.path.join('correction_masks', os.path.basename(source_path).replace('.png', '.npy'))
        mask = np.load(mask_path)
        
        # read image from source_path
        source_image = cv2.imread('./data/' + source_path)
        source_image = cv2.resize(source_image, (224, 224))

        # calculate mask_image
        target_image = source_image + mask

    else:
        # read image from source_path
        source_image = cv2.imread('./data/' + source_path)

        # calculate target_image
        target_image = source_image

    # save target_image to target_path
    target_path = './output/' + target_path
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    cv2.imwrite(target_path, target_image)

# process images in parallel
def process_images_in_parallel(mapping):
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda x: process_image(x[0], x[1]), mapping), total=len(mapping), desc='Processing'))
    
if __name__ == "__main__":

    # create directories
    os.makedirs('./output', exist_ok=True)
    os.makedirs('./correction_masks', exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./source', exist_ok=True)

    # check if the file exists
    if not os.path.exists(extract_to):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"File {zip_path} not found")
        else:
            with tarfile.open('source/correction_masks.tar.gz', 'r:gz') as tar:
                tar.extractall(path='correction_masks/')    

    # check if the file exists
    if not os.path.exists('./data'):
        if not os.path.exists('./source/radimagenet.tar.gz'):
            raise FileNotFoundError(f"File source/radimagenet.tar.gz not found")
        else:
            untar_file('./source/radimagenet.tar.gz', './data')

    # read csv files
    file_list = read_csv_file('./source/RadiologyAI_test.csv')
    label_mapping = get_label(file_list)
    test_mapping = get_target(file_list, label_mapping, 'test')

    file_list = read_csv_file('./source/RadiologyAI_val.csv')
    label_mapping = get_label(file_list)
    val_mapping = get_target(file_list, label_mapping, 'val')

    file_list = read_csv_file('./source/RadiologyAI_train.csv')
    label_mapping = get_label(file_list)
    train_mapping = get_target(file_list, label_mapping, 'train')

    # merge dictionaries test_mapping, val_mapping, train_mapping
    mapping = test_mapping + val_mapping + train_mapping
    
    # process images in parallel
    process_images_in_parallel(mapping)

