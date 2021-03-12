# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:03:06 2021

@author: johan
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import tifffile as tf
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def createExp_dir(root):
    
    time_string = '{:d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(
                    datetime.now().year,
                    datetime.now().month,
                    datetime.now().day,
                    datetime.now().hour,
                    datetime.now().minute,
                    datetime.now().second)
    
    base = os.path.join(root, 'Experiment_' + time_string)
    dir_test = os.path.join(base, 'Test')
    dir_train = os.path.join(base, 'Train')
    os.mkdir(base)
    
    os.mkdir(dir_train)
    os.mkdir(dir_test)
    
    dirs = {'EXP_DIR': base,
            'TRAIN_IMG_DIR': dir_train,
            'TEST_IMG_DIR': dir_test,
            'KFOLD': os.path.join(base, 'RLE_kfold.csv'),
            'RLE_DATA': os.path.join(base, 'RLE_tiles.csv')}
    
    return dirs


def scan_directory(directory, img_type='labels', outname=None):
    
    images = os.listdir(directory)
    images = [x for x in images if img_type in x and x.endswith('tif')]
    
    df = pd.DataFrame(columns=['Image_ID'])
    df['Image_ID'] = images
    
    return df
    

def rle_directory(directory, img_type='labels', outname=None, get_rle=False):
    """
    Convert tif images in directory to run-length encoded representation
    """

    images = os.listdir(directory)
    images = [x for x in images if img_type in x and x.endswith('tif')]
    
    df = pd.DataFrame(columns=('Image_ID', 'RLE'))
    df['Image_ID'] = images

    for fimage in tqdm(images):
        
        img = tf.imread("/".join([directory, fimage]))
        
        # correct 8-bit fuck up in ppatchify.ijm
        img[img == 128] = 1
        img[img == 255] = 2
        
        rle = mask2rle(img)
        df.loc[df['Image_ID'] == fimage, 'RLE'] = rle
        df.loc[df['Image_ID'] == fimage, 'Image_ID'] = fimage.replace('labels', 'image')
    
    if outname is None:
        outname = os.path.join(directory, 'RLE_labels.csv')

    df.to_csv(outname)
    return df

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


def rle2mask(string, width, height):
    
    array = np.zeros((height, width)).flatten()
    rle = [int(x) for x in string.split(' ')]
    
    vals = rle[::2]
    runLengths = rle[1::2]
    
    index = 0
    for i in range(len(vals)):
        array[index:index + runLengths[i]] = vals[i]
        index += runLengths[i]
    
    return array.reshape((-1, width))
    

    
def mask2rle(array):
    
    array = array.flatten()    
    lastValue = -1
    runLength = 0
    rle = []

    for x in array:
        currentValue = x
        
        if currentValue != lastValue:
            rle.append(lastValue)
            rle.append(runLength)
            runLength = 1
            lastValue = currentValue
        else:
            runLength +=1
            
    return ' '.join([str(x) for x in rle[2:]])

def matplotlib_imshow(img, one_channel=False):
    fig,ax = plt.subplots(figsize=(10,6))
    ax.imshow(img.permute(1,2,0).numpy())
    
def visualize(**images):
    """PLot images in one row."""
    images = {k:v.numpy() for k,v in images.items() if isinstance(v, torch.Tensor)} #convert tensor to numpy 
    n = len(images)
    plt.figure(figsize=(16, 8))
    image, mask = images['image'], images['mask']
    
    image = (image - image.min())/(image.max() - image.min())
    
    
    plt.imshow(image.transpose(1,2,0), vmin=0, vmax=1)
    if mask.max()>0:
        plt.imshow(mask.squeeze(0), alpha=0.25)
    plt.show()
    
if __name__ == '__main__':
    root = r'E:\Promotion\Projects\2021_Necrotic_Segmentation\src\Tiles'
    # df = rle_directory(root)