# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:18:26 2021

@author: johan
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import aicsimageio
import aicspylibczi

def tilify(image, mask, name, outdir, tsize=256):
    
    tiles_x = image.shape[1]//tsize
    tiles_y = image.shape[2]//tsize
    
    for i in tqdm(range(tiles_x), desc=name):
        for j in range(tiles_y):
            img_patch = image[:,
                              i*tsize: i*tsize + tsize,
                              j*tsize: j*tsize + tsize]
            msk_patch = mask[i*tsize: i*tsize + tsize,
                              j*tsize: j*tsize + tsize]
            
            # No tiles with majority (>95%) background
            if np.sum(msk_patch == 1) >0.95*tsize**2:
                continue
            
            # No tiles with weird background zeros
            if np.sum(img_patch.sum(axis=0) == 0) > 100:
                continue
            
            # format string for tile location
            position = '[x={:d} y={:d} w={:d} h={:d}]'.format(i, j, tsize, tsize)
            
            # write tile to file
            tf.imwrite(os.path.join(outdir, name + ' ' + position + '.tif'), img_patch)
            tf.imwrite(os.path.join(outdir, name + ' ' + position + '-labelled.tif'), msk_patch)
    
    
if __name__ == '__main__':
    data_dir = r'E:\Promotion\Projects\2021_Necrotic_Segmentation\src\IF\raw'
    tile_dir = os.path.join(os.path.dirname(data_dir), 'tiles')    
    
    mask_ID = '-labelled'    
    files = os.listdir(data_dir)
    
    df = pd.DataFrame(columns=['Image', 'Mask'])
    df.Mask = [f for f in files if mask_ID in f]
    df.Image = [f.replace(mask_ID, '').replace('.tif', '.czi') for f in df.Mask]
    
    # irerate over samples in dataframe
    for i, sample in df.iterrows():
        czi = aicspylibczi.CziFile(os.path.join(data_dir, sample.Image))
        C0 = czi.read_mosaic(C=0, scale_factor=1)[0][None, :, :]  # read channel and add channel dimension
        C1 = czi.read_mosaic(C=1, scale_factor=1)[0][None, :, :]
        C2 = czi.read_mosaic(C=2, scale_factor=1)[0][None, :, :]
        Image = np.vstack([C0, C1, C2])
        Mask = tf.imread(os.path.join(data_dir, sample.Mask))
        
        tilify(Image, Mask,
               name=sample.Image.split('.')[0],
               outdir=tile_dir)
    
    
    
    
