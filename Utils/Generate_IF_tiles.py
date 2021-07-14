# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:18:26 2021

@author: johan
"""

import os
import cv2
import pandas as pd
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import aicsimageio

# def tilify(sample_dir):
    
   
            
    
    
if __name__ == '__main__':
    data_dir = r'E:\Promotion\Projects\2020_Radiomics\Data'
    n_samples= 20
    
    samples = []
    for root, subdirs, files in os.walk(data_dir):
        if "Histology" in root and "1_seg" in subdirs:
            
            # no censored data
            if "censored" in root:
                continue
            
            samples.append(root)
    
    # draw some random samples from the data pool
    samples = np.asarray(samples)[np.random.randint(0, len(samples), n_samples)]
    
    for sample in samples:
        try:
            img = [x for x in os.listdir(sample) if "IF" in x and (x.endswith('tif') or x.endswith('czi'))][0]
        except:
            continue
        
        if img.endswith('tif'):
            img = tf.imread(os.path.join(sample, img))
        elif img.endswith('czi'):
            AICS = aicsimageio.AICSImage(os.path.join(sample, img))
            
            
        mask = tf.imread(os.path.join(sample, "1_seg", "IF_seg_Simple_Segmentation.tif"))