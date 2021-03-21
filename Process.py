# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:30:57 2021

Forward processing of HE image necrosis segmentation

Source: https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb

@author: johan
"""

import javabridge
import bioformats
import os
from Utils import helper_functions
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch


class InferenceDataset():
    def __init__(self, image_base_dir, filename, augmentation=None,
                 patch_size=512, stride=16, n_classes=3, series=3, target_pixsize = 3.5):
        
        self.image_base_dir = image_base_dir
        self.filename = os.path.join(image_base_dir, filename)
        self.augmentation = augmentation
        self.resolution = 0.4418 * 2**series
        
        # Read czi image
        self.image = bioformats.load_image(self.filename, c=None, z=0, t=0, series=series)
        self.resample(target_pixsize)
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))        
        
        self.patch_size = patch_size
        self.stride = stride
        self.inner = patch_size - 2*self.stride
        self.offset = 0
        
        self.prediction = np.zeros_like(self.image, dtype='float32')
        self.prediction[0,:,:] = 1  # background default to 100%
        
        # assign indeces on 2d grid
        self.index_map = np.arange(0, self.__len__(), 1).reshape(self.shape())
        
    def resample(self, resolution):
        """
        Resamples self.image to a given pixelsize <resolution>    
        """
        
        factor = self.resolution/resolution
        outsize = [x*factor if x !=3 else 3 for x in np.array(self.image.shape, dtype=float)]
        self.image = resize(self.image, np.floor(outsize))
        self.resolution /= factor
        
    def shape(self):
        return (np.array(self.image.shape[1:]))//self.inner
        
    def __len__(self):
        return self.shape()[0] * self.shape()[1]
    
    def __getitem__(self, key):
        
        
        ind = np.argwhere(self.index_map == key)[0]
        i, j = ind[0], ind[1]
        
        stride = self.stride
        inner = self.inner
        offset = self.offset
       
        image = self.image[:, 
                           offset + i*inner: offset + i*inner + inner + 2*stride,
                           offset + j*inner: offset + j*inner + inner + 2*stride]
        
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        return {'image': image}
    
    def __setitem__(self, key, value):
        
        ind = np.argwhere(self.index_map == key)[0]
        i, j = ind[0], ind[1]
        
        # crop center of processed tile
        patch = value[:,
                      self.stride : self.patch_size - self.stride,
                      self.stride : self.patch_size - self.stride]
        
        stride = self.stride
        inner = self.inner
        offset = self.offset
        
        self.prediction[:,
                        offset + stride + i*inner : offset + stride + i*inner + inner,
                        offset + stride + j*inner : offset + stride + j*inner + inner] += patch
        
            
    def predict(self, model, device='cuda', batch_size=6, n_offsets=6, max_offset=128):
        
        dataloader = DataLoader(self, batch_size=batch_size,
                                shuffle=False, num_workers=4)
        
        offsets = np.arange(0, max_offset)[::int(max_offset/n_offsets)]
        
        with torch.no_grad():
            
            for offset in offsets:
                
                tk0 = tqdm(dataloader, total=len(dataloader))
                self.offset = offset
            
                # iterate over dataloader
                for i, data in enumerate(tk0):
                    
                    tk0.set_postfix(offset=offset)
                    data['image'] = data['image'].to(device).float()
                    
                    # out = data['image'].cpu().numpy()
                    out = data['image'].cpu().numpy()
                    out = model(data['image'])
                    out = torch.sigmoid(out).detach().cpu().numpy()
                    
                    # iteratively write results from batches to prediction map
                    for b_idx in range(out.shape[0]):
                        self[i*batch_size + b_idx] = out[b_idx]
                    
                self.prediction[0][self.prediction[0]==0] = 1
                        
        self.prediction /= n_offsets

root = r'E:\Promotion\Projects\2021_Necrotic_Segmentation'
RAW_DIR = root + r'\src\Raw'
BST_MODEL = root + r'\data\\Experiment_20210320_150935\model\bst_model512_fold4_0.7847.bin'
DEVICE = 'cuda'
PATCH_SIZE = 512
STRIDE = 32
batch_size = 6


if __name__ == '__main__':
    
    javabridge.start_vm(class_path=bioformats.JARS)
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=3, 
        activation=None,
    )
    
    aug_test = A.Compose([
        A.Normalize()
        # A.ToTensorV2()
    ])
    
    model.load_state_dict(torch.load(BST_MODEL))
    model = model.to(DEVICE)
    
    samples = helper_functions.scan_directory(RAW_DIR, img_type='czi')
    
    for i, sample in enumerate(samples.Image_ID):
        test_dataset = InferenceDataset(RAW_DIR, sample,
                                        patch_size=PATCH_SIZE, stride=STRIDE)
        # blubb = test_dataset[0]
        test_dataset.predict(model, batch_size=batch_size)
        plt.figure()
        plt.imshow(test_dataset.prediction[0,:,:])
        plt.figure()
        plt.imshow(test_dataset.prediction[1,:,:])
        plt.figure()
        plt.imshow(test_dataset.prediction[2,:,:])
        
    javabridge.kill_vm()

    
