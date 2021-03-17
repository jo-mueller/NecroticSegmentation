# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:30:57 2021

Forward processing of HE image necrosis segmentation

Source: https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb

@author: johan
"""

import cv2
import os
from Utils import helper_functions
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
import torch

import torch.cuda.memory as memory

class InferenceDataset():
    def __init__(self, image_base_dir, filename, augmentation=None,
                 patch_size=512, stride=16, n_classes=3):
        
        self.image_base_dir = image_base_dir
        self.filename = os.path.join(image_base_dir, filename)
        self.augmentation = augmentation
        
        self.image = cv2.imread(self.filename)
        
        
        self.patch_size = patch_size
        self.stride = stride
        self.inner = patch_size - 2*self.stride
        
        self.prediction= np.zeros((self.image.shape[0],
                                   self.image.shape[1],
                                   n_classes), dtype=np.float32)
        
    def shape(self):
        return (np.array(self.image.shape[:2]) - 2*self.stride)//self.inner
        
    def __len__(self):
        return self.shape()[0] * self.shape()[1]
    
    def __getitem__(self, key):
        
        i = key // self.shape()[0]
        j = key % self.shape()[0]
        
        stride = self.stride
        inner = self.inner
       
        image = self.image[i*inner: i*inner + inner + 2*stride,
                           j*inner: j*inner + inner + 2*stride, :]
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image  = sample['image']

        return {'image': image}
    
    def __setitem__(self, key, value):
        
        i = key // self.shape()[0]
        j = key % self.shape()[0]
        
        # crop center of processed tile
        patch = value[self.stride : value.shape[0] - self.stride,
                      self.stride : value.shape[1] - self.stride, :]
        
        stride = self.stride
        inner = self.inner
        self.prediction[stride + i*inner : stride + i*inner + inner,
                        stride + j*inner : stride + j*inner + inner, :] = patch
            
    def predict(self, model, device='cuda', batch_size=6):
        
        dataloader = DataLoader(self, batch_size=batch_size,
                                shuffle=False, num_workers=4)
        tk0 = tqdm(dataloader, total=len(dataloader))
        
        with torch.no_grad():
            # iterate over dataloader
            for i, data in enumerate(tk0):
                data['image'] = data['image'].to(device)
                
                out = model(data['image'])
                out = out.detach().cpu().numpy()
                
                # iteratively write results from batches to prediction map
                for b_idx in range(batch_size):
                    self[i*batch_size + b_idx] = out[b_idx].transpose((1, 2, 0))
                
# def post_process(probability, threshold, min_size):
#     mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
#     num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
#     predictions = np.zeros((512, 512), np.float32)
#     num = 0
#     for c in range(1, num_component):
#         p = (component == c)
#         if p.sum() > min_size:
#             predictions[p] = 1
#             num += 1
#     return predictions, num
    


root = r'E:\Promotion\Projects\2021_Necrotic_Segmentation'
RAW_DIR = root + r'\src\Raw'
BST_MODEL = root + r'\data\Experiment_20210316_195512\model\bst_model512_fold4_0.4859.bin'
DEVICE = 'cuda'
PATCH_SIZE = 512
STRIDE = 64
batch_size = 6


if __name__ == '__main__':
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=3, 
        activation=None,
    )
    
    model.load_state_dict(torch.load(BST_MODEL))
    model = model.to(DEVICE)
    
    # Test transforms
    TEST_TFMS = albu.Compose([albu.Normalize(), ToTensor(), ])
    
    samples = helper_functions.scan_directory2(RAW_DIR)
    
    for i, sample in enumerate(samples.Image_ID):
        test_dataset = InferenceDataset(RAW_DIR, sample, TEST_TFMS,
                                        patch_size=PATCH_SIZE, stride=STRIDE)
        test_dataset.predict(model, batch_size=batch_size)

    
