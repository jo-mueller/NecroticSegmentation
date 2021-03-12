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

import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
import torch

import torch.cuda.memory as memory

class TestDataset():
    def __init__(self, sample_sub, image_base_dir, augmentation=None):
        self.image_base_dir = image_base_dir
        self.image_ids      = sample_sub.Image_ID.values
        self.augmentation   = augmentation
    
    def __getitem__(self, i):
        image_id  = self.image_ids[i]
        img_path  = os.path.join(self.image_base_dir, image_id) 
        image     = cv2.imread(img_path)
        image     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image  = sample['image']

        return {
            'image': image, 
        }
        
    def __len__(self):
        return len(self.image_ids)
    
def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((512, 512), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num
    
def predict(test_dataloader, model, device='cuda'):
    
    encoded_pixels = []
    losses = helper_functions.AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(test_dataloader, total=len(test_dataloader))
    
    # iterate over dataloader
    for b_idx, data in enumerate(tk0):
        data['image'] = data['image'].to(device)
        out   = model(data['image'])
        out   = out.detach().cpu().numpy()[:, 0, :, :]
        
        # get single images from batch
        for out_ in out:
            import pdb;pdb.set_trace()
            if out_.shape != (512, 512):
                out_ = cv2.resize(out_, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(out_, 0.5, 3500)
            if num_predict == 0:
                encoded_pixels.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixels.append(r)
    return encoded_pixels

root = r'E:\Promotion\Projects\2021_Necrotic_Segmentation'
RAW_DIR = root + r'\src\Raw'
BST_MODEL = root + r'\data\Experiment_20210312_014430\model\bst_model512_fold4_0.7933.bin'
DEVICE = 'cuda'


if __name__ == '__main__':
    
    # Train transforms
    TFMS = albu.Compose([albu.HorizontalFlip(),
                         albu.Rotate(10),
                         albu.Normalize(),
                         ToTensor()])
    
    # Test transforms
    TEST_TFMS = albu.Compose([albu.Normalize(), ToTensor(), ])
    
    samples = helper_functions.scan_directory2(RAW_DIR)
    test_dataset    = TestDataset(samples, RAW_DIR, TEST_TFMS)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4)
    
    model = smp.Unet(
        encoder_name='se_resnext50_32x4d', 
        encoder_weights='imagenet', 
        classes=3, 
        activation=None,
    )
    
    model.load_state_dict(torch.load(BST_MODEL))
    model = model.to('cuda')
    
    predict(test_dataloader, model, DEVICE)