# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:30:57 2021

Forward processing of HE image necrosis segmentation

Source: https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb

@author: johan
"""

import aicspylibczi
import cv2
import PIL
import tifffile as tf
import os

import yaml
from Utils import helper_functions
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch
from torch import nn

class InferenceDataset():
    def __init__(self, filename, n_classes, **kwargs):
        
        self.resolution = kwargs.get('resolution', 0.4418)
        self.target_pixsize = kwargs.get('target_pixsize', 2.5)
        self.patch_size = kwargs.get('patch_size', 128)
        self.stride = kwargs.get('stride', 16)
        self.density = kwargs.get('density', 16)
        self.augmentation = kwargs.get('augmentation', None)
        self.batch_size = kwargs.get('batch_size', 20)
        self.device = kwargs.get('device', 'cuda')
        
        self.filename = filename
        self.resolution = 0.4418
        self.n_classes = n_classes
        
        
        # Read czi image
        self.czi = aicspylibczi.CziFile(self.filename)
        self.image = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/self.target_pixsize)[0]
        self.image = self.image[:, :, ::-1]
        self.resolution = self.target_pixsize
        # self.resample(target_pixsize)
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))
        
        self.prediction = np.zeros((self.n_classes, self.image.shape[1], self.image.shape[2]),
                                   dtype='float32')
        
        # assign indeces on 2d grid and pad to prevent index errors
        self.create_samplelocations()
        # self.create_index_map()
        # self.pad()
        
    def create_samplelocations(self):
        """
        Prepares a list of locations where the image will be fed forward
        through the network
        """
        
        # create sampling locations, omit half-tile sized margin at image edge
        X = np.arange(self.patch_size//2, self.image.shape[1] - self.patch_size//2, self.density)
        Y = np.arange(self.patch_size//2, self.image.shape[2] - self.patch_size//2, self.density)
        
        self.locations = []
        for x in tqdm(X, desc='Browsing image...'):
            for y in Y:
                patch = self.image[:
                                   x - self.patch_size//2 : x + self.patch_size//2,
                                   y - self.patch_size//2 : y + self.patch_size//2]
                if np.sum(patch.sum(axis=0) == 0) > 50 or np.sum(patch.sum(axis=0) >= 3*354):
                    continue
                self.locations.append([x, y])
        
        
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, key):
        
        loc = self.locations[key]
        x, y = loc[0], loc[1]
        
        patch = self.image[:
                           x - self.patch_size//2 : x + self.patch_size//2,
                           y - self.patch_size//2 : y + self.patch_size//2]
            
        return {'image': patch}
    
    def __setitem__(self, key, value):
        
        # crop center of processed tile
        patch = value[:,
                      self.stride : - self.stride,
                      self.stride : - self.stride]
        
        size = patch.shape[1]
        x, y = self.locations[key]
        
        self.prediction[:,
                        x - size//2 : x + size//2,
                        y - size//2 : y + size//2] += patch
            
    def predict(self, model):

        with torch.no_grad():
            dataloader = DataLoader(self, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)        
            tk0 = tqdm(dataloader, total=self.__len__(), desc='Tilewise forward segmentation...')
            for b_idx, data in enumerate(tk0):
                    
                data['image'] = data['image'].to(self.device).float()
                data['prediction'] = model(data['image'])
                out = torch.sigmoid(data['prediction']).detach().cpu().numpy()
                
                # iteratively write results from batches to prediction map
                for b_idx in range(out.shape[0]):
                    self[i*batch_size + b_idx] = out[b_idx]
                        
        # average predictions and undo padding
        self.prediction /= (self.n_offsets + 1)
        self.prediction = self.prediction[:,
                                          self.anchor[0]: self.anchor[0] + self.anchor[2],
                                          self.anchor[1]: self.anchor[1] + self.anchor[3]]
        
    def postprocess(self, Class_cutoffs=[0.5, 0.5, 0.4]):
        """
        Create labelmap from probabilities
        """
        # for i in range(len(Class_cutoffs)):
        #     self.prediction[i] = (self.prediction[i] > Class_cutoffs[i]) * i
        self.prediction = np.argmax(self.prediction, axis=0)
        
    
    def export(self, filename):
        """
        Export prediction map to file with deflation compression
        """
        
        tf.imwrite(filename, self.prediction)


    

# =============================================================================
# CONFIG
# =============================================================================
root = r'E:\Promotion\Projects\2021_Necrotic_Segmentation'
RAW_DIR = r'E:\Promotion\Projects\2020_Radiomics\Data'
EXP = root + r'\data\Experiment_20210713_001952'
DEVICE = 'cuda'
STRIDE = 16
Redo = True
MAX_OFFSET = 64
N_OFFSETS = 10
SERIES = 2

if __name__ == '__main__':
        
    # read config
    with open(os.path.join(EXP, "params.yaml"), "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    IMG_SIZE = int(data['Input']['IMG_SIZE']/2)
    batch_size = int(data['Hyperparameters']['BATCH_SIZE'] * 4)
    PIX_SIZE = data['Input']['PIX_SIZE']
    BST_MODEL = data['Output']['Best_model']
    N_CLASSES = data['Hyperparameters']['N_CLASSES']
    LEARNING_RATE = data['Hyperparameters']['LEARNING_RATE']
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=N_CLASSES, 
        activation=None,
    )
    
    model.load_state_dict(torch.load(BST_MODEL)['model_state_dict'])
    
    model = model.to(DEVICE)
    model.eval()
    
    # for child in model.modules():
    #     for cchild in child.modules():
    #         if type(cchild)==nn.BatchNorm2d:
    #             cchild.track_running_stats = False
    
    aug_forw = A.Compose([
        PadIfNeeded(min_width=IMG_SIZE, min_height=IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT)
    ])
    
    # Scan database for raw images
    samples = helper_functions.scan_database(RAW_DIR, img_type='czi')
    samples = samples[samples.Image_ID == r'E:\Promotion\Projects\2020_Radiomics\Data\N182a_SAS_0_0\HE_E16b_0032.czi']
    
    # iterate over raw images
    model.eval()
    for i, sample in samples.iterrows():
        outpath = os.path.join(sample.Directory, '1_seg', 'HE_seg_DL.tif')
        
        if os.path.exists(outpath) and not Redo:
            continue
        else:
            try:
                ds = InferenceDataset(RAW_DIR, sample.Image_ID,
                                      n_classes=N_CLASSES,
                                      series=SERIES,
                                      patch_size=IMG_SIZE,
                                      stride=STRIDE,
                                      augmentation=aug_forw,
                                      target_pixsize=PIX_SIZE,
                                      max_offset=MAX_OFFSET,
                                      n_offsets=N_OFFSETS)
                ds.predict(model, batch_size=batch_size)
                ds.export(outpath)
            except Exception:
                print('Error in {:s}'.format(sample.Image_ID))
                pass

    
