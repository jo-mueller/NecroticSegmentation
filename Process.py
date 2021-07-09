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
    def __init__(self, filename,
                 patch_size=512, stride=16, n_classes=3, series=3, target_pixsize = 2.0,
                 max_offset=128, n_offsets=3, **kwargs):
        
        aug_forw = A.Compose([
            PadIfNeeded(min_width=IMG_SIZE, min_height=IMG_SIZE,
                        border_mode=cv2.BORDER_REFLECT)
        ])
        
        self.filename = filename
        self.augmentation = kwargs.get('augmentation', aug_forw)  # padding as default augmentation
        self.resolution = 0.4418
        
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.inner = int(patch_size - 2*stride)
        self.offset = 0
        self.max_offset = max_offset
        self.n_offsets = n_offsets
        
        # Read czi image
        self.czi = aicspylibczi.CziFile(self.filename)
        self.image = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/target_pixsize)[0]
        self.image = self.image[:, :, ::-1]
        self.resolution = target_pixsize
        # self.resample(target_pixsize)
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))
        
        self.prediction = np.zeros_like(self.image, dtype='float32')
        
        # assign indeces on 2d grid and remove black tiles from grid
        self.create_coords()
        self.prune_coords()

    def create_coords(self, stepsize=16):
        
        image = self.image
        self.N_map = np.zeros_like(image)
        
        x = np.arange(self.patch_size//2, self.image.shape[1] - self.patch_size//2, stepsize)
        y = np.arange(self.patch_size//2, self.image.shape[2] - self.patch_size//2, stepsize)
        
        # check if forbidden pixels are in the vicinity of these pixels
        self.coords = []
        for _x in x:
            for _y in y:
                self.coords.append([_x, _y])
                
        self.coords = np.asarray(self.coords)
                
    def prune_coords(self):
        """
        Removes the coordinates from the list where the image is weird black/white
        """
        white_list = []
        black_list = []
        for i in tqdm(range(self.__len__()), desc='Testing tiles...'):
            patch = self.__getitem__(i)['image'].numpy()
            if np.sum(patch == 0) > 100 or np.sum(patch.sum(axis=0) >= 3*254) > 100:
                black_list.append(i)
            else:
                white_list.append(i)
        
        print('Pruned {:d} indeces from calculation locations.'.format(self.__len__() - len(white_list) ))
        self.coords = self.coords[white_list]


    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, key):
        
        coord = self.coords[key]
        
        ps = self.patch_size
        stride = self.stride
        image = self.image[:,
                           coord[0] - ps//2 + stride: coord[0] + ps//2 - stride,
                           coord[1] - ps//2 + stride: coord[1] + ps//2 - stride]
        
        # apply augmentations, transpose axii foor this
        if self.augmentation:
            sample = self.augmentation(image=image.transpose((1,2,0)))
            image = sample['image'].transpose((2,0,1))

        return {'image': torch.from_numpy(image.copy())}
    
    def __setitem__(self, key, value):
        
        stride = self.stride
        coord = self.coords[key]
        ps = self.patch_size
        
        patch = value[:,
                      stride : - stride,
                      stride : - stride]
        
        self.prediction[:,
                        coord[0] - ps//2 + stride : coord[0] + ps//2 - stride,
                        coord[1] - ps//2 + stride : coord[1] + ps//2 - stride] +=patch
        
        self.N_map[:,
                   coord[0] - ps//2 + stride : coord[0] + ps//2 - stride,
                   coord[1] - ps//2 + stride : coord[1] + ps//2 - stride] += 1
        
            
    def predict(self, model, device='cuda', batch_size=6):
        
        with torch.no_grad():            
                
            dataloader = DataLoader(self, batch_size=batch_size,
                    shuffle=False, num_workers=0)

            # iterate over dataloader
            tk0 = tqdm(dataloader)
            
            for i, data in enumerate(tk0):
                
                data['image'] = data['image'].to(device).float()
                data['prediction'] = model(data['image'])
                out = torch.sigmoid(data['prediction']).detach().cpu().numpy()
                
                # iteratively write results from batches to prediction map
                for b_idx in range(out.shape[0]):
                    self[i*batch_size + b_idx] = out[b_idx]
                        
        # average predictions
        # self.prediction = np.divide(self.prediction, self.N_map)     
        
        
    def postprocess(self, **kwargs):
        """
        Export prediction map to file with deflation compression
        """
        
        project = kwargs.get('project', True)
        filename = kwargs.get('filename', None)
        rescale = kwargs.get('rescale', True)
        
        if project:
            self.prediction = np.argmax(self.prediction, axis=0)
            
        if rescale:
            w = self.czi.get_mosaic_bounding_box().w
            h = self.czi.get_mosaic_bounding_box().h
            self.prediction = cv2.resize(self.prediction,
                                         (w, h),
                                         interpolation=cv2.INTER_NEAREST)
        if filename is not None:
            tf.imwrite(filename, self.prediction)
        
        return self.prediction
    

# =============================================================================
# CONFIG
# =============================================================================
root = r'E:\Promotion\Projects\2021_Necrotic_Segmentation'
RAW_DIR = r'E:\Promotion\Projects\2021_MicroQuant\ImgData'
EXP = root + r'\data\Experiment_20210426_110720'
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
        classes=3, 
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
    
    # iterate over raw images
    model.eval()
    for i, sample in samples.iterrows():
        outpath = os.path.join(sample.Directory, '1_seg', 'HE_seg_DL.tif')
        
        if os.path.exists(outpath) and not Redo:
            continue
        else:
            try:
                ds = InferenceDataset(os.path.join(sample.Directory, sample.Image_ID),
                                      series=SERIES,
                                      patch_size=IMG_SIZE,
                                      stride=STRIDE,
                                      augmentation=aug_forw,
                                      target_pixsize=PIX_SIZE,
                                      max_offset=MAX_OFFSET,
                                      n_offsets=N_OFFSETS)
                ds.predict(model, batch_size=batch_size)
                ds.postprocess()
                # ds.export()
            except Exception:
                print('Error in {:s}'.format(sample.Image_ID))
                pass

    
