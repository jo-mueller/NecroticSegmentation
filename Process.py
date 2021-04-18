# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:30:57 2021

Forward processing of HE image necrosis segmentation

Source: https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb

@author: johan
"""

import javabridge
import bioformats
import cv2
import tifffile as tf
import os
import yaml
from Utils import helper_functions
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
from skimage.transform import resize

import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch
from torch import nn


class InferenceDataset():
    def __init__(self, image_base_dir, filename, augmentation=None,
                 patch_size=512, stride=16, n_classes=3, series=3, target_pixsize = 2.0,
                 max_offset=128, n_offsets=3):
        
        self.image_base_dir = image_base_dir
        self.filename = os.path.join(image_base_dir, filename)
        self.augmentation = augmentation
        self.resolution = 0.4418 * 2**series
        
        self.patch_size = patch_size
        self.stride = stride
        self.inner = patch_size - 2*stride
        self.offset = 0
        self.max_offset = max_offset
        self.n_offsets = n_offsets
        
        # Read czi image
        self.image = bioformats.load_image(self.filename, c=None, z=0, t=0, series=series).astype(np.float32)
        # self.image = np.rot90(self.image)
        self.resample(target_pixsize)
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))
        
        self.prediction = np.zeros_like(self.image, dtype='float32')
        
        # assign indeces on 2d grid and pad to prevent index errors
        self.create_index_map()
        self.pad()
        
    def pad(self):
        """
        Expands image to have a size that's a integer factor of the tile size
        """
        
        # see how many tiles fit in the raw image and round up
        self.dimensions = self.image.shape
        shape = np.ceil(np.asarray(self.image.shape)/self.inner) * self.inner
        shape[np.argmin(shape)] = 3  # force channel dimension to 3
        
        #  difference in shape
        ds = shape - self.image.shape
        
        # preserve old shape for later cropping
        width = self.image.shape[1]
        height = self.image.shape[2]
        self.anchor = [0, 0, width, height]
        
        # calculate left/right/top/down padding margins
        # Add offset to prevent overshoot at edges
        x = 0
        xx = int(ds[1] - x) + self.max_offset
        
        y = 0
        yy = int(ds[2] - y) + self.max_offset
        
        # Execute padding
        self.image = np.pad(self.image, ((0,0), (x, xx), (y, yy)),
                            mode='constant', constant_values=0)
        self.prediction = np.pad(self.prediction, ((0,0), (x, xx), (y, yy)),
                            mode='constant', constant_values=0)
        
        
        
    def resample(self, resolution):
        """
        Resamples self.image to a given pixelsize <resolution>    
        """
        
        factor = self.resolution/resolution
        outsize = [x*factor if x !=3 else 3 for x in np.array(self.image.shape, dtype=float)]
        self.image = resize(self.image, np.floor(outsize))
        self.resolution /= factor
        
    def create_index_map(self):
        
        shape = (np.array(self.image.shape[1:]))//self.inner
        
        self.index_map = np.arange(0, shape[0] * shape[1], 1).astype(np.float32)
        np.random.shuffle(self.index_map)
        self.index_map = self.index_map.reshape(shape)
        
        # Check which of the locations refer to dumb background areas
        for i in range(self.__len__()):
            patch = np.sum(self[i]['image'], axis=0)
            
            if np.sum(patch == 0) > patch.size//2 or np.sum(patch == 3*255) > patch.size//2:
                self.index_map[self.index_map == i] = np.nan
        
        # Now, fill the deleted indices so that a continuous range of indices is provided
        idx = 0
        while True:
            if idx == np.sum(~np.isnan(self.index_map)):
                break
            if (self.index_map == idx).any():
                idx += 1
                continue
            else:
                self.index_map[self.index_map >= idx] -= 1
                idx = 0
        
        return 1
        
    def __len__(self):
        return int(np.nanmax(self.index_map) + 1)
    
    def __getitem__(self, key):
        
        try:
            ind = np.argwhere(self.index_map == key)[0]
        except Exception:
            pass
        i, j = ind[0], ind[1]
        
        inner = self.inner
        offset = self.offset
       
        image = self.image[:, 
                           offset + i*inner: offset + i*inner + inner,
                           offset + j*inner: offset + j*inner + inner]
        
        if self.augmentation:
            sample = self.augmentation(image=image.transpose((1,2,0)))
            image = sample['image'].transpose((2,0,1))

        return {'image': image}
    
    def __setitem__(self, key, value):
        try:
            ind = np.argwhere(self.index_map == key)[0]
        except Exception:
            pass
        i, j = ind[0], ind[1]
        
        # crop center of processed tile
        patch = value[:,
                      self.stride : self.patch_size - self.stride,
                      self.stride : self.patch_size - self.stride]
        
        inner = self.inner
        offset = self.offset
        
        self.prediction[:,
                        offset + i*inner : offset + i*inner + inner,
                        offset + j*inner : offset + j*inner + inner] += patch
        
            
    def predict(self, model, device='cuda', batch_size=6):
                
        if self.n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.arange(0, self.max_offset)[::int(self.max_offset/self.n_offsets)]
        
        with torch.no_grad():
            
            tk0 = tqdm(offsets, desc=os.path.basename(self.filename), total=len(offsets))
            for offset in tk0:
                
                self.offset = offset
                dataloader = DataLoader(self, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    
                # iterate over dataloader
                for i, data in enumerate(dataloader):
                    
                    data['image'] = data['image'].to(device).float()
                    data['prediction'] = model(data['image'])
                    out = torch.sigmoid(data['prediction']).detach().cpu().numpy()
                    # helper_functions.visualize_batch(data, 0, 0, 0)
                    
                    tk0.set_postfix(offset=offset, batch=i)
                    
                    # iteratively write results from batches to prediction map
                    for b_idx in range(out.shape[0]):
                        self[i*batch_size + b_idx] = out[b_idx]
                        
        # average predictions and undo padding
        self.prediction /= (self.n_offsets + 1)
        self.prediction = self.prediction[:,
                                          self.anchor[0]: self.anchor[0] + self.anchor[2],
                                          self.anchor[1]: self.anchor[1] + self.anchor[3]]
        # self.postprocess()
    # 
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
EXP = root + r'\data\Experiment_20210414_204625'
DEVICE = 'cuda'
STRIDE = 32
Redo = True
MAX_OFFSET = 128
N_OFFSETS = 20
SERIES = 2

if __name__ == '__main__':
    
    javabridge.start_vm(class_path=bioformats.JARS)
    
    # read config
    with open(os.path.join(EXP, "params.yaml"), "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    IMG_SIZE = data['Input']['IMG_SIZE']
    batch_size = 32
    PIX_SIZE = data['Input']['PIX_SIZE']
    BST_MODEL = data['Output']['Best_model']
    N_CLASSES = data['Hyperparameters']['N_CLASSES']
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=3, 
        activation=None,
    )
    
    model.load_state_dict(torch.load(BST_MODEL))
    model = model.to(DEVICE)
    model.eval()
    for module in model.modules():
        for child in module.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False

    
    aug_forw = A.Compose([
        # A.Normalize(),
        PadIfNeeded(min_width=IMG_SIZE, min_height=IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT)
    ])
    
    # Scan database for raw images
    samples = helper_functions.scan_database(RAW_DIR, img_type='czi')
    
    # iterate over raw images
    for i, sample in samples.iterrows():
        outpath = os.path.join(sample.Directory, '1_seg', 'HE_seg_DL.tif')
        
        if os.path.exists(outpath) and not Redo:
            continue
        else:
            try:
                ds = InferenceDataset(RAW_DIR, sample.Image_ID,
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

        
    # javabridge.kill_vm()

    
