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


class InferenceDataset():
    def __init__(self, image_base_dir, filename, augmentation=None,
                 patch_size=512, stride=16, n_classes=3, series=3, target_pixsize = 3.5):
        
        self.image_base_dir = image_base_dir
        self.filename = os.path.join(image_base_dir, filename)
        self.augmentation = augmentation
        self.resolution = 0.4418 * 2**series
        
        self.patch_size = patch_size
        self.stride = stride
        self.inner = patch_size - 2*self.stride
        self.offset = 0
        
        # Read czi image
        self.image = bioformats.load_image(self.filename, c=None, z=0, t=0, series=series)
        self.resample(target_pixsize)
        self.pad()
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))        
        
        self.prediction = np.zeros_like(self.image, dtype='float32')
        
        # assign indeces on 2d grid
        self.index_map = np.arange(0, self.__len__(), 1).reshape(self.shape())
        
    def pad(self):
        """
        Expands image to have a size that's a integer factor of the tile size
        """
        
        self.dimensions = self.image.shape
        shape = np.ceil(np.asarray(self.image.shape)/self.patch_size) * self.patch_size
        shape[np.argmin(shape)] = 3
        
        ds = shape - self.image.shape
        width = self.image.shape[0]
        height = self.image.shape[1]
        
        x = int(ds[0]//2)
        xx = int(ds[0] - x)
        
        y = int(ds[1]//2)
        yy = int(ds[1] - y)
        
        self.image = np.pad(self.image, ((x, xx), (y, yy), (0,0)),
                            mode='constant', constant_values=0)
        self.anchor = [x, y, width, height]
        
        
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
        
            
    def predict(self, model, device='cuda', batch_size=6, n_offsets=3, max_offset=16):
        
        dataloader = DataLoader(self, batch_size=batch_size,
                                shuffle=False, num_workers=4)
        
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.arange(0, max_offset)[::int(max_offset/n_offsets)]
        
        with torch.no_grad():
            
            tk0 = tqdm(offsets, desc=os.path.basename(self.filename), total=len(offsets))
            for offset in tk0:
                
                self.offset = offset

                # iterate over dataloader
                for i, data in enumerate(dataloader):
                    
                    data['image'] = data['image'].to(device).float()
                    out = model(data['image'])
                    out = torch.sigmoid(out).detach().cpu().numpy()
                    
                    tk0.set_postfix(offset=offset, batch=i)
                    
                    # iteratively write results from batches to prediction map
                    for b_idx in range(out.shape[0]):
                        self[i*batch_size + b_idx] = out[b_idx]
                        
        # average predictions and undo padding
        self.prediction /= (n_offsets + 1)
        self.prediction = self.prediction[:,
                                          self.anchor[0]: self.anchor[0] + self.anchor[2],
                                          self.anchor[1]: self.anchor[1] + self.anchor[3]]
        
        # self.postprocess()
    
    def postprocess(self, Class_cutoffs=[0.5, 0.5, 0.4]):
        """
        Create labelmap from probabilities
        """
        for i in range(len(Class_cutoffs)):
            self.prediction[i] = (self.prediction[i] > Class_cutoffs[i]) * i
        self.prediction = np.argmax(self.prediction, axis=0)
        
    
    def export(self, filename):
        """
        Export prediction map to file with deflation compression
        """
        
        tf.imwrite(filename, self.prediction)

root = r'E:\Promotion\Projects\2021_Necrotic_Segmentation'
RAW_DIR = r'E:\Promotion\Projects\2020_Radiomics\Data'
BST_MODEL = root + r'\data\Experiment_20210328_221305_4µm5_ts256_bs20\model\bst_model256_fold4_0.7073.bin'
DEVICE = 'cuda'
PATCH_SIZE = 256
STRIDE = 16
batch_size = 20
Redo = True


if __name__ == '__main__':
    
    javabridge.start_vm(class_path=bioformats.JARS)
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=3, 
        activation=None,
    )
    
    aug_test = A.Compose([
        A.Normalize(),
        PadIfNeeded(min_width=PATCH_SIZE, min_height=PATCH_SIZE)
        # A.ToTensorV2()
    ])
    
    model.load_state_dict(torch.load(BST_MODEL))
    model = model.to(DEVICE)
    
    samples = helper_functions.scan_database(RAW_DIR, img_type='czi')
    
    for i, sample in samples.iterrows():
        outpath = os.path.join(sample.Directory, '1_seg', 'HE_seg_DL.tif')
        
        if os.path.exists(outpath) and not Redo:
            continue
        else:
            try:
                ds = InferenceDataset(RAW_DIR, sample.Image_ID,
                                      patch_size=PATCH_SIZE, stride=STRIDE)
                ds.predict(model, batch_size=batch_size)
                ds.export(outpath)
            except Exception:
                print('Error in {:s}'.format(sample.Image_ID))
                pass

        
    javabridge.kill_vm()

    
