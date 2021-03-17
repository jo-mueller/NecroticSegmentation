# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:48:39 2021

@author: johan
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile as tf
import cv2
import tifffile as tf
import matplotlib.pyplot as plt

import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

from sklearn.model_selection import KFold

import segmentation_models_pytorch as smp
from Utils import helper_functions

from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

class Dataset():
    def __init__(self, rle_df, image_base_dir, augmentation=None):
        self.df             = rle_df
        self.image_base_dir = image_base_dir
        self.image_ids      = rle_df.Image_ID.values
        self.augmentation   = augmentation
    
    def __getitem__(self, i):
        image_id  = self.image_ids[i]
        
        mask_path = os.path.join(self.image_base_dir, image_id)
        img_path = mask_path.replace('-labelled', '')
        image = cv2.imread(img_path, 1).transpose((2,0,1)).astype(float)
        image[:,(np.sum(image, axis=0) == 0)] = 255
        
        mask = np.argmax(tf.imread(mask_path), axis=0)
        mask = mask[None, :, :]
        
        
        return {'image': image, 'mask' : mask}
        
    def __len__(self):
        return len(self.image_ids)


root = os.getcwd()
src = root + r'\src\QuPath_Tiling\tiles'

# Config
FOLD_ID = 4
BATCH_SIZE = 6
USE_SAMPLER = False
SAMPLER  = None
num_workers = 0
LEARNING_RATE = 2e-5
criterion = CrossEntropyLoss()
TRAIN_MODEL = True
EPOCHS = 300
IMG_SIZE = 512
EVALUATE = True
N_CLASSES = 3
PATIENCE = 30
device= 'cuda'

styles = ['--', '-', ':']


if __name__ == '__main__':
    
    # Create paths for train/test image data
    DIRS = helper_functions.createExp_dir(root + '/data')  
    
    # Read labels to dataframe with RLE encoding
    df = helper_functions.scan_directory(src, img_type='-labelled')
    # df = helper_functions.rle_directory(src, outname=DIRS['RLE_DATA'])
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=N_CLASSES, 
        activation=None,
    )
    
    model = model.to(device)
    model.train()        
    
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3,5,6,7,8,9,10,11,13,15], gamma=0.75)

    es = helper_functions.EarlyStopping(patience=PATIENCE, mode='max')
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax1 = ax.twinx()
    plt.ion()
    
    train_score = []
    test_score = []
    h_train, h_legend = None, None
    h_test  = [None for x in range(N_CLASSES)]
    flag = True
    
    # create 5 folds train/test groups
    kf = KFold(n_splits=5, shuffle=True)
    df['kfold']=-1
    for fold, (train_index, test_index) in enumerate(kf.split(df.Image_ID)):
            df.loc[test_index, 'kfold'] = fold
    
    # single fold training for now, rerun notebook to train for multi-fold
    TRAIN_DF = df.query(f'kfold!={FOLD_ID}').reset_index(drop=True)
    VAL_DF   = df.query(f'kfold=={FOLD_ID}').reset_index(drop=True)
    
    # train dataset
    train_dataset = Dataset(TRAIN_DF, src)
    val_dataset   = Dataset(VAL_DF, src)
    
    # dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    
    if TRAIN_MODEL:
        for epoch in range(EPOCHS):
            
            # train
            optimizer.zero_grad()
            tk0 = tqdm(train_dataloader, total=len(train_dataloader))
            
            for b_idx, data in enumerate(tk0):
                
                # move images on GPU
                for key, value in data.items():
                    data[key] = value.to(device).float()
                    
                # train
                # helper_functions.visualize_batch(data)
                out  = model(data['image'])
                loss = criterion(out, data['mask'][:, 0].long())
                loss.backward()
                optimizer.step()
                tk0.set_postfix(loss=loss.cpu().detach().numpy())

            # evaluate
            with torch.no_grad():
                tk1 = tqdm(val_dataloader, total=len(val_dataloader))
                dice = np.zeros(N_CLASSES)
                for b_idx, data in enumerate(tk1):
                    
                    # Eval
                    out = model(data['image'].to(device).float())                 
                    out = torch.argmax(out, dim=1).view(-1)
                    mask = data['mask'].view(-1)
                    
                    dice += jaccard_score(mask.cpu(), out.cpu(), average=None)
                
                dice /= b_idx
                tk1.set_postfix(score=np.mean(dice))
                    
            train_score.append(loss)
            test_score.append(dice)
                
            h_train = ax.plot(train_score, color='orange', label='Training loss')
            
            try:
                [ax1.lines.pop(i) for i in range(N_CLASSES)]
            except Exception:
                pass
                
            for i in range(len(dice)):
                score = np.concatenate( test_score, axis=0 ).reshape(-1, 3)
                h_test[i] = ax1.plot(score[:,i], color='blue',
                                     label='Dice score label {:d}'.format(i),
                                     linestyle=styles[i])

            if flag:
                h_legend = ax.legend()
                h_legend1 = ax1.legend(loc='upper center')
                flag = False
            plt.show()
            plt.pause(1)
            
            scheduler.step()
            dice = np.mean(dice)
            print(f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL DICE: {dice}")
            es(dice, model, model_path=f"{DIRS['EXP_DIR']}/model/bst_model{IMG_SIZE}_fold{FOLD_ID}_{np.round(dice, 4)}.bin")
            best_model = f"../data/bst_model{IMG_SIZE}__fold{FOLD_ID}_{np.round(es.best_score,4)}.bin"
            if es.early_stop:
                print('\n\n -------------- EARLY STOPPING -------------- \n\n')
                break




