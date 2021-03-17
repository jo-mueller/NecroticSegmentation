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
# from Utils.Losses import ComboLoss, MixedLoss, LovaszLossSigmoid
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
        image     = cv2.imread(img_path, 1)
        mask      = np.argmax(tf.imread(mask_path), axis=0)
        
        # apply augmentations
        if self.augmentation:
            sample = {"image": image, "mask": mask}
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return {'image': image, 'mask' : mask}
        
    def __len__(self):
        return len(self.image_ids)


def train_one_epoch(train_loader, model, optimizer, loss_fn, accumulation_steps=1, device='cuda'):
    losses = helper_functions.AverageMeter()
    model = model.to(device)
    model.train()
    
    if accumulation_steps > 1: 
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=len(train_loader))
    
    for b_idx, data in enumerate(tk0):
        for key, value in data.items():
            data[key] = value.to(device)
            
        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()

        out  = model(data['image'])
        loss = loss_fn(out, torch.argmax(data['mask'], dim=1))
        
        # backpropagation
        with torch.set_grad_enabled(True):
            loss.backward()
            if (b_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        # update weights
        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])
    return losses.avg



root = os.getcwd()
src = root + r'\src\QuPath_Tiling\tiles\N182a_SAS_1_1_170905-Scene-1-ScanRegion0_HE'

# Config
FOLD_ID = 4
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 2
USE_SAMPLER = False
SAMPLER  = None
num_workers = 0
LEARNING_RATE = 2e-5
criterion = BCEWithLogitsLoss()
TRAIN_MODEL = True
EPOCHS = 300
IMG_SIZE = 512
EVALUATE = True
N_CLASSES = 3
PATIENCE = 20


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
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3,5,6,7,8,9,10,11,13,15], gamma=0.75)

    es = helper_functions.EarlyStopping(patience=PATIENCE, mode='max')
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax1 = ax.twinx()
    plt.ion()
    
    train_score = []
    test_score = []
    h_train, h_test, h_legend = None, None, None
    flag = True
    
    
    if TRAIN_MODEL:
        for epoch in range(EPOCHS):
            
            # create 5 folds train file
            kf = KFold(n_splits=5, shuffle=True)
            df['kfold']=-1
            for fold, (train_index, test_index) in enumerate(kf.split(df.Image_ID)):
                    df.loc[test_index, 'kfold'] = fold
            df.to_csv(DIRS['KFOLD'], index=False)
            
            # single fold training for now, rerun notebook to train for multi-fold
            df = pd.read_csv(DIRS['KFOLD'])
            TRAIN_DF = df.query(f'kfold!={FOLD_ID}').reset_index(drop=True)
            VAL_DF   = df.query(f'kfold=={FOLD_ID}').reset_index(drop=True)
            
            # Train transforms
            TFMS = albu.Compose([albu.HorizontalFlip(),
                                 albu.Rotate(10),
                                 albu.Normalize(),
                                 ToTensor()])
            
            # Test transforms
            TEST_TFMS = albu.Compose([albu.Normalize(), ToTensor(), ])
            
            # train dataset
            train_dataset = Dataset(TRAIN_DF, src, TFMS)
            val_dataset   = Dataset(VAL_DF, src, TEST_TFMS)
            
            # dataloaders
            train_dataloader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, 
                                          num_workers=num_workers)
            val_dataloader   = DataLoader(val_dataset, VALID_BATCH_SIZE, shuffle=False, num_workers=num_workers)
            
            
            loss = train_one_epoch(train_dataloader, model, optimizer, criterion)
            dice = helper_functions.evaluate(val_dataloader, model,
                                             metric=helper_functions.metric,
                                             n_classes=N_CLASSES)
            
            train_score.append(loss)
            test_score.append(dice)
                
            h_train = ax.plot(train_score, color='orange', label='Training loss')
            h_test = ax1.plot(test_score, color='blue', label='Dice validation score')
            if flag:
                h_legend = ax.legend()
                h_legend1 = ax1.legend(loc='upper center')
                flag = False
            plt.show()
            plt.pause(1)
            
            scheduler.step()
            print(f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL DICE: {dice}")
            es(dice, model, model_path=f"{DIRS['EXP_DIR']}/model/bst_model{IMG_SIZE}_fold{FOLD_ID}_{np.round(dice,4)}.bin")
            best_model = f"../data/bst_model{IMG_SIZE}__fold{FOLD_ID}_{np.round(es.best_score,4)}.bin"
            if es.early_stop:
                print('\n\n -------------- EARLY STOPPING -------------- \n\n')
                break




