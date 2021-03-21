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
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp
from Utils import helper_functions

from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import torch
from torch.nn import CrossEntropyLoss
import albumentations as A

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
        image = cv2.imread(img_path, 1).astype(float)
        image[:,(np.sum(image, axis=0) == 0)] = 255
        
        # mask = tf.imread(mask_path)
        mask = np.argmax(tf.imread(mask_path), axis=0)
        # mask = mask[None, :, :]
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask= sample['image'], sample['mask']
            
        
        
        return {'image': image.transpose((2,0,1)), 'mask' : mask[None,: , :]}
        
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
EPOCHS = 150
IMG_SIZE = 512
EVALUATE = True
N_CLASSES = 3
PATIENCE = 40
device= 'cuda'
keep_chekpoints = True

styles = ['--', '-', ':']


if __name__ == '__main__':
    
    # Create paths for train/test image data
    DIRS = helper_functions.createExp_dir(root + '/data')  
    
    # Read label tiles to dataframe
    df = helper_functions.scan_directory(src, img_ID='-labelled')
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=N_CLASSES, 
        activation=None,
    )
    
    model = model.to(device)
    model.train()
    
    aug_train = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5)         
        ], p=0.8)
    ])
    
    aug_test = A.Compose([
        # A.Normalize()
        # A.ToTensorV2()
    ])

    

    
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3,5,6,7,8,9,10,11,13,15], gamma=0.75)

    es = helper_functions.EarlyStopping(patience=PATIENCE, mode='max')
    
    # create 5 folds train/test groups
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    df['kfold']=-1
    for fold, (train_index, test_index) in enumerate(kf.split(X = df.Image_ID, y=df.has_all_labels)):
            df.loc[test_index, 'kfold'] = fold
    
    # single fold training for now, rerun notebook to train for multi-fold
    TRAIN_DF = df.query(f'kfold!={FOLD_ID}').reset_index(drop=True)
    VAL_DF   = df.query(f'kfold=={FOLD_ID}').reset_index(drop=True)
    
    # train/validation dataset
    train_dataset = Dataset(TRAIN_DF, src, aug_train)
    val_dataset   = Dataset(VAL_DF, src, aug_test)
    
    # dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax1 = ax.twinx()
    plt.ion()
    
    train_score = []
    test_score = []
    h_train, h_legend = None, None
    h_test  = [None for x in range(N_CLASSES)]
    flag = True
    
    if TRAIN_MODEL:
        for epoch in range(EPOCHS):
            
            # train
            optimizer.zero_grad()
            tk0 = tqdm(train_dataloader, total=len(train_dataloader))
            
            visualize_flag = True
            for b_idx, data in enumerate(tk0):
                
                # move images on GPU
                for key, value in data.items():
                    data[key] = value.to(device).float()
                    
                # train
                
                data['prediction']  = model(data['image'])
                loss = criterion(data['prediction'], data['mask'][:, 0].long())
                loss.backward()
                optimizer.step()
                tk0.set_postfix(loss=loss.cpu().detach().numpy())
                
                # try:
                #     if np.mean(test_score[-1]) > 0.65 and visualize_flag:
                        
                #         helper_functions.visualize_batch(data)
                #         visualize_flag = False
                # except Exception:
                #     pass

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
                    tk1.set_postfix(score=np.mean(dice))
                
                dice /= len(tk1)
                train_score.append(loss.cpu().detach())
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

    # # keep only most recent model
    # if not keep_chekpoints:
    #     models =  os.listdir(DIRS['Models'])
    #     models = [os.path.join(DIRS['Models'], x) for x in models]
    #     for model in models:
    #         if not model
    #     [os.path.remove(os.path.join(DIRS['Models'] + model))]
    


