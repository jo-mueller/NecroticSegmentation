# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:48:39 2021

@author: johan
"""

import os
from tqdm import tqdm
import numpy as np
import tifffile as tf
import cv2
import matplotlib.pyplot as plt

import yaml

from sklearn.model_selection import StratifiedKFold, KFold
import segmentation_models_pytorch as smp
from Utils import helper_functions

from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import torch
from torch.nn import CrossEntropyLoss
import albumentations as A

class Dataset():
    def __init__(self, df, image_base_dir, augmentation=None):
        self.df             = df
        self.image_base_dir = image_base_dir
        self.image_ids      = df.Image_ID.values
        self.augmentation   = augmentation
    
    def __getitem__(self, i):
        
        mask_path = os.path.join(self.image_base_dir, self.df.Mask_ID.loc[i])
        img_path = os.path.join(self.image_base_dir, self.df.Image_ID.loc[i])
        
        image = cv2.imread(img_path, 1).astype(np.float32)
        mask = np.argmax(tf.imread(mask_path), axis=0)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask= sample['image'], sample['mask']
            
        return {'image': image.transpose((2,0,1)), 'mask' : mask[None,: , :]}
        
    def __len__(self):
        return len(self.image_ids)


root = os.getcwd()
src = root + r'\src\QuPath_Tiling\tiles_256_2.5'

# Config
FOLD_ID = 4
BATCH_SIZE =20
USE_SAMPLER = False
SAMPLER  = None
num_workers = 0
LEARNING_RATE = 2e-5
criterion = CrossEntropyLoss()
EPOCHS = 150
IMG_SIZE = int(os.path.basename(src).split('_')[1])
PIX_SIZE = float(os.path.basename(src).split('_')[2])
EVALUATE = True
N_CLASSES = 3
PATIENCE = 20
device= 'cuda'
keep_checkpoints = True

# training visualization
N_visuals = 10
n_visuals = 0

if __name__ == '__main__':
    
    # Create paths for train/test image data
    DIRS = helper_functions.createExp_dir(root + '/data')  
    
    # Read label tiles to dataframe
    df = helper_functions.scan_directory(src)
    
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
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.Normalize()
        A.RandomBrightnessContrast(p=0.2)
    ])
    
    aug_test = A.Compose([
        # A.Normalize()
        ])

    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                      milestones=[20, 40, 60, 80, 100],
                                                      gamma=0.75)

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
    
    # dataloaders & PerformanceMeter
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=num_workers)
    Monitor = helper_functions.PerformanceMeter()
    
    # score lists
    train_score = []
    test_score = []
    
    # train
    for epoch in range(EPOCHS):
        
        optimizer.zero_grad()
        tk0 = tqdm(train_dataloader, total=len(train_dataloader))
        
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

        # evaluate
        with torch.no_grad():
            tk1 = tqdm(val_dataloader, total=len(val_dataloader))
            dice = np.zeros(N_CLASSES)
            for b_idx, data in enumerate(tk1):
                
                # Eval
                data['prediction'] = model(data['image'].to(device).float())                 
                out = torch.argmax(data['prediction'], dim=1).view(-1)
                mask = data['mask'].view(-1)
                
                score = jaccard_score(mask.cpu(), out.cpu(), average=None)
                dice += score
                tk1.set_postfix(score=np.mean(dice))
                
                # visualize some samples
                if np.random.rand() > 0.99 and n_visuals < N_visuals:
                    fig_batch = helper_functions.visualize_batch(data, epoch, loss.cpu().detach().numpy(), score.mean())
                    fig_batch.savefig(os.path.join(DIRS['Performance'],
                                                   f'Batch_visualization_{n_visuals}_EP{epoch}_Dice{np.round(score.mean())}.png'))
                    plt.close(fig_batch)
                    n_visuals += 1
            
            dice /= len(tk1)
            train_score.append(loss.cpu().detach())
            test_score.append(dice)
        
        # Plot progress
        Monitor.update(train_loss=loss.cpu().detach(), valid_score=dice)
        dice = np.mean(dice)
        
        # Scheduling and early stopping
        scheduler.step()
        print(f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL DICE: {dice}")
        es(dice, model, model_path=f"{DIRS['EXP_DIR']}/model/bst_model{IMG_SIZE}_fold{FOLD_ID}_{np.round(dice, 4)}.bin")
        best_model = f"bst_model{IMG_SIZE}_fold{FOLD_ID}_{np.round(es.best_score,4)}.bin"
        if es.early_stop:
            print('\n\n -------------- EARLY STOPPING -------------- \n\n')
            break

    # keep only best model
    models =  os.listdir(DIRS['Models'])
    for model in models:
        if not model == best_model:
            os.remove(os.path.join(DIRS['Models'], model))
            
    # save performance data and config
    Monitor.figure.savefig(os.path.join(DIRS['Performance'], 'Training_Validation_Loss.png'))
    Monitor.finalize(os.path.join(DIRS['Performance'], 'Training_Validation_Loss.csv'))
    
    Config = {
    'Hyperparameters': {
        'BATCH_SIZE' : BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'CRITERION': criterion.__str__(),
        'EPOCHS': EPOCHS,
        'N_CLASSES': N_CLASSES,
        'PATIENCE': PATIENCE},
    'Input':{
        'IMG_SIZE': IMG_SIZE,
        'PIX_SIZE': PIX_SIZE},
    'Output':{
        'Best_model': os.path.join(DIRS['Models'], best_model)
        }
    }
    
    with open(os.path.join(DIRS['EXP_DIR'], "params.yaml"), 'w') as yamlfile:
        data = yaml.dump(Config, yamlfile)
    
    


