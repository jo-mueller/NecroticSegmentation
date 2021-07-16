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

from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
from Utils import helper_functions

from torch.utils.data import DataLoader, WeightedRandomSampler
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
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = tf.imread(mask_path)
        if len(mask.shape) == 3:
            mask = np.argmax(mask, axis=0)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask= sample['image'], sample['mask']
            
        return {'image': image.transpose((2,0,1)), 'mask' : mask[None,: , :]}
    
        
    def __len__(self):
        return len(self.image_ids)


root = os.getcwd()
src = r'E:\Promotion\Projects\2021_Necrotic_Segmentation\src\IF\tiles_256_0.44'

# Config
BATCH_SIZE =31
USE_SAMPLER = False
SAMPLER  = None
num_workers = 0
LEARNING_RATE = 2e-5
criterion = CrossEntropyLoss()
EPOCHS = 150
IMG_SIZE = int(os.path.basename(src).split('_')[1])
PIX_SIZE = float(os.path.basename(src).split('_')[2])
EVALUATE = True
PATIENCE = 10
device= 'cuda'
keep_checkpoints = True
sampling = True

INFERENCE_MODE = False

# label_names = kwargs.get('label_names', dict({0: 'Background',
#                                               1: 'Non-Vital',
#                                               2: 'Vital',
#                                               3: 'SMA',
#                                               4: 'CutArtifact'}))
label_names = dict({0: 'Background',
                    1: 'Background2',
                    2: 'Vessels',
                    3: 'Hypoxia',
                    4: 'Perfusion'})

if __name__ == '__main__':
    
    # Create paths for train/test image data
    DIRS = helper_functions.createExp_dir(root + '/data')
    
    # Read label tiles to dataframe
    df, n_classes = helper_functions.scan_tile_directory(src, remove_empty_tiles=True)
    
    # Get occurrences of labels
    fig, weights = helper_functions.get_label_weights(df, n_classes=n_classes, label_names=label_names)
    
    # save characteristics
    fig.savefig(os.path.join(DIRS['EXP_DIR'], 'Class_distributions.png'))
    plt.close(fig)
    
    df_file = os.path.join(os.path.dirname(src), os.path.basename(src) + '.csv')
    if not os.path.exists(df_file):
        df.to_csv(df_file)
    
    # load segmentation model
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=n_classes, 
        activation=None,
    )
    model = model.to(device)
    
    # Specify augmentations
    aug_train = A.Compose([
        A.VerticalFlip(p=0.5),     
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ])
    
    aug_test = A.Compose([])
    
    # Specify optimizer & early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(0,1000, 20), gamma=0.75)
    es = helper_functions.EarlyStopping(patience=PATIENCE, mode='max')    
    
    # create 1/5th train/test split according to parent image
    kf = KFold(n_splits=5, shuffle=True)
    parents = df.Parent_image.unique()
    train_index, test_index = next(kf.split(parents), None)
    train_img, test_img = parents[train_index], parents[test_index]
    
    for i, entry in enumerate(df.iterrows()):
        df.loc[i, 'Cohort'] = 'Train' if any([x in entry[1].Parent_image for x in train_img]) else 'Test'
    
    # single fold training
    TRAIN_DF = df[df.Cohort == 'Train'].reset_index(drop=True)
    VAL_DF   = df[df.Cohort == 'Test'].reset_index(drop=True)
    
    # visualize class distributions
    fig_train, _ = helper_functions.get_label_weights(TRAIN_DF, label_names=label_names)
    fig_val, _ = helper_functions.get_label_weights(VAL_DF, label_names=label_names)
    
    fig_train.savefig(os.path.join(DIRS['EXP_DIR'], 'Class_distributions_train.png'))
    plt.close(fig_train)
    fig_val.savefig(os.path.join(DIRS['EXP_DIR'], 'Class_distributions_test.png'))
    plt.close(fig_val)
    
    
    # train/validation dataset
    train_dataset = Dataset(TRAIN_DF, src, aug_train)
    val_dataset   = Dataset(VAL_DF, src, aug_test)
    
    # Create samplers
    if sampling:
        weighted_sampler_train = WeightedRandomSampler(
                                    weights=torch.from_numpy(TRAIN_DF.Weight.to_numpy()),
                                    num_samples=len(TRAIN_DF),
                                    replacement=True)
        weighted_sampler_test = WeightedRandomSampler(
                                    weights=torch.from_numpy(VAL_DF.Weight.to_numpy()),
                                    num_samples=len(VAL_DF),
                                    replacement=True)
        shuffle_train = False
        shuffle_test = False

    else:
        shuffle_train = True
        shuffle_test = False
        weighted_sampler_test = None
        weighted_sampler_train = None
    
    # dataloaders & PerformanceMeter
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=num_workers,
                                  sampler=weighted_sampler_train)
    
    val_dataloader   = DataLoader(dataset=val_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=num_workers,
                                  sampler=weighted_sampler_test)
    
    Monitor = helper_functions.PerformanceMeter(n_classes=n_classes, label_names=label_names)
    
    # score lists
    train_score = []
    test_score = []
    # seen_labels =  np.zeros((n_classes))
    
    # train
    for epoch in range(EPOCHS):
        
        model.train()
        optimizer.zero_grad()
        tk0 = tqdm(train_dataloader, total=len(train_dataloader))
        
        for b_idx, data in enumerate(tk0):
            
            # move images on GPU
            for key, value in data.items():
                data[key] = value.to(device).float()
                
            # # Count encountered labels
            # for idx in range(data['mask'].shape[0]):
            #     _labels = torch.unique(data['mask'][idx]).cpu().numpy().astype(int)
            #     seen_labels[_labels] += 1
                
            # train
            data['prediction']  = model(data['image'])
            loss = criterion(data['prediction'], data['mask'][:, 0].long())
            loss.backward()
            optimizer.step()
            tk0.set_postfix(loss=loss.cpu().detach().numpy())

        # evaluate
        model.eval()
        with torch.no_grad():
            
            tk1 = tqdm(val_dataloader, total=len(val_dataloader))
            dice = np.zeros(n_classes)
            for b_idx, data in enumerate(tk1):
                
                # Eval
                data['prediction'] = model(data['image'].to(device).float())                 
                out = torch.argmax(data['prediction'], dim=1).view(-1)
                mask = data['mask'].view(-1)
                
                score = jaccard_score(mask.cpu(), out.cpu(), average=None,
                                      labels=np.arange(0, n_classes, 1))
                dice += score
                tk1.set_postfix(score=np.mean(dice))
                
                # visualize some samples
                if np.random.rand() > 0.98:
                    fig_batch = helper_functions.visualize_batch(data, epoch, loss.cpu().detach().numpy(), score.mean(), n_classes=n_classes)
                    fig_batch.savefig(os.path.join(DIRS['Performance'],
                                                   f'Batch_visualization_{epoch}_EP{epoch}_Dice{np.round(score.mean())}.png'))
                    plt.close(fig_batch)
            
            dice /= len(tk1)
            train_score.append(loss.cpu().detach())
            test_score.append(dice)
        
        # Plot progress
        Monitor.update(train_loss=loss.cpu().detach(), valid_score=dice)
        dice = np.mean(dice)
        
        # Scheduling and early stopping
        scheduler.step()
        print(f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL DICE: {dice}")
        es(epoch, dice, model, optimizer,
           model_path=f"{DIRS['EXP_DIR']}/model/bst_model{IMG_SIZE}_{np.round(dice, 4)}.bin")
        best_model = f"bst_model{IMG_SIZE}_{np.round(es.best_score,4)}.bin"
        
        # save performance data and config
        Monitor.figure.savefig(os.path.join(DIRS['Performance'], 'Training_Validation_Loss.png'))
        Monitor.finalize(os.path.join(DIRS['Performance'], 'Training_Validation_Loss.csv'))

        if es.early_stop:
            print('\n\n -------------- EARLY STOPPING -------------- \n\n')
            break

    # keep only best model
    models =  os.listdir(DIRS['Models'])
    for m in models:
        if not m == best_model:
            os.remove(os.path.join(DIRS['Models'], m))
            

    
    Config = {
    'Hyperparameters': {
        'BATCH_SIZE' : BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'CRITERION': criterion.__str__(),
        'EPOCHS': EPOCHS,
        'N_CLASSES': n_classes,
        'PATIENCE': PATIENCE},
    'Input':{
        'IMG_SIZE': IMG_SIZE,
        'PIX_SIZE': PIX_SIZE},
    'Output':{
        'Best_model': os.path.join(DIRS['Models'], best_model)
        },
    'Labels': label_names
    }
    
    with open(os.path.join(DIRS['EXP_DIR'], "params.yaml"), 'w') as yamlfile:
        data = yaml.dump(Config, yamlfile)
    


