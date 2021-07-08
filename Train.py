# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:48:39 2021

@author: johan
"""

# from aicsimageio import AICSImage
import os
from tqdm import tqdm
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

import yaml

from sklearn.model_selection import StratifiedKFold, KFold
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from Utils import helper_functions as hf


from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import jaccard_score
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import albumentations as A
# from albumentations.augmentations.transforms import PadIfNeeded

class Dataset():
    def __init__(self, df, image_base_dir, augmentation=None, **kwargs):
        
        self.one_hot_encoding = kwargs.get('one_hot_enc', False)
        
        self.df             = df
        self.image_base_dir = image_base_dir
        self.image_ids      = df.Image_ID.values
        self.augmentation   = augmentation
    
    def __getitem__(self, i):
        
        mask_path = os.path.join(self.image_base_dir, self.df.Mask_ID.loc[i])
        img_path = os.path.join(self.image_base_dir, self.df.Image_ID.loc[i])
        
        # image = cv2.imread(img_path, 1)
        # img = AICSImage(img_path)
        # msk = AICSImage(mask_path)
        
        # image = img.get_image_data("SYX", C=0, T=0, Z=0).astype('uint8')
        # mask = msk.get_image_data("SYX", C=0, T=0, Z=0).astype('uint8')
        
        image = tf.imread(img_path).transpose((2,0,1)).astype('uint8')
        mask = tf.imread(mask_path).astype('uint8')
        
        # if one hot encoding is disabled, the index with the highest label is
        # chosen as the label to be predicted
        if self.one_hot_encoding == False:
            mask = np.argmax(mask, axis=0)
            mask = mask[None, :, :]  # add empty channel dimension            
        
        # apply albumentations
        sample = {'image': image, 'mask' : mask}
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)      
            
        if sample['image'].shape[0] != 3:
            sample['image'] = sample['image'].transpose((1,0,2))
            sample['mask'] = sample['mask'].transpose((1,0,2))
            
        return sample
        
    def __len__(self):
        return len(self.image_ids)


root = os.getcwd()
src = root + r'\src\QuPath_Tiling_Validated\tiles_256_2.5_nonVital_Vital_Hypoxia'

# Config
BATCH_SIZE = 25
USE_SAMPLER = False
SAMPLER  = None
num_workers = 0
LEARNING_RATE = 2e-5
criterion = BCEWithLogitsLoss()
score_func = DiceLoss('multilabel')
EPOCHS = 200
IMG_SIZE = int(os.path.basename(src).split('_')[1])
PIX_SIZE = float(os.path.basename(src).split('_')[2])
OHE = True  # one hot encoding
PATIENCE = 20
device= 'cuda'
keep_checkpoints = True

if __name__ == '__main__':
    
    # Create paths for train/test image data
    DIRS = hf.createExp_dir(root + '/data')
    
    # Read label tiles to dataframe
    df, labels = hf.scan_tile_directory(src,
                                        remove_empty_tiles = True, 
                                        mask_ID = '_0Background_1Necrosis_2Vital_3Hypoxia')
    N_CLASSES = len(labels)
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=N_CLASSES, 
        activation=None,
    )
    
    model = model.to(device)
    
    aug_train = A.Compose([
        A.VerticalFlip(p=0.5),     
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.Normalize()
        A.RandomBrightnessContrast(p=0.2)
        # A.Normalize()
    ])
    
    aug_test = A.Compose([
        # A.Normalize()
        ])

    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                      milestones=[20, 40, 60, 80, 100],
                                                      gamma=0.75)

    es = hf.EarlyStopping(patience=PATIENCE, mode='max')
    
    # create 1/5th train/test split according to parent image
    kf = KFold(n_splits=5, shuffle=True)
    parents = df.Parent.unique()
    train_index, test_index = next(kf.split(parents), None)
    train_img, test_img = parents[train_index], parents[test_index]
    
    for i, entry in enumerate(df.iterrows()):
        df.loc[i, 'Cohort'] = 'Train' if any([x in entry[1].Parent for x in train_img]) else 'Test'
    
    # single fold training
    TRAIN_DF = df[df.Cohort == 'Train'].reset_index(drop=True).sample(frac=1)
    VAL_DF   = df[df.Cohort == 'Test'].reset_index(drop=True).sample(frac=1)
    
    # Apply class weighting
    ccount = [i for i in hf.get_class_distribution(TRAIN_DF, labels).values()]
    weights = 1.0/np.asarray(ccount)
    sampler = WeightedRandomSampler(weights, BATCH_SIZE)
    
    # train/validation dataset
    train_dataset = Dataset(TRAIN_DF, src, aug_train, one_hot_enc=OHE)
    val_dataset   = Dataset(VAL_DF, src, aug_test, one_hot_enc=OHE)
    
    # dataloaders & PerformanceMeter
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
    Monitor = hf.PerformanceMeter(n_classes=N_CLASSES)
    
    # score lists
    train_score = []
    test_score = []
    
    # train
    for epoch in range(EPOCHS):
        
        model.train()
        optimizer.zero_grad()
        tk0 = tqdm(train_dataloader, total=len(train_dataloader))
        
        for b_idx, data in enumerate(tk0):
            
            # move images on GPU
            for key, value in data.items():
                data[key] = value.to(device).float()
                
            # train
            data['prediction']  = model(data['image'])
            loss = criterion(data['prediction'].view(1,-1), data['mask'].view(1,-1))
            loss.backward()
            optimizer.step()
            tk0.set_postfix(loss=loss.cpu().detach().numpy())

        # evaluate
        model.eval()
        with torch.no_grad():
            
            tk1 = tqdm(val_dataloader, total=len(val_dataloader))
            score = 0
            for b_idx, data in enumerate(tk1):
                
                # Eval
                data['prediction'] = model(data['image'].to(device).float()) 
                
                score = score_func(data['mask'].to(device).view(-1,1), data['prediction'].view(-1,1))
                tk1.set_postfix(score)
                
                # visualize some samples
                if np.random.rand() > 0.98:
                    fig_batch = hf.visualize_batch(data, epoch, loss.cpu().detach().numpy(), score.mean(), n_classes=N_CLASSES)
                    fig_batch.savefig(os.path.join(DIRS['Performance'],
                                                   f'Batch_visualization_EP{epoch}_Dice{np.round(score.mean())}.png'))
                    plt.close(fig_batch)
            
            score /= len(tk1)
            train_score.append(loss.cpu().detach())
            test_score.append(score)
        
        # Plot progress
        Monitor.update(train_loss=loss.cpu().detach(), valid_score=score)
        score = np.mean(score)
        
        # Scheduling and early stopping
        scheduler.step()
        print(f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL DICE: {score}")
        es(epoch, score, model, optimizer,
           model_path=f"{DIRS['EXP_DIR']}/model/bst_model{IMG_SIZE}_{np.round(score, 4)}.bin")
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


