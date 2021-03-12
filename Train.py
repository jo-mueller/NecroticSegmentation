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

from pathlib import Path

import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

from sklearn.model_selection import KFold

from Utils import Unet
import segmentation_models_pytorch as smp
from Utils import helper_functions

from torch.utils.data import DataLoader
from Utils.Losses import ComboLoss, MixedLoss, LovaszLossSigmoid
import torch

class Dataset():
    def __init__(self, rle_df, image_base_dir, augmentation=None):
        self.df             = rle_df
        self.image_base_dir = image_base_dir
        self.image_ids      = rle_df.Image_ID.values
        self.augmentation   = augmentation
    
    def __getitem__(self, i):
        image_id  = self.image_ids[i]
        img_path  = os.path.join(self.image_base_dir, image_id)
        mask_path = img_path.replace('image', 'labels')
        image     = cv2.imread(img_path, 1)
        mask      = cv2.imread(mask_path, 1)     
        # apply augmentations
        if self.augmentation:
            sample = {"image": image, "mask": mask}
            sample = self.augmentation(**sample)
            image, mask = sample['image'], sample['mask']

        return {'image': image, 'mask' : mask}
        
    def __len__(self):
        return len(self.image_ids)
    
class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        model_path = Path(model_path)
        parent = model_path.parent
        os.makedirs(parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Model saved at at {}!".format(
                    self.val_score, epoch_score, model_path
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def train_one_epoch(train_loader, model, optimizer, loss_fn, accumulation_steps=1, device='cuda'):
    losses = AverageMeter()
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
        loss = loss_fn(out, data['mask'])
        
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

# pixel-wise accuracy
def acc_metric(input, target):
    inp = torch.where(input>0.5, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
    acc = (inp.squeeze(1) == target).float().mean()
    return acc

def metric(probability, truth, threshold=0.5, reduction='none'):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice

def evaluate(valid_loader, model, device='cuda', metric=metric):
    losses = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            out   = model(data['image'])
            out   = torch.sigmoid(out)
            dice  = metric(out, data['mask']).cpu()
            losses.update(dice.mean().item(), valid_loader.batch_size)
            tk0.set_postfix(dice_score=losses.avg)
    return losses.avg

root = os.getcwd()
src = root + '/src/Tiles'

# Config
FOLD_ID = 4
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
USE_SAMPLER = False
SAMPLER  = None
num_workers = 0
PRETRAINED = False
PRETRAINED_PATH = None
LEARNING_RATE = 2e-5
USE_CRIT = True
CRITERION =LovaszLossSigmoid()
TRAIN_MODEL = True
EPOCHS = 50
IMG_SIZE = 512
EVALUATE = True

if __name__ == '__main__':
    
    # Create paths for train/test image data
    DIRS = helper_functions.createExp_dir(root + '/data')  
    
    # Read labels to dataframe with RLE encoding
    df = helper_functions.scan_directory(src)
    # df = helper_functions.rle_directory(src, outname=DIRS['RLE_DATA'])
    
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
    train_dataloader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, 
                                  shuffle=True if not USE_SAMPLER else False, 
                                  num_workers=num_workers, 
                                  sampler=SAMPLER if USE_SAMPLER else None)
    val_dataloader   = DataLoader(val_dataset, VALID_BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    model = smp.Unet(
        encoder_name='se_resnext50_32x4d', 
        encoder_weights='imagenet', 
        classes=3, 
        activation=None,
    )
    
    if PRETRAINED: 
        model.load_state_dict(torch.load(PRETRAINED_PATH))
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3,5,6,7,8,9,10,11,13,15], gamma=0.75)
    
    criterion = MixedLoss(10.0, 2.0) if not USE_CRIT else CRITERION 
    es = EarlyStopping(patience=10, mode='max')
    
    if TRAIN_MODEL:
        for epoch in range(EPOCHS):
            loss = train_one_epoch(train_dataloader, model, optimizer, criterion)
            dice = evaluate(val_dataloader, model, metric=metric)
            scheduler.step()
            print(f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL DICE: {dice}")
            es(dice, model, model_path=f"../data/bst_model{IMG_SIZE}_fold{FOLD_ID}_{np.round(dice,4)}.bin")
            best_model = f"../data/bst_model{IMG_SIZE}__fold{FOLD_ID}_{np.round(es.best_score,4)}.bin"
            if es.early_stop:
                print('\n\n -------------- EARLY STOPPING -------------- \n\n')
                break
    if EVALUATE:
        valid_score = evaluate(val_dataloader, model, metric=metric)
        print(f"Valid dice score: {valid_score}")




