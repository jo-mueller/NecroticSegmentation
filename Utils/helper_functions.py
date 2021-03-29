# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:03:06 2021

@author: johan
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import tifffile as tf
import cv2
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F

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
            

def evaluate(valid_loader, model, device='cuda', n_classes=3, metric=metric):
    losses = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            out = torch.argmax(model(data['image']), dim=1)
            dice = metric(out, data['mask']).cpu()
            losses.update(dice.mean().item(), valid_loader.batch_size)
            tk0.set_postfix(dice_score=losses.avg)
    return losses.avg

def createExp_dir(root):
    
    time_string = '{:d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(
                    datetime.now().year,
                    datetime.now().month,
                    datetime.now().day,
                    datetime.now().hour,
                    datetime.now().minute,
                    datetime.now().second)
    
    base = os.path.join(root, 'Experiment_' + time_string)
    dir_test = os.path.join(base, 'Test')
    dir_train = os.path.join(base, 'Train')
    os.mkdir(base)
    
    os.mkdir(dir_train)
    os.mkdir(dir_test)
    
    dirs = {'EXP_DIR': base,
            'TRAIN_IMG_DIR': dir_train,
            'TEST_IMG_DIR': dir_test,
            'KFOLD': os.path.join(base, 'RLE_kfold.csv'),
            'Models': os.path.join(base, 'model')}
    
    return dirs


def scan_database(directory, img_type='czi'):
    """
    Scans a radiomics database and finds HE images
    """
    images = []
    dirs = []
    for root, subdirs, filenames in os.walk(directory):
        
        for filename in filenames:
            if 'NK' in root:
                continue
            if 'HE' in filename and filename.endswith(img_type):
                images.append(os.path.join(root, filename))
                dirs.append(root)
    
    df = pd.DataFrame(columns=['Image_ID', 'Directory'])
    df.Image_ID = images
    df.Directory = dirs
    
    return df


def scan_directory(directory, img_ID='.tif', mask_ID = '-labelled.tif', img_type='tif', outname=None):
    """
    Scans directory with mixed image/label images for label images.
    Has to be tif. Returns a dataset with all images
    """
    
    df = pd.DataFrame(columns=['Mask_ID', 'Image_ID', 'has_all_labels'])
    
    files = os.listdir(directory)
    masks = [x for x in files if mask_ID in x]
    images = [x.replace(mask_ID, img_ID) for x in masks]
    df['Image_ID'] = images
    df['Mask_ID'] = masks
    
    to_delete = []
    
    for i, sample in tqdm(df.iterrows()):
        label = cv2.imread(os.path.join(directory, sample.Mask_ID), 1)
        label = np.argmax(label, axis=2)
        df.loc[i, ('has_all_labels')] = len(np.unique(label))
        
        img = tf.imread(os.path.join(directory, sample.Image_ID))
        if np.sum((img == 255) + (img == 0)) > 100:
            has_background = True
        else:
            has_background = False
            
        if has_background:
            to_delete.append(i)
            
    df.drop(to_delete, axis=0, inplace=True)
    df.loc[:, ('has_all_labels')] = df.loc[:, ('has_all_labels')] == 3
    
    return df.reset_index()

def visualize_batch(sample):
    n_batch = sample['image'].size()[0]
    keys = list(sample.keys())
    
    fig, axes = plt.subplots(nrows=len(keys), ncols=n_batch, figsize=(12, 4))
    
    for i, key in enumerate(keys):
        for ibx in range(n_batch):
            ax = axes[i, ibx]
            img = sample[key][ibx].cpu().detach().numpy().transpose((1,2,0))
            ax.imshow(img/img.max())
        ax.set_ylabel(key)
            
    
if __name__ == '__main__':
    root = r'E:\Promotion\Projects\2020_Radiomics\Data'
    df = scan_database(root)