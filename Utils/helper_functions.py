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
import copy
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
# import torch.nn.functional as F

class PerformanceMeter:
    def __init__(self, n_classes):
        
        # Make plot for performance display
        self.figure, self.TrainAx = plt.subplots(nrows=1, ncols=1)
        self.ValidAx = self.TrainAx.twinx()
        plt.ion()
        
        self.train_score = []
        self.valid_score = []
        self.n_classes = n_classes
        
        # Prepare legend
        self.valid_curve = []
        self.loss_curve = self.TrainAx.plot([],
                                            color='orange',
                                            label='Training loss')[0]
        
        self.styles = {0: '-', 1: '--', 2: ':'}
        self.labels = {0: 'Background', 1: 'Necrosis', 2: 'Vital'}
        plt.show()
        
    def update(self, train_loss, valid_score):
        self.train_score.append(train_loss)
        self.valid_score.append(valid_score)
        
        # clear previous curves
        self.TrainAx.clear()
        self.ValidAx.clear()
        
        # Replot
        x = np.arange(0, len(self.train_score), 1)
        valid_score = np.concatenate(self.valid_score, axis=0).reshape(-1, self.n_classes)
        h_loss = self.TrainAx.plot(x, self.train_score,
                                   color='orange',
                                   label='Training loss')[0]
        h_valid = []
        
        for i in range(self.n_classes):
            h_valid.append(self.ValidAx.plot(x, valid_score[:,i],
                                             color='blue',
                                             linestyle=self.styles[i],
                                             label=self.labels[i])[0])
            
        # Legend
        lines = [h_loss] + h_valid
        labels = ['Training loss'] + [self.labels[i] for i in self.labels.keys()]
        self.TrainAx.legend(lines, labels, loc='lower center',
                            bbox_to_anchor=(0.5, 1.03), fancybox=True,
                            shadow=True, ncol=self.n_classes+1)
        
        # Ax labels
        self.TrainAx.set_ylabel('Training loss')
        self.ValidAx.set_ylabel('Dice validation score')
        self.ValidAx.set_ylim(0,1)
        self.TrainAx.set_xlabel('Epoch [#]')
        
        plt.show()
        plt.pause(0.05)
        
    def finalize(self, path):
        epochs = np.arange(0, len(self.train_score), 1)
        
        df = pd.DataFrame(columns=['Epoch',
                                   'Training Loss',
                                   'Validation Loss 1',
                                   'Validation Loss 2',
                                   'Validation Loss 3'])
        df.loc[:, ('Epoch')] = epochs
        valid_score = np.concatenate(self.valid_score, axis=0).reshape(-1, self.n_classes)
        df.loc[:, ('Training Loss')] = [x.numpy() for x in self.train_score]
        df.loc[:, ('Validation Loss 1')] = valid_score[:, 0]
        df.loc[:, ('Validation Loss 2')] = valid_score[:, 1]
        df.loc[:, ('Validation Loss 3')] = valid_score[:, 2]
        
        df.to_csv(path)
        
        return 1
        
        
        
class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model_instance = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch, epoch_score, model, optimizer, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, epoch_score, model, optimizer, model_path)
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
            self.save_checkpoint(epoch, epoch_score, model, optimizer, model_path)
            self.model_path = model_path
            self.counter = 0

    def save_checkpoint(self, epoch, epoch_score, model, optimizer, model_path):
        model_path = Path(model_path)
        parent = model_path.parent
        os.makedirs(parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Model saved at at {}!".format(
                    self.val_score, epoch_score, model_path
                )
            )
            self.best_model_instance = copy.deepcopy(model)  # keep instance of best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_score,
            }, model_path)
            # torch.save(model.state_dict(), model_path)
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
    dir_perf = os.path.join(base, 'Performance')
    
    os.mkdir(base)
    os.mkdir(dir_train)
    os.mkdir(dir_test)
    os.mkdir(dir_perf)
    
    dirs = {'EXP_DIR': base,
            'TRAIN_IMG_DIR': dir_train,
            'TEST_IMG_DIR': dir_test,
            'KFOLD': os.path.join(base, 'RLE_kfold.csv'),
            'Models': os.path.join(base, 'model'),
            'Performance': dir_perf}
    
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
    
def get_nlabels(label):
    
    chs = np.min(label.shape)
    label = label.reshape(chs, -1).sum(axis=1)
    return np.sum(label > 0) + 1
    

def scan_directory(directory, img_ID='.tif',
                   mask_ID = '-labelled.tif', img_type='tif', outname=None, **kwargs):
    """
    Scans directory with mixed image/label images for label images.
    Has to be tif. Returns a dataset with all images
    """
    
    remove_empty_tiles = kwargs.get('remove_empty_tiles', True)
    thin_empty_tiles = kwargs.get('thin_empty_tiles', 1)
    
    df = pd.DataFrame(columns=['Mask_ID', 'Image_ID', 'has_all_labels',
                               'Is_Background', 'Parent_image'])
    
    files = os.listdir(directory)
    masks = [x for x in files if mask_ID in x]
    images = [x.replace(mask_ID, img_ID) for x in masks]
    df['Image_ID'] = images
    df['Mask_ID'] = masks
    
    for i, sample in tqdm(df.iterrows()):
        
        # Determine parent image of tile
        parent = sample.Image_ID.split(' ')[0]
        label = tf.imread(os.path.join(directory, sample.Mask_ID))
        image = np.sum(tf.imread(os.path.join(directory, sample.Image_ID)), axis=2)
        
        df.loc[i, ('Parent_image')] = parent
        df.loc[i, ('has_all_labels')] = get_nlabels(label)
        df.loc[i, ('Is_Background')] = True if np.sum(label) == 0 else False
        df.loc[i, ('Is_OmittedTile')] = True if np.sum(image == 0) > 0 or np.sum(image == 3*255) > 0 else False
            
    if thin_empty_tiles != 1:
        for i, sample in df.iterrows():
            if sample.Is_Background or sample.Is_OmittedTile:
                f_mask = os.path.join(directory, sample.Mask_ID)
                f_img = os.path.join(directory, sample.Image_ID)
                
                if np.random.random() > thin_empty_tiles:
                    os.remove(f_mask)
                    os.remove(f_img)
                    
    
    if remove_empty_tiles:
        for i, sample in df.iterrows():
            if sample.Is_Background or sample.Is_OmittedTile:
                f_mask = os.path.join(directory, sample.Mask_ID)
                f_img = os.path.join(directory, sample.Image_ID)
                
                os.remove(f_mask)
                os.remove(f_img)
        
        df = df[df.Is_Background == False]
        
        df.loc[:, ('has_all_labels')] = df.loc[:, ('has_all_labels')] == 3
    
    return df.reset_index()

def visualize_batch(sample, epoch, loss, mean_dice, max_samples=8):
    n_batch = sample['image'].size()[0]
    keys = list(sample.keys())
    
    if n_batch > max_samples:
        n_batch = max_samples
    fig, axes = plt.subplots(nrows=len(keys), ncols=n_batch, figsize=(2*n_batch, 2*len(keys)))
    
    sample = {entry: sample[entry].cpu().detach().numpy().astype(float) for entry in sample.keys()}
    for ibx in range(n_batch):
        
        for k, key in enumerate(keys):
            
            img = sample[key][ibx].transpose((1,2,0))
            # Ground truth
            if key == 'image':
                img += np.finfo(float).eps
                axes[k, ibx].imshow((img - img.min())/(img.max() - img.min()))
                axes[k, 0].set_ylabel('Raw image')
            elif key == 'mask':
                img += np.finfo(float).eps
                axes[k, ibx].imshow(img)
                axes[k, 0].set_ylabel('Mask image')
            elif key == 'prediction':
                axes[k, ibx].imshow(torch.sigmoid(torch.tensor(img)))
                axes[k, 0].set_ylabel('Prediction')
            
            axes[k, ibx].axis('off')  # no ticks on subplots
    
    
    
    
    fig.tight_layout()
    
    plt.suptitle('Batch visualization (Epoch: {:d}, loss: {:.2f}, mean valid. score: {:.2f}'.format(
        epoch, loss, mean_dice))
    plt.pause(0.05)
    plt.subplots_adjust(top=0.95)
    
    return fig
            
    
# if __name__ == '__main__':
#     A = PerformanceMeter()
#     A.update(0.52, [0.9, 0.8, 0.7])
#     A.update(0.45, [0.9, 0.8, 0.7])
#     A.update(0.35, [0.9, 0.8, 0.7])