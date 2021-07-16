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
    def __init__(self, n_classes, **kwargs):
        
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
        
        self.styles = kwargs.get('styles', {0: 'solid', 1: 'dotted', 2: 'dashed',
                                            3: 'dashdot', 4: (0, (3, 1, 1, 1))})
        self.labels = kwargs.get('label_names', {0: 'Background', 1: 'Necrosis', 2: 'Vital',
                                            3: 'SMA', 4: 'Cut Artifact'})
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


def scan_database(directory, img_type='czi', identifier='HE'):
    """
    Scans a radiomics database and finds HE images
    """
    images = []
    dirs = []
    for root, subdirs, filenames in os.walk(directory):
        
        for filename in filenames:
            if 'NK' in root:
                continue
            if identifier in filename and filename.endswith(img_type):
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
    

def scan_tile_directory(directory, **kwargs):
    
    """
    Scans directory with mixed image/label images for label images.
    Has to be tif. Returns a dataset with all images.
    """
    
    pardir = os.path.dirname(directory)
    dset_name = os.path.basename(directory)
    
    df_file = os.path.join(pardir, dset_name + '.csv')
    if os.path.exists(df_file):
        df = pd.read_csv(df_file, dtype={'Mask_ID': str,
                                         'Image_ID': str,
                                         'Parent': str,
                                         'Omitted': bool,
                                         '0': bool,
                                         '1': bool,
                                         '2': bool,
                                         '3': bool,
                                         '4': bool,
                                         '5': bool,
                                         'Weight': float},
                           converters={"Labels": lambda string: [int(x) for x in string.strip('[]').split(' ')]})
        
        classes = np.unique(np.hstack(df.Labels))
        print('Found existing dataframe for source directory.')
        return df, len(classes)
    
    img_type = kwargs.get('img_type', 'tif')
    img_ID = kwargs.get('img_ID', '')
    mask_ID = kwargs.get('mask_ID', '-labelled')
    
    remove_empty_tiles = kwargs.get('remove_empty_tiles', True)
    
    df = pd.DataFrame(columns=['Mask_ID', 'Image_ID', 'Parent', 'Labels',
                               'Omitted'])
    
    files = [x for x in os.listdir(directory) if x.endswith(img_type)]
    df['Mask_ID'] = [x for x in files if mask_ID in x]
    # images = [x.replace(mask_ID, img_ID) for x in masks]
    # df['Image_ID'] = images
    
    for i, sample in tqdm(df.iterrows(), desc='Browsing tiles', total=len(df)):
        # Find the raw image counterpart of this image and store in df
        image = sample['Mask_ID'].replace(mask_ID, '')
        if image in files:
            df.loc[i, 'Image_ID'] = image
        else:
            df.loc[i, 'Image_ID'] = np.nan
        
        # Determine parent image of tile
        parent = sample.Image_ID.split(' ')[0]
        df.loc[i, ('Parent')] = parent
        
        # Get mask and image, check dimensions
        mask = tf.imread(os.path.join(directory, sample.Mask_ID))
        image = tf.imread(os.path.join(directory, sample.Image_ID))
        
        # get present labels and write to df
        if len(mask.shape) == 3:
            present_labels = np.unique(np.argmax(mask, axis=0))
        else:
            present_labels = np.unique(mask)
        df.loc[i, ('Labels')] = present_labels
        
        # Check for bullshit background
        prjctn = np.sum(image, axis=2)
        df.loc[i, ('Omitted')] = True if (prjctn == 0).sum() > 0 or (prjctn == 3*255).sum() > 0 else False
        
    # remove the omitted tiles from the data directory and the dataframe
    if remove_empty_tiles:
        
        if (df.Omitted == True).sum() > 0:        
            n_rmvd = 0
            tk = tqdm(df.iterrows(), desc='Removing empty tiles')
            for i, sample in tk:
                if sample.Omitted:
                    f_mask = os.path.join(directory, sample.Mask_ID)
                    f_img = os.path.join(directory, sample.Image_ID)
                    
                    os.remove(f_mask)
                    os.remove(f_img)
                    tk.set_postfix_str(f'Removed {n_rmvd} tiles')
                    n_rmvd += 1
    df = df[df.Omitted == False]
    
    # Add separate column for each label
    classes = np.unique(np.hstack(df.Labels))
    df = df.reindex(columns = ['Parent_image'] + df.columns.tolist() + [f'{x}' for x in classes])
    
    for i, sample in tqdm(df.iterrows(), desc='Assigning labels'):
        for j in classes:
            if j in sample.Labels:
                df.loc[i, (f'{j}')] = True
            else:
                df.loc[i, (f'{j}')] = False
    
    for i, sample in tqdm(df.iterrows(), desc='Assigning parent images'):
        parent = sample.Image_ID.split(' [')[0]
        df.loc[i, ('Parent_image')] = parent
    
    return df, len(classes)

def get_label_weights(df, **kwargs):
    
    """
    Fuunction that counts how often each present label in the dataframe occurs
    """
    
    label_names = kwargs.get('label_names', dict({0: 'Background',
                                                  1: 'Non-Vital',
                                                  2: 'Vital',
                                                  3: 'SMA',
                                                  4: 'CutArtifact'}))
    n_classes = kwargs.get('n_classes', 3)
    classes = [x for x in label_names.keys()]
    occurrences = []
    
    for idx in classes:
        n = df[f'{idx}'].sum()
        occurrences.append(n)
        
    fig, axes = plt.subplots(nrows=1, ncols=2)        
    ax = axes[0]
    ax.bar(classes, occurrences, color=['orange', 'blue', 'green', 'magenta', 'cyan'])
    ax.set_ylim(0, ax.get_ylim()[1]*1.25)  # add 25% more y-range
    
    for i in range(n_classes):
        ax.text(classes[i], occurrences[i],
                '{:s}\nn={:d}'.format(label_names[classes[i]], occurrences[i]),
                rotation=90,
                verticalalignment='bottom',
                horizontalalignment='center')
        
    ax.set_ylabel('Label occurrence')
    ax.set_xticks([])
    
    # Plot weights        
    ax1 = axes[1]
    weights = [1.0/x for x in occurrences]
    ax1.bar(classes, weights, color=['orange', 'blue', 'green', 'magenta', 'cyan'])
    ax1.set_ylim(0, ax1.get_ylim()[1]*1.35)  # add 25% more y-range
    
    for i in range(n_classes):
        ax1.text(classes[i], weights[i],
                '{:s}\nweight={:.2e}'.format(label_names[classes[i]], weights[i]),
                rotation=90,
                verticalalignment='bottom',
                horizontalalignment='center')

    ax1.set_ylabel('Label weight')
    ax1.set_xticks([])
    
    fig.tight_layout()
    
    if not 'Weight' in df.columns:
        # Implement sampling: assign weight to each sample
        for i, sample in tqdm(df.iterrows(), desc='Assigning weights'):
            for idx in range(n_classes):
                if sample[f'{classes[idx]}'] == True:
                    df.loc[i, 'Weight'] = weights[idx]
    
    return fig, occurrences


def visualize_batch(sample, epoch, loss, mean_dice, **kwargs):
    """
    Function to create a visualization of images from a ingle batch
    """
    
    n_classes = kwargs.get('n_classes', 3)
    max_samples = kwargs.get('max_samples', 8)
    
    # get dimensions and available image types
    n_batch = sample['image'].size()[0]
    keys = list(sample.keys())
    
    if n_batch > max_samples:
        n_batch = max_samples
    fig, axes = plt.subplots(nrows=len(keys), ncols=n_batch, figsize=(2*n_batch, 2*len(keys)))
    
    # get data
    sample = {entry: sample[entry].cpu().detach().numpy().astype(float) for entry in sample.keys()}
    
    # iterate over images in batch
    for ibx in range(n_batch):
        
        # iterate over keys (e.g. raw, mask, prediction)
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
                for im in axes[k, 0].get_images():
                    im.set_clim(0, n_classes)
                    
            elif key == 'prediction':
                if n_classes > 3:
                    pred = torch.argmax(torch.tensor(img), dim=2)
                else:
                    pred = img
                    
                axes[k, ibx].imshow(pred)
                for im in axes[k, ibx].get_images():
                    im.set_clim(0, n_classes)
                    
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