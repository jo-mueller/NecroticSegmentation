# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:57:53 2021

Makes a nice train/test loss valid score plot

@author: johan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

root = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
exps = os.listdir(root)

line_styles = {0: '-', 1: '--', 2: ':'}
line_labels = {0: 'Background', 1: 'Necrosis', 2: 'Vital'}

fig, ax = plt.subplots(1,1)

DFs = []
for exp in exps:
    df = pd.read_csv(os.path.join(root, exp, 'Performance', 'Training_Validation_Loss.csv'))
    DFs.append(df)
    
    ax.plot(df['Epoch'], df['Training Loss'], color='orange', alpha=0.1)
    


df = pd.concat(DFs, axis=1)
means = df.stack().groupby(level=[0,1]).mean().unstack()
stds = df.stack().groupby(level=[0,1]).std().unstack()

ax.plot(means['Epoch'], means['Training Loss'], color='orange')
ax.fill_between(means['Epoch'],
                means['Training Loss'] - stds['Training Loss'],
                means['Training Loss'] + stds['Training Loss'], color='orange',
                alpha=0.3)

ax1 = ax.twinx()
labels = ['Validation Loss 1', 'Validation Loss 2', 'Validation Loss 3']
for i, label in enumerate(labels):
    ax1.plot(means['Epoch'], means[label], color='blue',
             label=line_labels[i], linestyle=line_styles[i])
    ax1.fill_between(means['Epoch'],
                means[label] - stds[label],
                means[label] + stds[label], color='blue',
                alpha=0.3)
    
ax.set_ylabel('Training loss')
ax1.set_ylabel('Dice validation score')
