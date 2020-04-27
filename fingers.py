# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:23:17 2020

@author: tharshi
"""


# some useful frameworks
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import re

# define useful helper functions
def plot_im(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x) * h
    plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')

# make pretty
plt.style.use('seaborn')
#%% data preprocessing

# import training images
directory = '../finger-counter-data/images'

files = os.listdir(directory)
n_files = len(files)

images = []
labels = []

# data prepr
for i in tqdm(range(n_files)):
    file = files[i]
    path = directory + '/' + file
    if file.endswith(".jpg"):
        
        # load image as grayscale
        im = Image.open(path)
        
        # load corresponding xml
        metadata = os.path.splitext(path)[0] + '.xml'
        
        # get bounding boxes and labels
        with open(metadata, 'r') as xml:
            soup = BeautifulSoup(xml, 'lxml')
            
            x_min = int(soup.select('xmin')[0].text)
            x_max = int(soup.select('xmax')[0].text)
            y_min = int(soup.select('ymin')[0].text)
            y_max = int(soup.select('ymax')[0].text)
            
            label = int(re.sub('-hand', '', soup.select('name')[0].text))
        
        labels.append(label)
            
        # crop and resize image
        im = im.crop((x_min, y_min, x_max, y_max))
        im = im.resize((100, 180), Image.ANTIALIAS)
        
        # add to list of images
        images.append(im)
        
    else:
        pass
#%%
# save images and labels as numpy arrays
np.save('images.npy', np.stack(images, axis=0))

hots = np.zeros((len(labels), max(labels) + 1))
hots[np.arange(len(labels)), labels] = 1
np.save('labels.npy', hots)
#%% Plot example photos

fig = plt.figure()
N = 10
n_rows = 2
n_cols = 5

for i in range(1, N + 1):
    idx = np.random.randint(len(images))
    im = images[idx]
    fig.add_subplot(n_rows, n_cols, i)
    plt.axis('off')
    plt.imshow(im)
    plt.title('label: {}'.format(labels[idx]))
    plt.savefig('sample_hands.png')