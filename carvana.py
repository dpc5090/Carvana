# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 14:53:00 2017

@author: Daniel
"""

import pandas as pd
import numpy as np
import os
os.chdir('C:\Users\Daniel\Documents\Kaggle\Carvana')
dataset = pd.read_csv('train_masks.csv')
x = dataset.rle_mask[0]
indarray = []
lenarray = []
pos = x.index(' ')
while pos != -1:
    pos = x.index(' ')
    indarray.append(x[:pos])
    
    x = x[pos+1:]
    try:
        pos = x.index(' ')
        lenarray.append(x[:pos])
        x = x[pos+1:]
    except:
        pos = -1
        lenarray.append(x)
        x = ''


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('00087a6bd4dc_01.jpg')
imgplot = plt.imshow(img)
plt.show()

mask = [0]*1280*1918

count = len(lenarray)
for x in range(0,count):
    for y in range(0,int(lenarray[x])):
        mask[int(indarray[x])+y] = 1
        
np_mask = np.array(mask)
np_mask.shape = (1280,1918)
imgplot = plt.imshow(np_mask)
plt.show()
        
img.shape = (1280*1918,3)
for x in range(0,1280*1918):
    if mask[x] == 0:
        img[x] = [255,255,255]
img.shape = (1280,1918,3)
imgplot = plt.imshow(img)
plt.show()

import glob
filelist = glob.glob('*.jpg')
#shapes of all images 1280,1918,3

img=mpimg.imread('00087a6bd4dc_01.jpg')
greyscaleimg = []
for x in img:
    for y in x:
        greyscaleimg.append((int(y[0])+int(y[1])+int(y[2]))/3)
        
img_grey = np.array(greyscaleimg)
img_grey.shape = (1280,1918)
plt.imshow(img_grey)
        
i = 1918
j = 1280
pix = 20
patches = []
for x in range(0,j/pix):
    for y in range(0,i/pix):
        patches.append([[x*pix,x*pix+pix-1],[y*pix,y*pix+pix-1]])
        


smallimg = []
for patch in patches:
    smallimg.append(np.mean(img_grey[patch[0][0]:patch[0][1]+1,patch[1][0]:patch[1][1]+1]))

smallimg = np.array(smallimg)
smallimg.shape = (j/pix,i/pix)
plt.imshow(smallimg)

def downsample(img,ratio):
    i = 1918
    j = 1280
    pix = ratio
    patches = []
    for x in range(0,j/pix):
        for y in range(0,i/pix):
            patches.append([[x*pix,x*pix+pix-1],[y*pix,y*pix+pix-1]])
            
    smallimg = []
    for patch in patches:
        smallimg.append(np.mean(img[patch[0][0]:patch[0][1]+1,patch[1][0]:patch[1][1]+1]))
    
    smallimg = np.array(smallimg)
    smallimg.shape = (j/pix,i/pix)
    plt.imshow(smallimg)
    

