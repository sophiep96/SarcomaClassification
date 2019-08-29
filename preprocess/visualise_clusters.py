#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:34:45 2019
@author: sophie
"""

#This code visualises the clusters which have been created through K-means clustering.

from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np
# get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import collections
from sklearn.cluster import KMeans
from shutil import copy
from tqdm import tqdm
#from skimage import data
from skimage.color import rgb2hed

C0_PATH = "/home/sophie/Documents/test_view_slide/labelled_tiles/110002962_A.39/110002962_A.39_0"
C1_PATH = "/home/sophie/Documents/test_view_slide/labelled_tiles/110002962_A.39/110002962_A.39_1"
C2_PATH = "/home/sophie/Documents/test_view_slide/labelled_tiles/110002962_A.39/110002962_A.39_2" 


def calculateMeanHematox(ihc_hed): 
    eu, hema, dab = ihc_hed[:, :, 1], ihc_hed[:, :, 0], ihc_hed[:, :, 2]
    size_i = len(hema)
    num_pixels = 512*512
    mean_hema = mean_eu = mean_dab = 0
    for i in range(size_i):
        size_j = len(hema[i])
        for j in range(size_j):
            mean_hema += hema[i][j]
            mean_eu += eu[i][j]
            mean_dab += dab[i][j]

    return mean_hema/num_pixels, mean_eu/num_pixels, mean_dab/num_pixels

rgb_means_0 = []
file_to_rgb_0 = collections.defaultdict(tuple)
rgb_to_file_0 = collections.defaultdict(list)
path_to_tiles_0 = C0_PATH 
for f in tqdm(listdir(path_to_tiles_0)):
    image_path = join(path_to_tiles_0, f)
    image = Image.open(image_path)
    pixels = np.array(image)
    ihc_hed = rgb2hed(pixels)
    rgb_mean = calculateMeanHematox(ihc_hed)
    file_to_rgb_0[f] = rgb_mean
    rgb_means_0.append(rgb_mean)
    rgb_to_file_0[rgb_mean].append(f)

rgb_means_1 = []
file_to_rgb_1 = collections.defaultdict(tuple)
rgb_to_file_1 = collections.defaultdict(list)
path_to_tiles_1 = C1_PATH 
for f in tqdm(listdir(path_to_tiles_1)):
    image_path = join(path_to_tiles_1, f)
    image = Image.open(image_path)
    pixels = np.array(image)
    ihc_hed = rgb2hed(pixels)
    rgb_mean = calculateMeanHematox(ihc_hed)
    file_to_rgb_1[f] = rgb_mean
    rgb_means_1.append(rgb_mean)
    rgb_to_file_1[rgb_mean].append(f)
    
rgb_means_2 = []
file_to_rgb_2 = collections.defaultdict(tuple)
rgb_to_file_2 = collections.defaultdict(list)
path_to_tiles_2 = C2_PATH 
for f in tqdm(listdir(path_to_tiles_2)):
    image_path = join(path_to_tiles_2, f)
    image = Image.open(image_path)
    pixels = np.array(image)
    ihc_hed = rgb2hed(pixels)
    rgb_mean = calculateMeanHematox(ihc_hed)
    file_to_rgb_2[f] = rgb_mean
    rgb_means_2.append(rgb_mean)
    rgb_to_file_2[rgb_mean].append(f)
        
channel1_0=[item[0] for item in rgb_means_0]   
channel2_0=[item[1] for item in rgb_means_0]
channel1_1=[item[0] for item in rgb_means_1]   
channel2_1=[item[1] for item in rgb_means_1]
channel1_2=[item[0] for item in rgb_means_2]   
channel2_2=[item[1] for item in rgb_means_2]

plt.scatter(channel1_0,channel2_0,color='red',marker='.',label='Cluster 1')
plt.scatter(channel1_1,channel2_1,color='blue',marker='.',label='Cluster 2')
plt.scatter(channel1_2,channel2_2,color='green',marker='.',label='Cluster 3')
plt.xlabel("Haematoxylin")
plt.ylabel("Eosin")
plt.title("Haematoxylin and Eosin values for tiles after K-means clustering")
plt.legend(loc='upper right')
plt.savefig('110002962_A.39_scatter.png')
plt.show()
