# Code created by Matineh Akhlaghinia to perform K-means clustering on tiles to sort the tiles into three clusters,
# normal cells, cancerous cells and tiles containing artefacts.

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
#from matplotlib.colors import LinearSegmentedColormap


OFFSET_initial = 40
OFFSET_final = 20
CLUSTERS = 3
DESTINATION_PATH = "/home/sophie/Documents/test_view_slide/test_labels2"
DATA_PATH = "/home/sophie/Documents/test_view_slide/test_tiles"


def cleanupCluster(path_to_cluster, file_name, offset,dir_exists=False):
    r = g = b = 0
    cnt = 0
    dest = join(DESTINATION_PATH, file_name, "deleted")
    if not dir_exists:
        os.makedirs(dest)
    r_mean, g_mean, b_mean = calculateMeanRGBInCluster(path_to_cluster)
    for f in tqdm(listdir(path_to_cluster)):
        image_path = join(path_to_cluster, f)
        r, g, b = calculateMeanRGB(image_path)
        if r < r_mean-offset or r > r_mean+offset or g < g_mean-offset or g > g_mean+offset or b < b_mean-offset or b > b_mean+offset:
            copy(image_path, dest)
            os.remove(image_path)
            


def calculateMeanRGBInCluster(path_to_cluster):
    r = g = b = 0
    cnt = 0
    files = listdir(path_to_cluster)
    for f in tqdm(files):
        if f in (".DS_Store"):
            continue
        image_path = join(path_to_cluster, f)
        rgb_mean = calculateMeanRGB(image_path)
        r += rgb_mean[0]
        g += rgb_mean[1]
        b += rgb_mean[2]
        cnt +=1

    return r/cnt, g/cnt, b/cnt


def applyKMeans(rgb_means, clusters=3):
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(rgb_means)
    image_and_cluster_tuple = zip(kmeans.labels_, rgb_means)
    clusters_dict = collections.defaultdict(list)
    for cluster, rgb in image_and_cluster_tuple:
        clusters_dict[cluster].append(rgb)
    return clusters_dict


def clusterAndStore(rgb_to_file, clusters_dict, src_path, dir_names, file_name):    
    # Create directories based on the clusters.
    dest = join(DESTINATION_PATH, file_name)
    for direc in dir_names:
        os.makedirs(join(dest, direc))
    
    for i in range(CLUSTERS):
        cluster = clusters_dict[i]
        for rgb in tqdm(cluster):
            f_name = rgb_to_file[rgb]
            for f in f_name:
                src = join(src_path, f)
                dst = join(dest, dir_names[i])
                copy(src, dst)
    



def calculateMeanRGB(path):
    image = Image.open(path)
    pixels = image.load()
    width, height = image.size
    num_pixels = width * height
    r_mean = g_mean = b_mean = 0
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            r_mean += r
            g_mean += g
            b_mean += b

    return r_mean/num_pixels, g_mean/num_pixels, b_mean/num_pixels


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



rgb_means = []
for file in listdir(DATA_PATH):
    if file in (".DS_Store"):
        continue
    file_to_rgb = collections.defaultdict(tuple)
    rgb_to_file = collections.defaultdict(list)
    # Path to the tiles directory for an image with 20.0 maginification level
    path_to_tiles = join(DATA_PATH, file, "{}_files".format(file), "20.0")
    # Create the directory for this slide image
    os.mkdir(join(DESTINATION_PATH, file))
    cleanupCluster(path_to_tiles, file, OFFSET_initial)
    for f in tqdm(listdir(path_to_tiles)):
        if f in (".DS_Store"):
            continue
        image_path = join(path_to_tiles, f)
        rgb_mean = calculateMeanRGB(image_path)
#        image = Image.open(image_path)
#        pixels = np.array(image)
#        ihc_hed = rgb2hed(pixels)
#        rgb_mean = calculateMeanHematox(ihc_hed)
        file_to_rgb[f] = rgb_mean
        rgb_means.append(rgb_mean)
        rgb_to_file[rgb_mean].append(f)

    # Cluster and store the results
    cluster_dirs_paths = ["".join([file,"_",str(i)]) for i in range(CLUSTERS)]
    clusters_dict = applyKMeans(rgb_means, clusters=CLUSTERS)
    clusterAndStore(rgb_to_file, clusters_dict, path_to_tiles, cluster_dirs_paths, file)



