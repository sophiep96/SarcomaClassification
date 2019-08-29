#Merges the original image with the gradient map for that image.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:03:29 2019

@author: Sophie Peacock
"""

import os
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map

image1 = Image.open("SFT.jpeg").convert('RGB')
image2 = Image.open("SFT_gradients.jpg").convert('RGB') #or L for grayscale
image3 = Image.new('RGB',(512,512),(255,255,255))
mask = Image.open("SFT_gradients.jpg").convert('L')

final_2 = Image.composite(image1,image3,mask)
final_2.save('final2.png')
