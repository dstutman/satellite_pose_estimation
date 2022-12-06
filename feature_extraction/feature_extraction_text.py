# -*- coding: utf-8 -*-
"""
Feature extraction test
Patrick Follis
4/12/22
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import filters
import cv2


# get image
image = imread('delphi.jpg', as_gray=True)
# plt.imshow(image);



# Sobel Kernel
ed_sobel = filters.sobel(image)


plt.imshow(ed_sobel, cmap='gray');

# plt.scatter(x=300, y=500, c='r', s=20)
# plt.show()




# threshold = 0.1
# rows, cols = np.where(image > threshold)

