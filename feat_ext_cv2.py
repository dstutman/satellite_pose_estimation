# -*- coding: utf-8 -*-
"""
CV2 method
"""

import cv2
# from google.colab.patches import cv2_imshow

img = cv2.imread('delphi.jpg', flags=1)
cv2.imshow('image',img)

orb = cv2.ORB_create(10)

keypoint, des = orb.detectAndCompute(img, None)
img_final = cv2.drawKeypoints(img, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('image',img_final)