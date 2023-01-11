# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 19:57:55 2017

@author: Weiyan Cai
Modified: Adam Sundberg
"""

import os
import scipy.io as sio
import numpy as np
import quaternion
import math
import matplotlib.pyplot as plt
import EPnP
import json
import operator as op




## OUTPUTS
# error = error of EPnP
# Rt = a combination of the rotations matrix [3x3] and the transposition [1x3] in form [4x3] [Rot|Trans]
# Cc = not quite sure
# Xc = position of key points 


class EPnPTest(object):
    def __init__(self):

        ##### CAMERA DATA INPTU #####
        # Focal Length in m
        f = 0.026  
        # Camera data in pixels
        # Focal length
        fx = 18571.42857
        fy = 18571.428571
        # Sensor dimentions  
        width = 4032
        height = 3024
        # Sensor Centroid
        u0 = width/2
        v0 = height/2
        # Camera Matrix
        self.m = fx/f
        self.A = np.array([[fx/self.m, 0, u0/self.m, 0], [0, fy/self.m, v0/self.m, 0], [0, 0, 1, 0]])
        
        self.epnp = EPnP.EPnP()


    def load_test_data(self):    
         
        # Load Image Key Points        
        f = open(os.getcwd() + '/input/CNN_results/IMG_6034.json')
        data = json.load(f)
        points = data['keypoints']
        self.Ximg_pix, self.Xworld= [], []
        for p in points[0]:
            self.Ximg_pix.append([[p[0]], [p[1]]])  
        for p in points[1]:
            self.Ximg_pix.append([[p[0]], [p[1]]])
            
        ##### INPUT VIRTUAL 3D MODEL KEYPOINTS #####
        self.Xworld = [[[0], [0], [0]], [[0], [0], [-0.10]], [[0], [0.10], [-0.10]], [[0], [0.10], [0]],
                  [[0.30], [0], [0]], [[0.30], [0], [-0.10]], [[0.30], [0.10], [-0.10]], [[0.30], [0.10], [0]]]

        # Remove key points which are not visisble from both image and model data
        N = [i for i in range(len(self.Ximg_pix)) if self.Ximg_pix[i] == [[0], [0]]]
        # Flag if number of key points is too small
        if len(N) > 4:
            return 0
        else: 
            count = 0
            for j in range(len(N)):
                self.Ximg_pix.pop(N[j]-count)
                self.Xworld.pop(N[j]-count)
                count += 1

            # Get length of key points
            self.n = len(self.Ximg_pix)
            return 1
        
        
    def load_base_data(self):    
                      
        # Load Image Key Points         
        f = open(os.getcwd() + '/input/annotations/IMG_6034.json')
        data = json.load(f)
        points = data['keypoints']
        self.Ximg_pix, self.Xworld= [], []
        for p in points[0]:
            self.Ximg_pix.append([[p[0]], [p[1]]])  
        for p in points[1]:
            self.Ximg_pix.append([[p[0]], [p[1]]])
            
        ##### INPUT VIRTUAL 3D MODEL KEYPOINTS #####
        self.Xworld = [[[0], [0], [0]], [[0], [0], [-0.10]], [[0], [0.10], [-0.10]], [[0], [0.10], [0]],
                  [[0.30], [0], [0]], [[0.30], [0], [-0.10]], [[0.30], [0.10], [-0.10]], [[0.30], [0.10], [0]]]

        # Remove key points which are not visisble from both image and model data
        N = [i for i in range(len(self.Ximg_pix)) if self.Ximg_pix[i] == [[0], [0]]]
        # Flag if number of key points is too small
        if len(N) > 4:
            return 0
        else: 
            count = 0
            for j in range(len(N)):
                self.Ximg_pix.pop(N[j]-count)
                self.Xworld.pop(N[j]-count)
                count += 1

            # Get length of key points
            self.n = len(self.Ximg_pix)
            return 1

        
    def draw_input_noisy_data(self):
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        plt.plot(0, 0, 'ok')
        for p in self.Ximg_pix:
            plt.plot(p[0], p[1], 'xg')
        axes.set_title('Input Data', fontsize=18)
        plt.grid()
        plt.show()
        
    def apply_EPnP(self):
        error, Rt, Cc, Xc, flag = self.epnp.efficient_pnp(np.array(self.Xworld), np.array(self.Ximg_pix)/self.m, self.A)
        # Flag for Complex Numbers
        if flag == False:
            return flag
        self.plot_3d_reconstruction("EPnP (Old)", Xc)
        R, T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP: \n", R)
        print("Transposition of EPnP: \n", T)
        print("Error of EPnP: \n", error)
        return R, T, flag
        
    def apply_EPnP_Gauss(self):
        error, Rt, Cc, Xc, flag = self.epnp.efficient_pnp_gauss(np.array(self.Xworld), np.array(self.Ximg_pix)/self.m, self.A)
        # Flag for Complex Numbers
        if flag == False:
            return flag
        self.plot_3d_reconstruction("EPnP (Gauss Newton)", Xc)
        R, T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP (Gauss Newton Optimization): \n", R)
        print("Transposition of EPnP (Gauss Newton Optimization): \n", T)
        print("Error of EPnP (Gauss Newton Optimization): \n", error)
        return R, T, flag
        
    def plot_3d_reconstruction(self, method, Xc):
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        plt.plot(0, 0, 'ok')
        for p in Xc:
            plt.plot(p[0], p[1], 'xg')
        axes.set_title(method + ' - Reprojection - Not in image plane', fontsize=18)
        plt.grid()
        plt.show()
        
    def reprojection_error(self, R1, T1, R2, T2):
        # Get translation Error
        Te = list( map(op.sub, T1, T2) )
        Te_v = np.concatenate( Te, axis=0 )
        Te_abs = np.linalg.norm(Te_v)
        Te_rel = (Te_abs/np.linalg.norm(R2))*100 
        # Get rotation Error
        q1 = quaternion.as_float_array(quaternion.from_rotation_matrix(R1))
        q2 = quaternion.as_float_array(quaternion.from_rotation_matrix(R2))
        qe_rel = (1-abs(np.dot(q1,q2)))*100 
        return Te_rel, qe_rel
    

if __name__ == "__main__":
    ET = EPnPTest()
    # Run Pose Estimation on CNN data
    ET.load_test_data()
    ET.draw_input_noisy_data()
    R1, T1, flag1 = ET.apply_EPnP()  
    # Run Pose Estimation on manually made data
    ET.load_base_data()
    R2, T2, flag2 = ET.apply_EPnP()
   
    if (flag1 and flag2 == True):
        Te, Re = ET.reprojection_error(R1, T1, R2, T2)
        print("Translation Error of CNN is:","%.2f" % Te, "%")
        print("Rotation error of CNN is:", "%.2f" % Re, "%")
    else:
        print("COMPLEX NUMBERS")

