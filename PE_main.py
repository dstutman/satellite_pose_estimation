# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 19:57:55 2017

@author: Weiyan Cai
Modified: Adam Sundberg

Current loop frequency (reg): 95.5 fps
Current loop frequency (Gauss): 78.473 fps
"""

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import EPnP
import json
import glob
import time

## OUTPUTS
# error = error of EPnP
# Rt = a combination of the rotations matrix [3x3] and the transposition [1x3] in form [4x3] [Rot|Trans]
# Cc = not quite sure
# Xc = position of key points 


class EPnPTest(object):
    def __init__(self):
        self.files = glob.glob(os.getcwd() + '/input/annotations/*.json')       
 
        
        # Focal Length in m
        f = 0.026  
        # Camera data in pixels
        # Focal length
        fx = 18571.42857
        fy = 18571.428571
        # Sensor Centroid
        u0 = 2016
        v0 = 1512
        # Sensor dimentions  
        width = 4032
        height = 3024
        
        self.m = fx/f
        self.A = np.array([[fx/self.m, 0, u0/self.m, 0], [0, fy/self.m, v0/self.m, 0], [0, 0, 1, 0]])
        
        self.epnp = EPnP.EPnP()



    def load_test_data(self, i):
        
        # Load Image Key Points
        f = open(self.files[i])               
        data = json.load(f)       
        points = data['keypoints']
        
        self.Ximg_pix, self.Xworld= [], []
        
        for p in points[0]:
            self.Ximg_pix.append([[p[0]], [p[1]]])
        for p in points[1]:
            self.Ximg_pix.append([[p[0]], [p[1]]])        
        

        data1 = sio.loadmat(os.getcwd() + '/input/input_data_noise.mat')
        self.Rt = data1['Rt']
        
        # 3D Model Key Points
        self.Xworld = [[[0], [0], [0]], [[0], [0], [-0.10]], [[0], [0.10], [-0.10]], [[0], [0.10], [0]],
                  [[0.30], [0], [0]], [[0.30], [0], [-0.10]], [[0.30], [0.10], [-0.10]], [[0.30], [0.10], [0]]]

        
        # Remove key points which are not visisble from both image and model data
        N = [i for i in range(len(self.Ximg_pix)) if self.Ximg_pix[i] == [[0], [0]]]
        
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
            
    

    def apply_EPnP(self):
        error, Rt, Cc, Xc, flag = self.epnp.efficient_pnp(np.array(self.Xworld), np.array(self.Ximg_pix)/self.m, self.A)
        if flag == False:
            return flag
#        self.plot_3d_reconstruction("EPnP (Old)", Xc)
#        print("Error of EPnP: ", error)
        self.R, self.T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP: \n", self.R)
        print("Transposition of EPnP: \n", self.T)
        print("Error of EPnP: \n", error)
        
    def apply_EPnP_Gauss(self):
        error, Rt, Cc, Xc, flag = self.epnp.efficient_pnp_gauss(np.array(self.Xworld), np.array(self.Ximg_pix)/self.m, self.A)
        if flag == False:
            return flag
#        self.plot_3d_reconstruction("EPnP (Gauss Newton)", Xc)
#        print("Error of EPnP (Gauss Newton Optimization): ", error)
        R, T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP (Gauss Newton Optimization): \n", R)
        print("Transposition of EPnP (Gauss Newton Optimization): \n", T)
        print("Error of EPnP (Gauss Newton Optimization): \n", error)
        
    # def plot_3d_reconstruction(self, method, Xc):
    #     fig = plt.figure()
    #     fig.set_size_inches(18.5, 13)
    #     axes = fig.add_subplot(1, 1, 1)
    #     plt.plot(0, 0, 'ok')
    #     # for p in self.Xcam:
    #     #     plt.plot(p[0], p[1], '.r')
    #     for p in Xc:
    #         plt.plot(p[0], p[1], 'xg')
    #     axes.set_title(method + ' - Reprojection Error', fontsize=18)
    #     plt.grid()
        
    #     fig.savefig(os.getcwd() + "/output/" + method + '_Reprojection_Error.png', dpi=100)
        # plt.show()
    

if __name__ == "__main__":
    ET = EPnPTest()
    i = 0
    # get the start time
    st = time.time()
    counter = 0

    while i < 100:
        dat_flag = ET.load_test_data(i)
        if dat_flag == 1:
            cpx_flag = ET.apply_EPnP()
            ET.apply_EPnP_Gauss()
            if cpx_flag == False:
                print("Complex numbers, data skipped")
                counter += 1
        else:
            print("Not enough key points, data skipped")
            counter +=1
        i += 1
    # get the end time
    et = time.time()
    
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("Number of images skipped:", counter)
