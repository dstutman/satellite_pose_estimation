# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 19:57:55 2017

@author: Weiyan Cai
"""

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import EPnP
import json

## OUTPUTS
# error = error of EPnP
# Rt = a combination of the rotations matrix [3x3] and the transposition [1x3] in form [4x3] [Rot|Trans]
# Cc = not quite sure
# Xc = position of key points 


class EPnPTest(object):
    def __init__(self):

        
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
        
        self.load_test_data()
        self.epnp = EPnP.EPnP()


    def load_test_data(self):    
#         f = open(os.getcwd() + '/input/annotations/IMG_2936.json')
        
#         data = json.load(f)
#         data = sio.loadmat(os.getcwd() + '/input/input_data_noise.mat')

#         self.A = data['A']
#         self.Rt = data['Rt']

#         points = data['point']
#         # points = data['keypoints']
#         self.Xcam, self.Ximg_true, self.Ximg_pix_true, self.Ximg, self.Ximg_pix, self.Xworld= [], [], [], [], [], []
#         for p in points[0]:
# #            self.Xcam.append(p[0])
# #            self.Ximg_true.append(p[1])
# #            self.Ximg_pix_true.append(p[2])
# #            self.Ximg.append(p[3])
#             self.Ximg_pix.append(p[4])
#             self.Xworld.append(p[5])
#         print(self.Ximg_pix)    
#         print(self.Xworld)

            
            
        f = open(os.getcwd() + '/input/annotations/IMG_6040.json')
                
        data = json.load(f)
        data1 = sio.loadmat(os.getcwd() + '/input/input_data_noise.mat')

        # self.A = data1['A']
        self.Rt = data1['Rt']
        


        points = data['keypoints']
        self.Xcam, self.Ximg_true, self.Ximg_pix_true, self.Ximg, self.Ximg_pix, self.Ximg_pix_X, self.Xworld= [], [], [], [], [], [], []
        for p in points[0]:
            self.Ximg_pix.append([[p[0]], [p[1]]])

            
        for p in points[1]:
            self.Ximg_pix.append([[p[0]], [p[1]]])
            

        self.Xworld = [[[0], [0], [0]], [[0], [0], [-0.10]], [[0], [0.10], [-0.10]], [[0], [0.10], [0]],
                  [[0.30], [0], [0]], [[0.30], [0], [-0.10]], [[0.30], [0.10], [-0.10]], [[0.30], [0.10], [0]]]


                

        N = [i for i in range(len(self.Ximg_pix)) if self.Ximg_pix[i] == [[0], [0]]]
        
        print(N)

        count = 0
        for i in range(len(N)):
            self.Ximg_pix.pop(N[i]-count)
            self.Xworld.pop(N[i]-count)
            count += 1

        self.n = len(self.Ximg_pix)
        

        
        print(np.array(self.Ximg_pix)/self.m)
        

        
    def draw_input_noisy_data(self):
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        plt.plot(0, 0, 'ok')
        # for p in self.Ximg_pix_true:
        #     plt.plot(p[0], p[1], '.r')
        for p in self.Ximg_pix:
            plt.plot(p[0], p[1], 'xg')
        axes.set_title('Noise in Image Plane', fontsize=18)
        plt.grid()
        
        # fig.savefig(os.getcwd() + '/output/Noise_in_Image_Plane.png', dpi=100)
        plt.show()
        
    def apply_EPnP(self):
        error, Rt, Cc, Xc, flag = self.epnp.efficient_pnp(np.array(self.Xworld), np.array(self.Ximg_pix)/self.m, self.A)
        if flag == False:
            return flag
        self.plot_3d_reconstruction("EPnP (Old)", Xc)
#        print("Error of EPnP: ", error)
        self.R, self.T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP: \n", self.R)
        print("Transposition of EPnP: \n", self.T)
        print("Error of EPnP: \n", error)
        
    def apply_EPnP_Gauss(self):
        error, Rt, Cc, Xc, flag = self.epnp.efficient_pnp_gauss(np.array(self.Xworld), np.array(self.Ximg_pix)/self.m, self.A)
        if flag == False:
            return flag
        self.plot_3d_reconstruction("EPnP (Gauss Newton)", Xc)
#        print("Error of EPnP (Gauss Newton Optimization): ", error)
        R, T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP (Gauss Newton Optimization): \n", R)
        print("Transposition of EPnP (Gauss Newton Optimization): \n", T)
        print("Error of EPnP: \n", error)
        
    def plot_3d_reconstruction(self, method, Xc):
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        plt.plot(0, 0, 'ok')
        for p in self.Xcam:
            plt.plot(p[0], p[1], '.r')
        for p in Xc:
            plt.plot(p[0], p[1], 'xg')
        axes.set_title(method + ' - Reprojection Error', fontsize=18)
        plt.grid()
        
        fig.savefig(os.getcwd() + "/output/" + method + '_Reprojection_Error.png', dpi=100)
        plt.show()
        
    def reprojection_error(self):
        self.Xworld = np.concatenate(self.Xworld).ravel().reshape(-1,3)
        
        n=len(self.Xworld)
        
        self.A = np.delete(self.A, -1, axis=1)
    
        P = self.A @ np.hstack([self.R,self.T])
        
        
    
        
        Xw_h = np.hstack([self.Xworld, np.ones((n,1))])
        
        
        Urep_ = (P @ Xw_h.transpose()).transpose()
        
        
        Urep = np.zeros((n,2))
        
        
        
    

if __name__ == "__main__":
    ET = EPnPTest()
    # ET.load_test_data()
    ET.draw_input_noisy_data()
    flag = ET.apply_EPnP() 
    ET.apply_EPnP_Gauss()
    if flag == True:
        ET.reprojection_error()
    else:
        print("COMPLEX NUMBERS")

