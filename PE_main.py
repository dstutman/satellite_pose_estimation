# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 19:57:55 2017

@author: Weiyan Cai
Modified: Adam Sundberg
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
        self.load_test_data()
        self.epnp = EPnP.EPnP()

    def load_test_data(self):
        
        # Load Image Key Points
        f = open(os.getcwd() + '/input/annotations/IMG_2936.json')                
        data = json.load(f)       
        points = data['keypoints']
        
        self.Ximg_pix, self.Xworld= [], []
        
        for p in points[0]:
            self.Ximg_pix.append([[p[0]], [p[1]]])
        for p in points[1]:
            self.Ximg_pix.append([[p[0]], [p[1]]])        
        
        # Load Camera Data (NEED TO CHANGE TO OUR CAMERA PARAMS)
        data1 = sio.loadmat(os.getcwd() + '/input/input_data_noise.mat')
        self.A = data1['A']
        self.Rt = data1['Rt']
        
        # 3D Model Key Points
        self.Xworld = [[[0], [0], [0]], [[0], [0], [-10]], [[0], [10], [-10]], [[0], [10], [0]],
                  [[30], [0], [0]], [[30], [0], [-10]], [[30], [10], [-10]], [[30], [10], [0]]]


        # Remove key points which are not visisble from both image and model data
        N = [i for i in range(len(self.Ximg_pix)) if self.Ximg_pix[i] == (0,0)]

        for i in range(len(N)):
            self.Ximg_pix.pop(N[i-1])
            self.Xworld.pop(N[i-1])

        # Get length of keypoints
        self.n = len(self.Ximg_pix)
        
        
    def apply_EPnP(self):
        error, Rt, Cc, Xc = self.epnp.efficient_pnp(np.array(self.Xworld), np.array(self.Ximg_pix), self.A)
        self.plot_3d_reconstruction("EPnP (Old)", Xc)
        R, T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP: \n", R)
        print("Transposition of EPnP: \n", T)
        
    def apply_EPnP_Gauss(self):
        error, Rt, Cc, Xc = self.epnp.efficient_pnp_gauss(np.array(self.Xworld), np.array(self.Ximg_pix), self.A)
        self.plot_3d_reconstruction("EPnP (Gauss Newton)", Xc)
        R, T, dump = np.hsplit(Rt, np.array([3,6]))
        print("Rotation of EPnP (Gauss Newton Optimization): \n", R)
        print("Transposition of EPnP (Gauss Newton Optimization): \n", T)
        
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
    

if __name__ == "__main__":
    ET = EPnPTest()
    ET.load_test_data()
    ET.apply_EPnP()
    ET.apply_EPnP_Gauss()

