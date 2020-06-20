#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:34:02 2020

@author: rinorcakaj
Task 4
"""

#imports
import numpy as np
import time
from matplotlib import pyplot as plt

#data
gmm_data_points = np.loadtxt("dataSets/gmm.txt")
N = len(gmm_data_points)
#initialization
gauss_1 = [np.array([0,0]),np.matrix([[1,0], [0,1]]),0.25] # mu , sigma, pi
gauss_2 = [np.array([1,0]),np.matrix([[1,0], [0,1]]),0.25]
gauss_3 = [np.array([2,0]),np.matrix([[1,0], [0,1]]),0.25]
gauss_4 = [np.array([3,0]),np.matrix([[1,0], [0,1]]),0.25]
gauss1 = [gauss_1, gauss_2, gauss_3, gauss_4]



def log_likehood_gaussian(data, gauss):
    result = 0
    for i in range(len(data)):
        result += np.log(gauss[0][2]*gaussian_density_function(data[i], gauss[0][0], gauss[0][1]) + gauss[1][2]*gaussian_density_function(data[i], gauss[1][0], gauss[1][1]) + gauss[2][2]*gaussian_density_function(data[i], gauss[2][0], gauss[2][1]) + gauss[3][2]*gaussian_density_function(data[i], gauss[3][0], gauss[3][1]))
    return result


# evaluate gaussian function
def gaussian_density_function(x, mu, sigma):
    x = np.array(x)
    c1 = 1 / np.sqrt((2*np.pi)**2 * np.linalg.det(sigma))
    c2 = np.exp(-0.5 * np.matmul((x - mu), np.matmul(np.linalg.inv(sigma),(x - mu)).T))
    y = c1 * c2
    return y.item()

#e step
def e_step(gauss):
    alpha = np.zeros((N, 4))
    for j in range(4):
        for i in range(N):
            alpha[i,j] = gauss[j][2] * gaussian_density_function(gmm_data_points[i], gauss[j][0], gauss[j][1])
    q=np.sum(alpha,axis=1)
    for i in range(len(q)):
        alpha[i,:]=np.divide(alpha[i,:],q[i])
    return alpha

#m step
def m_step(gauss, alpha):
    for j in range(4):
        # mu new
        
        # caclulate N_j
        N_j=np.sum(alpha[:,j])
        # calculate mu
        gauss[j][0] = np.array([0,0])
        for i in range(N):
            gauss[j][0] = gauss[j][0] + alpha[i,j]*gmm_data_points[i]
        gauss[j][0] = gauss[j][0] / N_j
        
        # sigma new
        gauss[j][1] = np.zeros([2,2])
        for i in range(N):
            gauss[j][1] = gauss[j][1] + alpha[i,j]*np.outer(gmm_data_points[i] - gauss[j][0],gmm_data_points[i] - gauss[j][0])
        gauss[j][1] = gauss[j][1] / N_j
        # pi new
        gauss[j][2] = N_j / N
    return gauss
 
def plot_all(gauss):
    delta = 0.25
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    xx,yy = np.meshgrid(x,y)
    height = np.zeros((len(xx), len(xx)))
    height2 = np.zeros((len(xx), len(xx)))
    height3 = np.zeros((len(xx), len(xx)))
    height4 = np.zeros((len(xx), len(xx)))
    for i in range(len(xx)):
        for j in range(len(xx)):
            height[i,j] = gaussian_density_function([xx[i,j], yy[i,j]], gauss[0][0], gauss[0][1])
            height2[i,j] = gaussian_density_function([xx[i,j], yy[i,j]], gauss[1][0], gauss[1][1])
            height3[i,j] = gaussian_density_function([xx[i,j], yy[i,j]], gauss[2][0], gauss[2][1])
            height4[i,j] = gaussian_density_function([xx[i,j], yy[i,j]], gauss[3][0], gauss[3][1])
    
    plt.scatter(gmm_data_points[:,0], gmm_data_points[:,1], color="black")
    plt.contour(xx,yy,height, colors ='red')
    plt.contour(xx,yy,height2, colors = 'blue')
    plt.contour(xx,yy,height3, colors = 'yellow')
    plt.contour(xx,yy,height4, colors = 'green')
    plt.show()
    return
            
if __name__ == "__main__":
 #   t=time.time()
    result=np.zeros([31])
    for i in range(31):
        a2=e_step(gauss1)
        gauss1 = m_step(gauss1, a2)
        result[i] = log_likehood_gaussian(gmm_data_points,gauss1)
        if i in [1,3,5,10,30]:
            plt.title("iteration="+str(i))
            plot_all(gauss1)
    plt.scatter(np.arange(31),result)
    plt.xlabel('iteration')
    plt.ylabel('log likelihood')
    plt.show()
#    elapsed = time.time()-t
   # print(elapsed)

