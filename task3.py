#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:32:56 2020

@author: rinorcakaj
"""
#imports
import math as m
import numpy as np
from matplotlib import pyplot as plt

#data
nonParamTrain = np.loadtxt("dataSets/nonParamTrain.txt")
nonParamTest = np.loadtxt("dataSets/nonParamTest.txt")


def histogram(data, bin_size):
    bins = np.arange(min(data), max(data) + bin_size, bin_size)
    hist = np.zeros((len(bins),2))
    hist[:,0] = bins
    for i in range(len(hist)):
        for j in range(len(data)):
            if hist[i,0] <= data[j] < hist[i+1,0]:
                hist[i,1] = hist[i,1] + 1
    hist[:,0] = hist[:,0] + 0.5 * bin_size
    
    plt.bar(hist[:,0], hist[:,1], color="red", width=bin_size)
    plt.title("bin size="+ str(bin_size))       
    plt.show()
    
def gaussian_kernel(data, x, sigma):
    result = 0
    for i in range(len(data)):
        result += m.exp(-(x-data[i])**2 / (2 * sigma**2))
    result = (result/(len(data) * m.sqrt(2 * m.pi * sigma**2)))
    return result

def log_likehood_gaussian(data, log_like_data, sigma):
    result = 0
    for i in range(len(log_like_data)):
        result += np.log(gaussian_kernel(data, log_like_data[i], sigma))
    return result

def log_likehood_neighbor(data, log_like_data, K):
    result = 0 
    for i in range(len(log_like_data)):
        result += m.log(k_nearest_neighbor(data, K, log_like_data[i]))
    return result

def k_nearest_neighbor(data,K,x):
    values = np.column_stack((data, data))
    values[:,0] = abs(values[:,0] - x)
    values = values[values[:,0].argsort()]
    result = values[K-1,0]
    v=result*2
    return (K / (v * len(data)))
    

if __name__ == "__main__":
    # settings 3a)
    #hist_data = nonParamTrain
    #bin_sizes = [0.02,0.5,2,0.4]
    
    # 3a) Histogramme
    #for hist_bin_size in bin_sizes:
     #   histogram(hist_data, hist_bin_size)
        
    
     # # settings
    data = nonParamTrain
    # # plot densities
    # x = np.linspace(-4,8,100)
    # y = []
    # for sigma in [0.03, 0.2, 0.8]:
    #     for i in range(len(x)):
    #         y.append(gaussian_kernel(data, x[i], sigma))
    #     plt.plot(x,y,label="sigma="+ str(sigma))
    #     plt.legend()
    #     y = []
    # plt.show()
    # # log-likehood
    
    # for sigma in [0.03, 0.2, 0.8]:
    #     log_likehood_solution = log_likehood_gaussian(data, data, sigma)
    #     print("sigma:", sigma, "log_likehood:", log_likehood_solution)
        
        
    # 3c) K - Nearest - Neighbor
    # x = np.linspace(-4,8,400)
    # y = []
    # for K in [2, 8, 35]:
    #     for i in range(len(x)):
    #         y.append(k_nearest_neighbor(data,K,x[i]))
    #     plt.plot(x,y,label="K="+ str(K))
    #     plt.legend()
    #     y = []
    # plt.show()
    
    # 3d) Log Likehood 
    
    for sigma in [0.03, 0.2, 0.8]:
        log_likehood_solution = log_likehood_gaussian(data, nonParamTest , sigma)
        print("sigma:", sigma, "log_likehood_gaussian:", log_likehood_solution)
        
    for K in [2, 8, 35]:
        log_likehood_solution = log_likehood_neighbor(data, nonParamTest, K)
        print("K:", K, "log_likehood_neighbor:", log_likehood_solution)
    for K in [2, 8, 35]:
        log_likehood_solution = log_likehood_neighbor(data, data, K)
        print("K:", K, "log_likehood_neighbor_training:", log_likehood_solution)
    
    
    