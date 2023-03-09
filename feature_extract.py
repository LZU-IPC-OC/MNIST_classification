# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:34:57 2022

@author: 30685
"""

import numpy as np
from torchvision import datasets,transforms
import csv
import pandas as pd


#MNIST Dataset

train_dataset = datasets.MNIST(root='./',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

test_dataset = datasets.MNIST(root='./',
                              train=False,
                              transform=transforms.ToTensor())

#均值池化
def avg_pooling(data, m, n):
    a,b = data.shape
    img_new = []
    for i in range(0,a,m):
        line = []#记录每一行
        for j in range(0,b,n):
            x = data[i:i+m,j:j+n]#选取池化区域
            line.append(np.sum(x)/(n*m))
        img_new.append(line)
    return np.array(img_new)

#最大池化
def max_pooling(data, m, n):
    a,b = data.shape
    img_new = []
    for i in range(0,a,m):
        line = []
        for j in range(0,b,n):
            x = data[i:i+m,j:j+n]
            line.append(np.max(x))
        img_new.append(line)
    return np.array(img_new)

def bit_change(data):
    size = data.shape 
    img_new = []
    for i in range(size[0]):
        if data[i] <= 0.125:
            img_new.append(0.125)
        elif data[i] > 0.125 and data[i] <= 0.25:
            img_new.append(0.25)
        elif data[i] > 0.25 and data[i] <= 0.375:
            img_new.append(0.375)
        elif data[i] > 0.375 and data[i] <= 0.5:
            img_new.append(0.5)
        elif data[i] > 0.5 and data[i] <= 0.625:
            img_new.append(0.625)
        elif data[i] > 0.625 and data[i] <= 0.75:
            img_new.append(0.75)
        elif data[i] > 0.75 and data[i] <= 0.875:
            img_new.append(0.875)
        elif data[i] > 0.875 and data[i] <= 1:
            img_new.append(1)    
    return np.array(img_new)
    
csvfile = open('C:/Users/30685/Desktop/documents/MATLAB/MNIST/result/MNIST_train_0.csv','w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(np.concatenate((['Label'],[i for i in range(0,14*14)])))
for i in range(50000):
    label = np.array(train_dataset[i][1])
    data0 = np.array(train_dataset[i][0].squeeze(0))  
    data0 = avg_pooling(data0,2,2)
    data2 = np.reshape(data0,(1,-1)).squeeze(0)
    data2 = bit_change(data2)
    csvwriter.writerow(np.concatenate(([label],data2)))

csvfile.close()

csvfile = open('./MNIST_train_0_original.csv','w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(np.concatenate((['Label'],[i for i in range(0,14*14)])))
for i in range(50000):
    label = np.array(train_dataset[i][1])
    data0 = np.array(train_dataset[i][0].squeeze(0))  
    data2 = np.reshape(data0,(1,-1)).squeeze(0)
    csvwriter.writerow(np.concatenate(([label],data2)))

csvfile.close()

csvfile = open('./MNIST_test_0.csv','w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(np.concatenate((['Label'],[i for i in range(0,14*14)])))
for i in range(10000):
    label = np.array(test_dataset[i][1])
    data0 = np.array(test_dataset[i][0].squeeze(0))  
    data0 = avg_pooling(data0,2,2)
    data2 = np.reshape(data0,(1,-1)).squeeze(0)
    data2 = bit_change(data2)
    csvwriter.writerow(np.concatenate(([label],data2)))

csvfile.close()

csvfile = open('./MNIST_test_1.csv','w')
csvwriter = csv.writer(csvfile)
for i in range(10000):
    data0 = np.array(test_dataset[i][0].squeeze(0))  
    data0 = avg_pooling(data0,2,2)
    data2 = np.reshape(data0,(1,-1)).squeeze(0)
    data2 = bit_change(data2)
    csvwriter.writerow(data2)

csvfile.close()
print('end')
    
    
        
