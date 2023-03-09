# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:50:53 2022

@author: 30685
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
from torch.autograd import Variable

data_header = ['1','2','3','4','5','6',7,'8','9','10','11','12']
label_header = ['A']
train_data = pd.read_csv("./train_data.csv",names = data_header)
train_label = pd.read_csv("./Label_data.csv",names = label_header)

train_data_0 = torch.tensor(np.array(train_data.astype('float32')))
train_label_0 = torch.tensor(np.array(train_label.astype('float32')),dtype=torch.long)

train_dest = TensorDataset(train_data_0,train_label_0)

train_loader = DataLoader(train_dest,batch_size = 100, shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(in_features = 12,out_features = 2,bias = False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self,x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        return out
    
model = Net()
print(model)

loss_func = torch.nn.CrossEntropyLoss()

opt = torch.optim.Adam(model.parameters(),lr = 0.1)

def train(epoch):
    running_loss = 0.0
    for batch_index,(data,target) in enumerate(train_loader):
        data,target = Variable(data),Variable(target)
        #print(data,target)
        target = target.squeeze(1)
        data = data.squeeze(1)
        opt.zero_grad()
        output = model(data)
        loss = loss_func(output,target)
        loss.backward
        opt.step()
        
        running_loss += loss.item()
        if batch_index % 200 == 0:
            print('Train Epoch:{} \tLoss:{:.6f}'.format(epoch,running_loss))
            
def test():
    test_loss = 0
    correct = 0
    for data,target in train_loader:
        with torch.no_grad():
            data , target = Variable(data),Variable(target)
        target = target.squeeze(1)
        output = model(data)
        # sum up batch loss
        test_loss += loss_func(output, target).item()
        # get the index of the max log-probability
        pred = torch.max(output.data,1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(train_loader.dataset)

        print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
              test_loss, correct, len(train_loader.dataset),
              100. * correct / len(train_loader.dataset)))

for epoch in range(10):
    train(epoch)
    test()
        
# 将各层的训练出来的权重和偏置写入到外部的txt文件中
parm = {}
for name,parameters in model.named_parameters():  # 定义的net中的named_parameters有layer1，2和它们分别的weight和bias
    print(name,':',parameters.size())           # 各项训练参数的size大小
    parm[name] = parameters.detach().numpy()    # 将矩阵转化为numpy矩阵
    print(parm[name].shape)

for key in parm.keys(): 
    if parm[key].ndim == 4:                        # 按照不同训练数据的名字来命名生成相应的txt文件
        for i in range(parm[key].shape[0]):
            for j in range(parm[key].shape[1]):
                key1 = key + '_' + str(i + 1) + '_' + str(j + 1) 
                name = "./" + key1 + '.txt'
                np.savetxt(name,parm[key][i,j,:,:],fmt = '%f',delimiter = ',')
    if parm[key].ndim == 2:
        name = "./"+ key + '.txt'
        np.savetxt(name,parm[key],fmt = '%f',delimiter = ',')

torch.save(model,"model.pt")    