import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.utils.data import random_split,DataLoader,TensorDataset
from torch.autograd import Variable

finalfts0 = pd.read_csv("./MNIST_train_0.csv")
finalfts1 = pd.read_csv("./MNIST_test_0.csv")
finalfts2 = pd.read_csv("./MNIST_test_2.csv")

train_dest = TensorDataset(torch.tensor(np.array(finalfts0[finalfts0.columns[1:]].astype('float32'))),torch.LongTensor(finalfts0['Label']))
test_dest = TensorDataset(torch.tensor(np.array(finalfts1[finalfts1.columns[1:]].astype('float32'))),torch.LongTensor(finalfts1['Label']))
test_dest_0 = TensorDataset(torch.tensor(np.array(finalfts2[finalfts2.columns[1:]].astype('float32'))),torch.LongTensor(finalfts2['Label']))


train_loader = DataLoader(train_dest,batch_size = 1,shuffle = True)   # 读取数据集中的数据
test_loader = DataLoader(test_dest,batch_size = 1)
test_loader_0 = DataLoader(test_dest_0,batch_size = 1)

loss_func = torch.nn.CrossEntropyLoss()

class Net(nn.Module):
    # 初始化模块
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding = 0, bias = False)  # 第一卷积层
        self.mp1 = nn.AvgPool2d(3,3)        # 平均池化层
        self.relu = nn.ReLU()               # 线性整流函数
        self.fc1 = nn.Linear(16, 10 , bias= False)  # 全连接层输出
        self.logsoftmax = nn.LogSoftmax(dim = 1)   # 柔性最大值函数之后取log

    #向前传播模块
    def forward(self, x):
        x = x.view(14,14)
        print(x)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        out = self.mp1(self.relu(self.conv1(x)))        # 第一层卷积后使用最大池化线性整流激活函数
        #out = self.relu(self.mp2(self.conv2(out)))      # 第二层卷积，之后同上
        #out = out.view(-1,16)                     # 展开成一维向量
        out = self.fc1(out)                             # 全连接层
        return self.logsoftmax(out)

# 模型文件位置
model_file = 'model.pt'   # 或者.pth格式的模型文件
# 创建模型对象
# 加载模型参数，若为cpu加载，则后面添加参数： map_location='cpu'
model = torch.load(model_file)      # cpu加载方式  ckpt = torch.load(model_file, map_location='cpu')
model.eval()

def output4():
    model.logsoftmax=nn.Sequential()
    return model
# model.conv1=nn.Sequential()
def output3():
    model = output4()
    model.fc1=nn.Sequential()
    return model
def output2():
    model = output3()
    model.relu=nn.Sequential()
    return model
def output1():
    model = output2()
    model.mp1=nn.Sequential()
    return model


output3()


# 定义优化器,使用亚当优化算法
#opt = torch.optim.Adam(model.parameters(),lr=0.01)


def test():
    test_loss = 0
    correct = 0
    output_temp=[]
    for data,target in test_loader_0:
        with torch.no_grad():
            data , target = Variable(data),Variable(target)
        #data , target = Variable(data,volatile=True) ,Variable(target)
        output = model(data)
        output_temp.append(output)
    return output_temp



print(test())