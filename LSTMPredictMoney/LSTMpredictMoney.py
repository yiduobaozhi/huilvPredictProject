__author__ = '947'
#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data as Data

class Config():
    def __init__(self):
        self.input_size = 1
        self.hidden_dim = 30
        self.num_layers = 1
        self.learning_rate = 0.01
        self.output_size = 1
        self.epoch = 800

class LSTMPredict(nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,output_size):
        super(LSTMPredict,self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(self.input_size,self.hidden_dim,self.num_layers)
        self.lstm2out = nn.Linear(self.hidden_dim,self.output_size)

    def forward(self,x):
        x,_ = self.lstm(x)
        seq,batch,hidden = x.shape
        x = x.view(seq*batch,hidden)
        out = self.lstm2out(x)
        out = F.relu(out)
        out = out.view(seq,batch,-1)
        return out



def handleTrainData():
    count = 0
    dataList = list()
    with open('E:/moneyPredictData.txt',encoding='UTF-8') as f:
        for data in f.readlines():
            count = count+1
            if(count == 1):
                continue
            time = data.split('	')[0]
            kai = float(data.split('	')[1])
            high = float(data.split('	')[2])
            low = float(data.split('	')[3])
            shou = float(data.split('	')[4])

            data = kai
            dataList.append(data)

            # data = Variable(torch.Tensor([data]))
            # print(data)
        dataLen = len(dataList)
        x = dataList[:dataLen-1]
        y = dataList[1:dataLen]

        return x,y

        # torch_dataset = Data.TensorDataset(x, y)
        # loader = Data.DataLoader(
        #     dataset=torch_dataset,      # torch TensorDataset format
        #     batch_size=100,           # mini batch size
        #     shuffle=True,               # 要不要打乱数据 (打乱比较好)
        #     num_workers=2,              # 多线程来读数据
        # )
        # return loader

def trianModel():
    config = Config()
    net = LSTMPredict(config.input_size,config.hidden_dim,config.num_layers,config.output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=config.learning_rate)
    x,y = handleTrainData()
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(-1,1,1)
    y = y.reshape(-1,1,1)

    # epochlist = list()
    # losslist  = list()
    for epoch in range(config.epoch):
        print(epoch)
        # epochlist.append(epoch)
        x = Variable(torch.Tensor(x))
        y = Variable(torch.Tensor(y))

        out = net(x)
        loss = criterion(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        net = net.eval()
        a = np.array(0.127)
        a= a.reshape(-1,1,1)
        var_data = Variable(torch.Tensor(a))
        pred_test = net(var_data) # 测试集的预测结果

        # 改变输出的格式

        if epoch % 10 == 0:
            print(pred_test.view(-1).data.numpy())
            print('loss:',loss.item())
            torch.save(net.state_dict(), "data\model")

        # losslist.append(loss.item())

    # plt.plot(epochlist, losslist, marker="*", linewidth=3, linestyle="--", color="orange")
    # plt.show()

def loadAndUseModel(model_path):
    config = Config()
    net = LSTMPredict(config.input_size,config.hidden_dim,config.num_layers,config.output_size)
    net.load_state_dict(torch.load(model_path))
    return net

def predict(num):
    model_path = 'data\model'
    net = loadAndUseModel(model_path)
    net = net.eval()
    a = np.array(num)
    a= a.reshape(-1,1,1)
    var_data = Variable(torch.Tensor(a))
    pred_test = net(var_data) # 测试集的预测结果
    print(pred_test.view(-1).data.numpy())
    return pred_test.view(-1).data.numpy()


if __name__ == '__main__':
    # trianModel()
    predict(0.127)