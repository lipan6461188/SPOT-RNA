#-*- coding:utf-8 -*- 

import os
import torch as t
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import numpy as np
np.random.seed(1080)

class InputConvUnit(nn.Module):
    def __init__(self, out_dim=48):
        super(InputConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=8, out_channels=out_dim, kernel_size=3, padding=1)
    def forward(self, x):
        y = self.conv(x)
        return y

class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, dim=48, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1+(dilation-1), dilation=dilation)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2+(dilation-1), dilation=dilation)
        self.in1 = nn.InstanceNorm2d(num_features=dim)
        self.in2 = nn.InstanceNorm2d(num_features=dim)
        self.dropout = nn.Dropout2d(p=0.25)
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
    def forward(self,x):
        y = self.elu1(x)
        y = self.in1(y)
        y = y.triu(diagonal=2)
        y = self.conv1(y)
        y = self.elu2(y)
        y = self.in2(y)
        y = y.triu(diagonal=2)
        y = self.dropout(y)
        y = self.conv2(y)
        return y + x

class BiLSTM(nn.Module):
    """
    Bi-directional LSTM
    """
    def __init__(self, in_dim=48, hidden_dim=400, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, 
            num_layers=num_layers, dropout=dropout, bidirectional=True)
    def forward(self, x):
        # seq_len, batch, input_size
        x = x.squeeze(0)
        x = x.permute([1,2,0])
        output,(h_n, c_n) = self.lstm(x)
        output = output.permute([2,0,1]).unsqueeze(0)
        return output

class BooleanMask(nn.Module):
    """
    Get upper triangle elements
    """
    def __init__(self, dim=48):
        super(BooleanMask, self).__init__()
        self.elu = nn.ELU()
        self.in1 = nn.InstanceNorm2d(num_features=dim)
    def forward(self, x):
        y = self.elu(x)
        y = self.in1(y)
        batch,channel,H,W = y.shape
        n = H
        mask = t.triu(t.ones(n,n),diagonal=2)
        return y[:,:,mask==1].squeeze().t()

class FCL(nn.Module):
    """
    Fully connected layer
    """
    def __init__(self, in_dim=48, hidden_layers=2, hidden_dim=512):
        super(FCL, self).__init__()
        
        self.hidden_layers = hidden_layers
        for i in range(1, self.hidden_layers+1):
            if i==1:
                setattr(self, 'fc1', nn.Linear(in_features=in_dim, out_features=hidden_dim) )
            else:
                setattr(self, 'fc'+str(i), nn.Linear(in_features=hidden_dim, out_features=hidden_dim) )
            setattr(self, 'dropout'+str(i), nn.Dropout(p=0.5) )
            setattr(self, 'bn'+str(i), nn.BatchNorm1d(num_features=hidden_dim) )
            setattr(self, 'elu'+str(i), nn.ELU() )
        
        if self.hidden_layers:
            self.fc_end = nn.Linear(in_features=hidden_dim, out_features=1)
        else:
            self.fc_end = nn.Linear(in_features=in_dim, out_features=1)
    def forward(self,x):
        y = x
        for i in range(1, self.hidden_layers+1):
            y = getattr(self, 'fc'+str(i))( y )
            y = getattr(self, 'elu'+str(i))( y )
            y = getattr(self, 'dropout'+str(i))( y )
            y = getattr(self, 'bn'+str(i))( y )
        y = self.fc_end(y)
        return y


