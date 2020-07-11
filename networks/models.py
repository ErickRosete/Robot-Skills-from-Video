#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:02:01 2020

@author: suresh, eric, jessica
"""

import utils.constants as constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import copy

class VisionNetwork(nn.Module):
    # reference: https://arxiv.org/pdf/2005.07648.pdf
    def __init__(self):
        super(VisionNetwork, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4), # shape: [N, 3, 299, 299]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), # shape: [N, 32, 73, 73]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), # shape: [N, 64, 35, 35]
            nn.ReLU(),
            SpatialSoftmax(num_rows=33, num_cols=33), # shape: [N, 64, 33, 33]
            # nn.Flatten(),
            nn.Linear(in_features=128, out_features=512), # shape: [N, 128]
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=64), # shape: [N, 512]
            )
    def forward(self, x):
        x = self.conv_model(x)
        return x # shape: [N, 64]


class SpatialSoftmax(nn.Module):
    # reference: https://arxiv.org/pdf/1509.06113.pdf
    # https://github.com/naruya/spatial_softmax-pytorch
    # https://github.com/cbfinn/gps/blob/82fa6cc930c4392d55d2525f6b792089f1d2ccfe/python/gps/algorithm/policy_opt/tf_model_example.py#L168
    def __init__(self, num_rows, num_cols):
        super(SpatialSoftmax, self).__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols

        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        self.x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H
        self.y_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # batch, C, W*H
        x = F.softmax(x, dim=2) # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map) # batch, C
        fp_y = torch.matmul(x, self.y_map) # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x # batch, C*2


class PlanRecognitionNetwork(nn.Module):
    def __init__(self):
        super(PlanRecognitionNetwork, self).__init__()
        self.in_features = constants.VISUAL_FEATURES + constants.N_DOF_ROBOT

        self.rnn_model = nn.Sequential(
            # bidirectional RNN
            nn.RNN(input_size=self.in_features, hidden_size=2048, num_layers=2, nonlinearity='relu', bidirectional=True, batch_first=True)
            ) # shape: [N, seq_len, 64+8]
        self.mean_fc = nn.Linear(in_features=4096, out_features=constants.PLAN_FEATURES) # shape: [N, seq_len, 4096]
        self.variance_fc = nn.Linear(in_features=4096, out_features=constants.PLAN_FEATURES) # shape: [N, seq_len, 4096]

    def forward(self, x):
        x, hn = self.rnn_model(x)
        x = x[:, -1] # we just need only last unit output
        mean = self.mean_fc(x)
        variance = F.softplus(self.variance_fc(x))
        return mean, variance # shape: [N, 256]


class PlanProposalNetwork(nn.Module):
    def __init__(self):
        super(PlanProposalNetwork, self).__init__()
        self.in_features = (constants.VISUAL_FEATURES + constants.N_DOF_ROBOT) + constants.VISUAL_FEATURES

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=2048), # shape: [N, 136]
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            )
        self.mean_fc = nn.Linear(in_features=2048, out_features=constants.PLAN_FEATURES) # shape: [N, 2048]
        self.variance_fc = nn.Linear(in_features=2048, out_features=constants.PLAN_FEATURES) # shape: [N, 2048]

    def forward(self, x):
        x = self.fc_model(x)
        mean = self.mean_fc(x)
        variance = F.softplus(self.variance_fc(x))
        return mean, variance # shape: [N, 256]

class PolicyNetwork(nn.Module):
    def __init__(self, seq_len):
        super(PolicyNetwork, self).__init__()
        self.in_features = (constants.VISUAL_FEATURES + constants.N_DOF_ROBOT) + constants.VISUAL_FEATURES + constants.PLAN_FEATURES

        self.rnn_model = nn.Sequential(
            # unidirectional RNN
            nn.RNN(input_size=self.in_features, hidden_size=2048, num_layers=2, nonlinearity='relu', bidirectional=False, batch_first=True)
            ) # shape: [N, seq_len, 256 + 137]
        self.seq_len = seq_len
        self.linears = []
        self.mean_fc = []
        self.variance_fc = []
        for i in range(self.seq_len):
            self.linears.append(nn.Linear(in_features=2048, out_features=constants.N_DOF_ROBOT)) # shape: [N, seq_len, 2048]

        self.mean_fc = nn.ModuleList(copy.deepcopy(self.linears))
        self.variance_fc = nn.ModuleList(copy.deepcopy(self.linears))

    def forward(self, x):
        x, hn = self.rnn_model(x)
        mean = []
        variance = []
        for i in range(self.seq_len):
            mean.append(self.mean_fc[i](x[:, i]))
            variance.append(F.softplus(self.variance_fc[i](x[:, i])))
        return torch.stack(mean, 1), torch.stack(variance, 1) # shape: [N, seq_len, 9]


if __name__ == '__main__':
    # data = torch.zeros([2,64,33,33]).cuda()
    # data[0,0,0,1] = 10
    # data[0,1,1,1] = 10
    # data[0,2,1,2] = 10
    # network = SpatialSoftmax(33, 33)
    # output = network(data)
    # print(output.shape)

    batch_size = 32
    seq_len = 5 # K-length plan

    data = torch.randn(batch_size, 3, 299, 299).cuda()
    network = VisionNetwork().cuda()
    output = network(data)
    print('visual features: ',output.shape)
    print('------------------')

    #input of shape (batch, seq_len, input_size)
    data = torch.randn(batch_size, seq_len, 73).cuda()
    #h_0 of shape (num_layers * num_directions, batch, hidden_size)
    network = PlanRecognitionNetwork().cuda()
    mean, variance = network(data)
    print('plan recognition mean: ',mean.shape)
    print('plan recognition variance: ',variance.shape)
    print('------------------')

    data = torch.randn(batch_size, 137).cuda()
    network = PlanProposalNetwork().cuda()
    mean, variance = network(data)
    print('plan proposal mean: ',mean.shape)
    print('plan proposal variance: ',variance.shape)
    print('------------------')


    #input of shape (batch, seq_len, input_size)
    data = torch.randn(batch_size, seq_len, 393).cuda()  # batch_size = 1
    #h_0 of shape (num_layers * num_directions, batch, hidden_size)
    network = PolicyNetwork(seq_len).cuda()
    mean, variance = network(data)
    print('policy mean: ',mean.shape)
    print('policy variance: ',variance.shape)
