# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jan. 28th 2021
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionNet(nn.Module):
    def __init__(self, input_dim, dimless_num, base_vec_num, lr, hidden_num=10, reverse_input=False):
        super(DimensionNet, self).__init__()
        self.reverse_input = reverse_input
        self.fc0 = nn.Linear(input_dim, base_vec_num, bias=False)
        self.fc1 = nn.Linear(base_vec_num, dimless_num, bias=False)
        self.fc2 = nn.Linear(dimless_num, hidden_num)
        self.fc3 = nn.Linear(hidden_num, hidden_num)
        self.fc4 = nn.Linear(hidden_num, hidden_num)
        self.fc5 = nn.Linear(hidden_num, hidden_num)
        self.fc6 = nn.Linear(hidden_num, 1)

    @staticmethod
    def load_pretrained_model(new_net, pretrained_model, pretrained_path):
        '''load pretrained model'''
        pretrained_model.load_state_dict(
            torch.load(pretrained_path, map_location='cpu'))
        pretrained_dict = pretrained_model.state_dict()
        new_net.load_state_dict(pretrained_dict)
        return new_net

    def forward(self, inputs):
        '''forward'''
        # scaling network
        # print('forward inputs', inputs.shape)
        x = self.fc0(inputs)
        x = self.fc1(x)

        # reverse input into original scale
        if self.reverse_input:
            x = 10**x

        # deep feedfoward netwrk
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x


class DimensionNetTest(nn.Module):
    def __init__(self, input_dim, dimless_num, base_vec_num, lr, hidden_num=10, reverse_input=False):
        super(DimensionNetTest, self).__init__()
        self.reverse_input = reverse_input
        self.fc0 = nn.Linear(input_dim, base_vec_num, bias=False)
        self.fc1 = nn.Linear(base_vec_num, dimless_num, bias=False)
        self.fc2 = nn.Linear(dimless_num, hidden_num)
        self.fc3 = nn.Linear(hidden_num, hidden_num)
        self.fc4 = nn.Linear(hidden_num, hidden_num)
        self.fc5 = nn.Linear(hidden_num, hidden_num)
        self.fc6 = nn.Linear(hidden_num, 1)

    @staticmethod
    def load_pretrained_model(new_net, pretrained_model, pretrained_path):
        '''load pretrained model'''
        pretrained_model.load_state_dict(
            torch.load(pretrained_path, map_location='cpu'))
        pretrained_dict = pretrained_model.state_dict()
        new_net.load_state_dict(pretrained_dict)
        return new_net

    def forward(self, x):
        '''forward'''
        # reverse input into original scale
        if self.reverse_input:
            x = 10**x

        # deep feedfoward netwrk
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x
