# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jan. 28th 2021
'''

import glob
import os

import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class CustomDataset(Dataset):
    '''
    inputs: log, 
    output: Normlized or not
    '''
    def __init__(self, dataset_file, input_labels, output_labels):
        super(Dataset, self).__init__()
        self.df = pd.read_csv(dataset_file)
        print(f'[DATASET] input_labels: {input_labels}') 
        print(f'[DATASET] output_labels: {output_labels}')
        print(f'[DATASET] origin df: \n{self.df.head()} \n', )
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.scaler = MinMaxScaler()
        # normlize outputs
        # if is_normlize_output:
        #     self.normlize_outputs()
        # # loglize inputs
        # if is_loglize_input:
        #     self.loglize(self.input_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df[self.input_labels].iloc[idx]
        label = self.df.iloc[idx][self.output_labels[0]]
        return (row.values.astype(np.float32), label.astype(np.float32))

    def __len__(self):
        return self.df.shape[0]

    def loglize(self, column_names):
        '''log10 operation for inputs'''
        for column_name in column_names:
            self.df[column_name] = np.log10(self.df[column_name])

    def inverse_log(self, column_names):
        '''Inverse to original from log10'''
        for column_name in column_names:
            self.df[column_name] = 10 ** self.df[column_name]

    def normlize_outputs(self, is_inverse=False):
        '''Normalizing or inverse normalization in columns'''
        x = self.df[self.output_labels].values  # df -> numpy
        if not is_inverse:
            # normalization
            x_new = self.scaler.fit_transform(x)
        else:
            # transform to original scale
            x_new = self.scaler.inverse_transform(x)
        self.df[self.output_labels] = pd.DataFrame(x_new)  # numpy -> df
        print(f'[DATASET] normlize output for : {self.output_labels}')
        print(f'[DATASET] normlize_outputs: {self.df.head()}')


class CustomDataset2(CustomDataset):
    '''log the output and inputs'''
    def __init__(self, dataset_file, input_labels, output_labels, is_loglize_input=True, is_loglize_output=True, is_normlize_output=False):
        super(CustomDataset2, self).__init__(dataset_file, input_labels, output_labels)
        if is_loglize_input:
            self.loglize(self.input_labels)
            print(f'[DATASET] after df (log input): \n{self.df.head()}')
        if is_loglize_output:
            self.loglize(self.output_labels)
            print(f'[DATASET] after df (log output): \n{self.df.head()}')
        if is_normlize_output:
            self.normlize_outputs()
            print(f'[DATASET] after df (norm output): \n{self.df.head()}')
        print(f'[DATASET] final head: \n{self.df[self.input_labels+self.output_labels].head()}')
        print(f'[DATASET] final describe: \n{self.df[self.input_labels+self.output_labels].describe()}')

def spliter(dataset, split_rato=0.2, shuffle_dataset=True, batch_size=8, random_seed=None):
    '''split dataset_all as training set and test set'''
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split_num = int(np.floor(split_rato * dataset_size))
    if shuffle_dataset:
        if random_seed:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        else:
            np.random.shuffle(indices)
        print(f'[DATASET] indices: {indices[:10]}')
    train_indices, test_indices = indices[split_num:], indices[:split_num]
    print(f'[DATASET] train_indices: {train_indices[:5]}, number: {len(train_indices)}')
    print(f'[DATASET] test_indices: {test_indices[:5]}, number: {len(test_indices)}')

    # Creating PT data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader


def test():
    ########################dataset_re########################
    dataset_file = '../dataset/dataset_re.csv'
    input_labels = ['u', 'v', 'd', 'k']
    output_labels = ['log(100*lambda)']

    dataset_all = CustomDataset2(
        dataset_file,
        input_labels,
        output_labels,
        # is_normlize_output=False,
        is_normlize_output=True,
        is_loglize_output=False,
        is_loglize_input=True,
    )
    print(dataset_all[0])
    
    train_loader, test_loader = spliter(dataset_all, shuffle_dataset=True, random_seed=1)

    for step, (inputs, targets) in enumerate(train_loader):
        print(inputs.numpy().shape, targets.numpy().shape)
        break


if __name__ == '__main__':
    test()
