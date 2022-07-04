# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jun, 2022
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Dataset(object):
    '''
    Load and parse dataset
    '''

    def __init__(self, dataset_path, input_list, output_list):
        self.dataset_path = dataset_path
        self.input_list, self.output_list = input_list, output_list

        self.df = self._load_dataset()
        self.df_train, self.df_test = self._split_dataset()

    def _load_dataset(self):
        '''load dataset'''
        df = pd.read_csv(self.dataset_path)
        return df
    
    def _split_dataset(self, test_size=0.2, random_state=1):
        '''randomly split dataset'''
        df_train, df_test = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return df_train, df_test

    def parser(self, is_shuffle=True, random_state=0):
        '''load dataset using numpy'''
        X_train = self.df_train[self.input_list].to_numpy()
        y_train = self.df_train[self.output_list].to_numpy().reshape(-1,)

        X_test = self.df_test[self.input_list].to_numpy()
        y_test = self.df_test[self.output_list].to_numpy().reshape(-1,)

        # shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    dataset_path = '../../dataset/dataset_keyhole.csv'
    input_list = ['etaP', 'Vs', 'r0', 'alpha', 'rho', 'cp', 'Tl-T0']
    output_list = ['e*']

    data_loader = Dataset(dataset_path, input_list, output_list)
    X_train, y_train, X_test, y_test = data_loader.parser()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
