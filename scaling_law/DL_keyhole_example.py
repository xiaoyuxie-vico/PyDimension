# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jun, 2022
'''

import sys
sys.path.append('/home/xie/projects/DimensionNet/tutorials/utils')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset
from solver import DimensionlessLearning

matplotlib.use('Agg')
plt.rcParams["font.family"] = 'Arial'
np.set_printoptions(suppress=True)


def test_keyhole_example(method):
    '''
    keyhole problem
    '''
    ################################### config ###################################
    dataset_path = '../dataset/dataset_keyhole.csv'
    input_list = ['etaP', 'Vs', 'r0', 'alpha', 'rho', 'cp', 'Tl-T0']
    output_list = ['e*']
    
    # dimension matrix
    D_in = np.array(
        [
            [2., 1., 1., 2., -3., 2., 0.],
            [-3., -1., 0., -1., 0., -2., 0.],
            [1., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., -1., 1.],
        ],
    )
    D_out = np.array(
        [
            [1.],
            [0.],
            [0.],
            [0.],
        ]
    )
    # best weights for Ke: 0.5, 1, 1
    # basis vectors in columns
    scaling_mat = np.array(
        [
            [0., 0., 1],
            [1, 2., -3],
            [1, 0., -2.],
            [-1, 0., 0.],
            [0., 0., -1],
            [0., -1, 0.],
            [0., -1, 0.]],
    )
    deg = 5

    ################################### dataset ########################################
    # load, split, and shuffle the dataset
    data_loader = Dataset(dataset_path, input_list, output_list)
    X_train, y_train, X_test, y_test = data_loader.parser()
    print(f'[Dataset] X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'[Dataset] X_test: {X_test.shape}, y_test: {y_test.shape}')

    ################################### training and testing ###########################
    print('[Training]')
    dimensionless_learning = DimensionlessLearning(X_train, y_train, scaling_mat)
    r2, power_index, scaling_coef = dimensionless_learning.fit(method=method)
    print(f'Final r2: {r2:.4f}, power_index: {power_index}, scaling_coef: {scaling_coef}')
    pred_train = dimensionless_learning.predict(X_train, power_index, scaling_coef, deg)
    pred_test = dimensionless_learning.predict(X_test, power_index, scaling_coef, deg)

    ################################### visualization ###################################
    print('[Visualization]')
    fig = plt.figure()
    plt.scatter(pred_train, y_train, label='Training set')
    plt.scatter(pred_test, y_test, label='Test set')
    plt.legend(fontsize=16)
    plt.savefig('../results/1.jpg', dpi=300)


if __name__ == '__main__':
    method = 'pattern_search'
    test_keyhole_example(method)
