# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Feb, 2022
'''

from derivative import dxdt
import sys
sys.path.append('/home/xie/projects/DimensionNet/discovery/utils')

import numpy as np
from sklearn.linear_model import Ridge, LassoLarsCV
from tvregdiff.tvregdiff import TVRegDiff
from PolyDiff import PolyDiffPoint

def TVRegDiffPoint(u, dx, index=None):
    u = u.flatten()
    n = len(u)

    if index == None: 
        index = int((n-1)/2)

    ux = TVRegDiff(u, 1, 0.1, dx=dx, plotflag=False,
                diffkernel='sq')
    uxx = TVRegDiff(ux, 1, 0.1, dx=dx, plotflag=False,
                diffkernel='sq')

    return ux[index], uxx[index]
    # return ux[index], uxx[index], ux, uxx


def test_TVRegDiff():
    n = 200
    x = np.linspace(-10, 10, n)
    np.random.seed(1)
    noise = np.random.normal(0, np.std(x), x.shape) * 0.05
    y_clean = np.sin(x)
    y_grad = np.cos(x)
    y_noise = y_clean + noise
    dx = x[1] - x[0]
    
    width = 60 # for ploy
    ########################Original data and noise data########################
    fig = plt.figure(figsize=(12, 4))
    fig.add_subplot(1, 3, 1)
    plt.plot(x[width:-width], y_noise[width:-width], label='Noisy data')
    plt.plot(x[width:-width], y_clean[width:-width], label='Clean data')
    # plt.legend(fontsize=15, loc=[1., 0])
    plt.legend(fontsize=10, loc=1)
    # plt.xlim([0, 10])
    plt.xlabel('t', fontsize=12)
    plt.ylabel('x', fontsize=12)
    plt.title('Clean and noisy data')
    
    ########################TVRegDiff############################################
    y_dot = TVRegDiff(y_noise, 1, 0.5, dx=dx, plotflag=False, diffkernel='sq')
    y_2dot = TVRegDiff(y_dot, 1, 0.5, dx=dx, plotflag=False, diffkernel='sq')

    fig.add_subplot(1, 3, 2)
    plt.plot(x[width:-width], y_grad[width:-width], label='Ground truth')
    plt.plot(x[width:-width], y_dot[width:-width], label='Prediction')
    # plt.legend(fontsize=15, loc=[1., 0])
    plt.legend(fontsize=10, loc=1)
    plt.title('TVRegDiff')
    plt.xlabel('t', fontsize=12)
    plt.ylabel('dx/dt', fontsize=12)
    # plt.xlim([0, 10])

    ########################PolyDiff############################################
    y_dot, y_2dot = [], []
    boundary_num = 10
    deg = 3
    for i in range(boundary_num, n-boundary_num):
        y_part = y_noise[i-boundary_num: i+boundary_num]
        x_diff = PolyDiffPoint(y_part, np.arange(2*boundary_num)*dx, deg, 2)
        y_dot.append(x_diff[0])
        y_2dot.append(x_diff[1])
    y_dot, y_2dot = np.array(y_dot), np.array(y_2dot)

    fig.add_subplot(1, 3, 3)
    plt.plot(x[width:-width], y_grad[width:-width], label='Ground truth')
    plt.plot(x[width:-width], y_dot[width-boundary_num:-width+boundary_num], label='Prediction')
    plt.legend(fontsize=10, loc=1)
    plt.title('PolyDiff')
    plt.xlabel('t', fontsize=12)
    plt.ylabel('dx/dt', fontsize=12)

    ########################Save results############################################
    plt.tight_layout()
    plt.savefig('../../results/test_TVRegDiff_compare.jpg', dpi=300)


def test_TVRegDiffPoint():
    '''
    test only calculate a center point
    '''
    n = 800
    x = np.linspace(-10, 10, n)
    np.random.seed(1)
    noise = np.random.normal(0, np.std(x), x.shape) * 0.05
    y_clean = np.sin(x)
    y_grad = np.cos(x)
    y_noise = y_clean + noise
    dx = x[1] - x[0]
    
    ux_list_true = []
    ux_list, uxx_list = [], []
    for i in range(20, 800-21):
        y_clean_sub = y_noise[i: i+20]
        # y_clean_sub = y_clean[i: i+20]
        ux_point, uxx_point = TVRegDiffPoint(y_clean_sub, dx)
        sub_n = y_clean_sub.shape[0]
        index = int((sub_n-1)/2)

        ux_list_true.append(y_grad[index+i])
        ux_list.append(ux_point)

    fig = plt.figure()
    plt.plot(range(len(ux_list)), ux_list, label='ux_list')
    plt.plot(range(len(ux_list_true)), ux_list_true, label='ux_list_true')
    plt.savefig('../../results/test_TVRegDiff.jpg', dpi=300)


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    test_TVRegDiff()

    # test_TVRegDiffPoint()

    