# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jan. 28th 2021
'''

import numpy as np 


def calculate_BIC(preds, targets, non_zero_term):
    '''
    calculate BIC
    modified the second part
    '''
    points_num = preds.shape[0]
    preds, targets = np.array(preds), np.array(targets)
    residuals = preds - targets
    # print('points_num * np.log(np.var(residuals))', points_num * np.log(np.var(residuals)))
    # print('np.exp(non_zero_term) * np.log(points_num)', np.exp(non_zero_term) * np.log(points_num))
    criterion = points_num * np.log(np.var(residuals)) \
            + np.exp(non_zero_term) * np.log(points_num)
    return criterion


def calculate_BIC_formal(preds, targets, non_zero_term):
    '''
    calculate BIC formal
    '''
    points_num = preds.shape[0]
    preds, targets = np.array(preds), np.array(targets)
    residuals = preds - targets
    criterion = points_num * np.log(np.var(residuals)) \
            + non_zero_term * np.log(points_num)
    return criterion

if __name__ == "__main__":
    preds = [1, 2, 3]
    targets = [1, 0, 1]
    non_zero_term = 1
    criterion = calculate_BIC(preds, targets, non_zero_term)
    print(criterion)
