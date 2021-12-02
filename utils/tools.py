# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jan. 28th 2021
'''

import json
import numpy as np


def calc_metrics(y_true, y_pred, source, metric_list):
    '''
    calculate the results of different metrics with val
    '''
    metric_res = {}
    for metric_name in metric_list:
        if metric_name == 'mean_relative_error':
            result_train = calc_mre(y_true, y_pred)
        else:
            result_train = eval(
                'metrics.{}(y_true, y_pred)'.format(metric_name))
        metric_res[metric_name+'_'+source] = round(float(result_train), 4)
    return metric_res

def calc_mre(y_true, y_pred):
    '''
    calculate mean_relative_error
    
    # example
    y_true = np.array([1, 2, 3]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('np.absolute(y_pred - y_true)', np.absolute(y_pred - y_true))
    print('np.absolute((y_pred - y_true) / y_true)', np.absolute((y_pred - y_true) / y_true))
    print(calc_mre(y_true, y_pred))
    '''
    # mre = np.mean(np.absolute((y_pred - y_true) / (y_true + 1e-10)))
    normalizer = np.mean(np.sqrt(y_true**2))
    # diff = torch.norm()
    # normalizer = torch.norm(y_true, 1)
    mre = np.mean(np.absolute(y_pred - y_true)) / (normalizer + 1e-5)
    return mre


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    # test calc_mre
    y_true = np.array([1, 2, 3]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('np.absolute(y_pred - y_true)', np.absolute(y_pred - y_true))
    print('np.absolute((y_pred - y_true) / y_true)', np.absolute((y_pred - y_true) / y_true))
    print(calc_mre(y_true, y_pred))
