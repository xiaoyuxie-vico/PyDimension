from os import XATTR_SIZE_MAX
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy
from sympy.utilities.lambdify import lambdify
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
import copy

matplotlib.use('Agg')
np.set_printoptions(suppress=True)

class SeqReg(object):

    def __init__(self):
        pass

    def normalize(self, X, y):
        '''
        Normalization the data
        '''
        norm_coef_X = np.mean(np.abs(np.mean(X, axis=0)))
        norm_coef_y = np.mean(np.abs(np.mean(y, axis=0)))
        norm_coef = min(norm_coef_X, norm_coef_y)
        # print('Before X', pd.DataFrame(np.concatenate([X, y], axis=1)).describe())
        X = X / norm_coef
        y = y / norm_coef
        # print('After X', pd.DataFrame(np.concatenate([X, y], axis=1)).describe())
        return X, y

    def fit_fixed_threshold(self, X, y, alpha=1.0, threshold=0.005, is_normalize=True):
        if is_normalize:
            X, y = self.normalize(X, y)
        
        # initialize a linear regression model
        # model = LinearRegression(fit_intercept=False)
        model = Ridge(fit_intercept=False, alpha=1)
        model.fit(X, y)
        # r2 = model.score(X, y)
        for idx in range(3):
            coef = model.coef_
            flag = np.repeat((np.abs(coef) > threshold).astype(int).reshape(1,-1), 
                             X.shape[0], axis=0)
            X1 = copy.copy(X)
            X1 = np.multiply(X1, flag)
            model.fit(X1, y)
            r2 = model.score(X1, y)
            print(f'training {idx} r2: {r2}')
        coef = np.squeeze(model.coef_)
        return coef, X1

    def fit_dynamic_thresh(self, X, y, non_zero_term=4, alpha=1.0, threshold=0.005, 
                is_normalize=True, fit_intercept=False, model_name='Ridge', max_iter=200):
        '''
        decrease the threshold when there are only limited non-zero terms
        and increase the threshold when thre are more non-zeros terms
        '''
        if is_normalize:
            X, y = self.normalize(X, y)
        
        # initialize a linear regression model
        if model_name == 'Ridge':
            model = Ridge(fit_intercept=fit_intercept, alpha=alpha)
        elif model_name == 'LR':
            model = LinearRegression(fit_intercept=fit_intercept)
        else:
            raise Exception('Wrong model_name.')
        model.fit(X, y)
        count = 0

        while count <= max_iter:
            coef = model.coef_
            flag = np.repeat((np.abs(coef) > threshold).astype(int).reshape(1,-1), 
                             X.shape[0], axis=0)
            cur_non_zero_term = np.sum(flag[0,:])
            X1 = copy.copy(X)
            X1 = np.multiply(X1, flag)
            model.fit(X1, y)
            r2 = model.score(X1, y)
            # print(f'training r2: {r2}, threshold: {threshold}, cur_non_zero_term: {cur_non_zero_term}')
            if cur_non_zero_term == non_zero_term:
                break
            elif cur_non_zero_term < non_zero_term:
                threshold *= 0.95
            else:
                threshold *= 1.05
            count += 1

        coef = np.squeeze(model.coef_)
        if fit_intercept:
            coef_list = coef.tolist()
            coef_list.append(float(model.intercept_))
            coef = np.array(coef_list)

        return coef, X1, r2


if __name__ == '__main__':
    df = pd.read_csv('../results/test_vorticity_data.csv')

    Re = 50
    print(df.head())

    X = df[[str(i) for i in range(29)]].to_numpy()
    y = df['29'].to_numpy().reshape(-1, 1)

    model = SeqReg()
    coef, X, r2 = model.fit_dynamic_thresh(X, y, is_normalize=False)

    names = [
        'u', 'v', 'w', 'wx', 'wy', 'wxx', 'wyy', 'wxy',
        'u*u', 'u*v', 'u*w', 'u*wx', 'u*wy', 'u*wxx', 'u*wyy', 'u*wxy',
        'v*v', 'v*w', 'v*wx', 'v*wy', 'v*wxx', 'v*wyy', 'v*wxy',
        'w*w', 'w*wx', 'w*wy', 'w*wxx', 'w*wyy', 'w*wxy',
    ]
    
    coef_res = [(each[0], round(each[1], 6)) for each in list(zip(names, coef.tolist())) if abs(each[1]) !=0.]
    coef_res = sorted(coef_res, key=lambda x: abs(x[1]), reverse=True)
    print(f'coef_res: {coef_res}')
    print('coef_best', 1/Re)
