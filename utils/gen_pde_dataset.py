# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Aug, 2021
'''

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

matplotlib.use('Agg')

class SpringMassDataset(object):
    def __init__(self, k, m, A0, c, v0=0, et=10):
        super(SpringMassDataset, self).__init__()
        self.k = k
        self.m = m
        self.A0 = A0
        self.c = c
        self.et = et
        self.v0 = v0
        self.Nt = int(1000)

        self.omega_n = np.sqrt(k / m)
        self.xi = c / 2 / np.sqrt(m * k)
        self.omega_d = self.omega_n * np.sqrt(1 - self.xi**2)
        self.A = np.sqrt(A0**2 + ((v0 + self.xi * self.omega_n * A0) / self.omega_d)**2)
        self.phi = np.arctan(self.omega_d * A0 / (v0 + self.xi * self.omega_n * A0))

    def solution(self):
        t = np.linspace(0, self.et, self.Nt, endpoint=False)
        x = self.A * np.exp(-self.xi * self.omega_n * t) * np.sin(self.omega_d * t + self.phi)

        fig = plt.figure()
        plt.plot(t, x)
        plt.savefig(f'../results/pde/pde_{self.k}_{self.m}_{self.c}_{self.A0}.jpg', dpi=300)
        plt.close()

        d = {
            't': t,
            'x': x,
            'k': [self.k] * self.Nt,
            'm': [self.m] * self.Nt,
            'A0': [self.A0] * self.Nt,
            'c': [self.c] * self.Nt,
            't_new': t / np.sqrt(self.k / self.m),
            'x_new': x / self.A0,
        }
        df = pd.DataFrame(d)
        # print(df.head())
        return df

def main():
    # k = 1
    # m = 1
    # A0 = 0.1

    res = []
    # c = 0.0001
    num = 0

    param_list = []
    for k in [5, 10, 20]:
        for m in [1, 2, 3]:
            for A0 in [0.1, 0.5, 1]:
                for c in [0.4, 0.6, 0.8]:
                    param_list.append((k, m, A0, c))

    random.seed(3)
    random.shuffle(param_list)

    # same pi3
    # param_list = [
    #     (3, 3, 0.1, 0.3),
    #     (4, 4, 1, 0.4)
    # ]

    # param_list = [
    #     (3, 3, 0.1, 0.3),
    #     (5, 1, 1, 0.8)
    # ]

    for (k, m, A0, c) in param_list[:3]:
        print(k, m, A0, c)
        dataset = SpringMassDataset(k, m, A0, c)
        df_each = dataset.solution()
        # df_each.to_csv(f'../results/dataset_oscillation_{num}.csv')
        res.append(df_each)
        num += 1

    df_all = pd.concat(res)
    df_all.to_csv('../dataset/dataset_oscillation.csv')
    print(df_all.head(), df_all.shape)

    # df_part = df_all.sample(n=500)
    # df_part.to_csv('../results/dataset_oscillation_500.csv')


    # k = 1
    # m = 1
    # A0 = 0.5
    # c = 0.01

    # dataset = SpringMassDataset(k, m, A0, c)
    # df_each = dataset.solution()
    # df_each.to_csv('../dataset/dataset_oscillation_500.csv')

if __name__ == '__main__':
    main()
