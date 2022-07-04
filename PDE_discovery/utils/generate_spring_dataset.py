# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Feb, 2022
'''

import numpy as np
import pandas as pd
from scipy.integrate import odeint

class SpringMassDataset2(object):
    '''
    Generate data for spring-mass-damping systems
    '''
    def __init__(self, k, m, A0, c, v0=0, et=20):
        super(SpringMassDataset2, self).__init__()
        self.k = k
        self.m = m
        self.A0 = A0
        self.c = c
        self.et = et
        self.v0 = v0
        self.Nt = int(500)
        
    def solution(self):
        # Initialization
        increment = self.et / self.Nt
        
        # Initial condition
        x_init = [self.A0, self.v0]
        t = np.linspace(0, self.et, self.Nt, endpoint=False)
        
        # Function that returns dx/dt
        def mydiff(x, t):
            dx1dt = x[1]
            dx2dt = (- self.c*x[1] - self.k*x[0]) / self.m
            dxdt = [dx1dt, dx2dt]
            return dxdt
        
        # Solve ODE
        x = odeint(mydiff, x_init, t)
        x1 = x[:,0] # displacement
        x2 = x[:,1] # velocity
        
        info = {'t': t, 'x': x1}
        df = pd.DataFrame(info)
        return df

class SpringMassDataset(object):
    '''
    Generate data for spring-mass-damping systems
    '''
    def __init__(self, k, m, A0, c, v0=0, et=20):
        super(SpringMassDataset, self).__init__()
        self.k = k
        self.m = m
        self.A0 = A0
        self.c = c
        self.et = et
        self.v0 = v0
        self.Nt = int(800)

        self.omega_n = np.sqrt(k / m)
        self.xi = c / 2 / np.sqrt(m * k)
        self.omega_d = self.omega_n * np.sqrt(1 - self.xi**2)
        self.A = np.sqrt(A0**2 + ((v0 + self.xi * self.omega_n * A0) / self.omega_d)**2)
        self.phi = np.arctan(self.omega_d * A0 / (v0 + self.xi * self.omega_n * A0))

    def solution(self):
        t = np.linspace(0, self.et, self.Nt, endpoint=False)
        x = self.A * np.exp(-self.xi * self.omega_n * t) * np.sin(self.omega_d * t + self.phi)
        info = {'t': t, 'x': x}
        df = pd.DataFrame(info)
        return df

    
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    k, m, A0, c = 20, 20, 0.5, 30
    dataset = SpringMassDataset2(k, m, A0, c)
    data_new = dataset.solution()
    
    dataset = SpringMassDataset(k, m, A0, c)
    data_old = dataset.solution()
    fig = plt.figure()
    plt.plot(data_old['t'], data_old['x'], label='old')
    plt.plot(data_new['t'], data_new['x'], label='new')
    plt.legend()
    plt.savefig('../results/1.jpg', dpi=300)