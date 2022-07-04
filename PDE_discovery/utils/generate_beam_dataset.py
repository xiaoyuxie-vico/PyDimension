# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:18:01 2022

@author: Jiachen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class Plasticbeam():
    '''
    Generate data for plastic beam under impulsive loading (small deformation, no axial force considered)
    '''
    def __init__(self, rho, H, L, V0, sigma0):
        super(Plasticbeam, self).__init__()
        self.rho = rho #density
        self.H = H #beam height
        self.L = L #half length of the beam
        self.m = rho * H #line density of unit width beam
        self.V0 = V0 #impulse load
        self.sigma0 = sigma0 #yield stress
        self.M0 = sigma0 * H ** 2 / 4 #yield moment
        self.pc = 2 * self.M0 / L**2 #critical loading
        self.T1 = self.m * V0 * L **2 / (6 * self.M0)   #time for 1st stage     
        self.T = self.m * V0 * L **2 / (2 * self.M0) #total time for 2 stages
    def grid(self, t_step = 500, x_step = 500):
        # Initialization
        self.t = np.linspace(0,self.T, t_step) #time grid
        self.x = np.linspace(0,self.L, x_step) #space grid beam centered at the origin
        self.t = self.t[1:] #remove boundary sigular point
        self.x = self.x[:-1] #remove boundary sigular point
        self.t1 = self.t[ : int(t_step/3)] #first one third is first stage
        self.t2 = self.t[int(t_step/3) : ] #remaining two thirds are the second stage
        self.ksi = self.L - np.sqrt(6 * self.M0 * self.t1 / (self.m * self. V0)) #location of plastic hinge
        self.ksidot = -3 * self.M0 / (self.m * self.V0 * (self.L - self.ksi)) #velocity of plastic hinge
        d = {
            'Time': self.t,
            'Rho': self.rho,
            'Height': self.H,
            'Initial velocity': self.V0,
            'Line mass': self.m,
            'Yield strength' : self.sigma0,
        }
        self.df = pd.DataFrame(d)
        return self.x, self.ksi, self.ksidot
    
    def helper(self):
        self.A2 = -self.m * self.V0 * self.ksi * (self.L - self.ksi/2) * self.ksidot / (self.L-self.ksi) **2
        self.B2 = -self.m * self.V0 * self.L ** 3 * self.ksidot / 3 / (self.L - self.ksi)**2 - self.A2 * self.L
    def solution_m(self):
        m = np.zeros((self.t.shape[0], self.x.shape[0]))
        t1_size = self.t1.shape[0]
        t2_size = self.t2.shape[0]
        for i in range(t1_size):
            for j in range(self.x.shape[0]):
                if self.x[j] < self.ksi[i]:
                    m[i][j] = self.M0
                else:
                    m[i][j] = self.m * self.V0 * (self.L * self.x[j] ** 2/2 - self.x[j]**3/6) * self.ksidot[i] / (self.L-self.ksi[i]) ** 2 + self.A2[i] * self.x[j] + self.B2[i]
        for i in range(t2_size):
            a_mid = -3 * self.M0/ (self.m * self.L**2)
            for j in range (self.x.shape[0]):
                m[i + t1_size][j] = self.m * (self.x[j] **2 /2 - self.x[j] ** 3/ (6*self.L)) * a_mid + self.M0 
        #fig = plt.figure()
        #plt.plot(self.x, m[100,:])
        #plt.close()x
        self.df = self.df.join(pd.DataFrame(m))

        # print(df.he
        return m, self.df   # m input in the pde
            
    def solution_v(self):
        v = np.zeros((self.t.shape[0], self.x.shape[0]))
        t1_size = self.t1.shape[0]
        t2_size = self.t2.shape[0]
        for i in range(t1_size):
            for j in range(self.x.shape[0]):
                if self.x[j] < self.ksi[i]:
                    v[i][j] = self.V0
                else:
                    v[i][j] = self.V0 * (self.L - self.x[j]) / (self.L-self.ksi[i])
        v_mid = np.zeros(t2_size)
        for i in range(t2_size):
                v_mid[i] = -3 * self.M0 * self.t2[i] / (self.m * self.L **2) + 3 * self.m * self.V0 / (2 * self.m)
                for j in range (self.x.shape[0]):
                    v[i + t1_size][j] = v_mid[i] * (1 - self.x[j]/self.L)
        #fig = plt.figure()
        #plt.plot(self.x, v[100,:])
        #plt.close()
        self.df = self.df.join(pd.DataFrame(v),rsuffix = 'v_')
        return v, self.df # dxdt input in the pde
    def solution(self):
        return self.df
    def validate(self,m,v):
        t_size = m.shape[0]
        x_size = m.shape[1]
        dt = self.t[2] - self.t[1]
        dx = self.x[2] - self.x[1]
        dvdt = np.zeros((t_size - 2, x_size - 2)) #solve derivative using central difference method
        dm2dx2 = np.zeros((t_size - 2, x_size - 2))
        for i in range(1, t_size - 2):
            for j in range(1, x_size - 2):
                dvdt[i][j] = (v[i+1][j] - v[i-1][j]) / 2 / dt
                dm2dx2[i][j] = (m[i][j+1] - 2 * m[i][j] + m[i][j-1]) / dx**2
                
        residual = dm2dx2 - self.m * dvdt
        return dm2dx2, dvdt, residual

# plastic = Plasticbeam(2e-9, 300, 2500, 500, 100)
# x, ksi, ksidot = plastic.grid()
# plastic.helper()
# m, df = plastic.solution_m()
# v, df = plastic.solution_v()



# dm2dx2, dvdt, r = plastic.validate(m,v)


res = []
num = 0

param_list = []
for rho in [2e-9, 4e-9, 8e-9]: #unit t/mm&3
    for H in [300, 400, 500]: #unit mm
        for L in [2500, 3500, 4500]: #unit mm
            for V0 in [500, 1000, 1500]: #unit mm/s
                for sigma0 in [100, 300, 400]: #unit MPa
                    param_list.append((rho, H, L, V0, sigma0))

random.seed(3)
random.shuffle(param_list)



for (rho, H, L, V0, sigma0) in param_list[:3]: #number of set of parameters used
    print(rho, H, L, V0, sigma0)
    dataset = Plasticbeam(rho, H, L, V0, sigma0)
    dataset.grid()
    dataset.helper()
    dataset.solution_m()
    dataset.solution_v()
    df_each = dataset.solution()
    res.append(df_each)
    num += 1

df_all = pd.concat(res)
#output all the data: format---     Time Rho	Height	Initial velocity	Line mass	Yield strength moment(x) velocity(x)
df_all.to_csv('E:\\OneDrive - Northwestern University\\Liu Research\\Dimensionless_solid_example\\results\\results.csv')
print(df_all.head(), df_all.shape)