#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import inv
import pandas as pd
import pysindy as ps
import random as rd
from scipy.sparse import data
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.optimize import least_squares
from tqdm import tqdm
matplotlib.use('Agg')

rcParams['figure.figsize'] = (8, 5)
plt.rcParams["font.family"] = "Arial"
rcParams['legend.fontsize'] = 14
rcParams['axes.labelsize'] = 16
plt.rcParams['font.size'] = '16'

rd.seed(0)


# In[12]:


df0 = pd.read_csv('../results/pde_dataset/dataset_oscillation_0.csv')
df1 = pd.read_csv('../results/pde_dataset/dataset_oscillation_1.csv')
df2 = pd.read_csv('../results/pde_dataset/dataset_oscillation_2.csv')


# In[13]:


df1.head()


# In[34]:

fig = plt.figure(figsize=(5,4))
plt.plot(df0['t'], df0['x'], label='Data 1')
plt.plot(df1['t'], df1['x'], label='Data 2')
plt.plot(df2['t'], df2['x'], label='Data 3')
plt.legend(fontsize=12)
plt.xlabel(r'Time: $t$'+' (s)', fontsize=16)
plt.ylabel(r'Displacement: $x$'+' (m)', fontsize=16)
plt.tight_layout()
plt.savefig('../results/pde_data_each.jpg', dpi=300)
# In[ ]:




