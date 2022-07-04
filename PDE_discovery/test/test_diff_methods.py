# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Feb, 2022
'''

from derivative import dxdt
import sys
# sys.path.append('/Users/vico/projects/DimensionNet')
sys.path.append('/home/xie/projects/DimensionNet')

import numpy as np
import pysindy as ps
from sklearn.linear_model import Ridge, LassoLarsCV

from utils.tvregdiff.poly_diff import PolyDiff, PolyDiffPoint
from utils.tvregdiff.tvregdiff import TVRegDiff


def PolyFitDiff(u, x, deg = 3, diff = 1, width = 5):
    
    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial
    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        points = np.arange(j - width, j + width + 1)
        
        # Fit to a polynomial
        reg = LassoLarsCV(cv=5, normalize=False)
        reg.fit(x[points].reshape(-1, 1), u[points].reshape(-1,))

        x_fine = np.linspace(x[points[0]], x[points[-1]], 500)
        y_pred = reg.predict(x_fine.reshape(-1, 1))

        x_dot = dxdt(y_pred.reshape(-1,), x_fine, kind="finite_difference", k=1)

        # Take derivatives
        du[j-width, 0] = x_dot[250]

    return du

def test_TVRegDiff():
    n = 800
    x = np.linspace(-10, 10, n)
    print(np.std(x))
    np.random.seed(1)
    noise = np.random.normal(0, np.std(x), x.shape) * 0.05
    y_clean = np.sin(x)
    y_grad = np.cos(x)
    y_noise = y_clean + noise
    dx = x[1] - x[0]
    
    width = 60 # for ploy
    ########################Original data and noise data########################
    fig = plt.figure(figsize=(7, 7))
    fig.add_subplot(2, 2, 1)
    plt.plot(x[width:-width], y_noise[width:-width], label='Noisy data')
    plt.plot(x[width:-width], y_clean[width:-width], label='Clean data')
    # plt.legend(fontsize=15, loc=[1., 0])
    plt.legend(fontsize=10, loc=1)
    # plt.xlim([0, 10])
    plt.xlabel('t', fontsize=12)
    plt.ylabel('x', fontsize=12)
    plt.title('Clean and noisy data')
    
    ########################TVRegDiff############################################
    #     u1a = TVRegDiff(y_noise, 2, 0.2, dx=dx, plotflag=False,
#                     precondflag=False)
    u1a = TVRegDiff(y_noise, 1, 0.5, dx=dx, plotflag=False,
                diffkernel='sq')
    
    fig.add_subplot(2, 2, 2)
    plt.plot(x[width:-width], y_grad[width:-width], label='Ground truth')
    plt.plot(x[width:-width], u1a[width:-width], label='Prediction')
    # plt.legend(fontsize=15, loc=[1., 0])
    plt.legend(fontsize=10, loc=1)
    plt.title('TVRegDiff')
    plt.xlabel('t', fontsize=12)
    plt.ylabel('dx/dt', fontsize=12)
    # plt.xlim([0, 10])
    
    ########################finite difference####################################
    # grad_f = ps.SINDyDerivative(kind='trend_filtered', order=0, alpha=1e-2)
    grad_f = ps.SINDyDerivative(kind='finite_difference', k=1)
#     grad_f = ps.SINDyDerivative(kind='savitzky_golay', left=0.1, right=0.1, order=3)
    x_dot = grad_f(y_noise, x)
    x_2dot = grad_f(x_dot, x)
    
    fig.add_subplot(2, 2, 3)
    plt.plot(x[width:-width], y_grad[width:-width], label='Ground truth')
    plt.plot(x[width:-width], x_dot[width:-width], label='Prediction')
    # plt.legend(fontsize=15, loc=[1., 0])
    plt.legend(fontsize=10, loc=1)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('dx/dt', fontsize=12)
    plt.title('FiniteDiff')
    # plt.xlim([0, 10])
    
    #########################polynomial##########################################
    _, x_dot = PolyDiff(y_noise, x, width=width, deg=3)
    
    fig.add_subplot(2, 2, 4)
    plt.plot(x[width:-width], y_grad[width:-width], label='Ground truth')
    plt.plot(x[width:-width], x_dot, label='Prediction')
    # plt.legend(fontsize=15, loc=[1., 0])
    plt.legend(fontsize=10, loc=1)
    plt.title('PolyDiff')
    plt.xlabel('t', fontsize=12)
    plt.ylabel('dx/dt', fontsize=12)
    # plt.xlim([0, 10])
    
    plt.tight_layout()
    plt.savefig('../results/2.jpg', dpi=300)
    
    
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    test_TVRegDiff()
