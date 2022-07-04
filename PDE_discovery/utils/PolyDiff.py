# -*- coding: utf-8 -*-

'''
Original code: https://github.com/snagcliffs/PDE-FIND
'''

from pyexpat import model
import numpy as np

def PolyDiffPoint(u, x, deg=3, diff=1, index=None):
    '''
    Poly diff
    '''
    n = len(x)
    # if index == None: index = int((n-1)/2)
    if index == None: index = (n-1)//2

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x, u, deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1, diff + 1):
        derivatives.append(poly.deriv(m=d)(x[index]))
    
    return derivatives

def PolyDiffPointNoise(u, x, deg=3, diff=1, index=None):
    '''
    Poly diff
    '''
    n = len(x)
    # if index == None: index = int((n-1)/2)
    if index == None: index = (n-1)//2

    # Fit to a polynomial
    coefs, poly = np.polynomial.chebyshev.Chebyshev.fit(x, u, deg)

    # Take derivatives
    derivatives = []
    for d in range(1, diff + 1):
        derivatives.append(poly.deriv(m=d)(x[index]))

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)
    poly_transform = PolynomialFeatures(3)   
    x_poly = poly_transform.fit_transform(x)
    reg = LinearRegression().fit(x_poly, u)
    r2 = reg.score(x_poly, u)
    u_reconstruct = reg.predict(x_poly)[index][0]

    return u_reconstruct, derivatives

def PolyDiffNoiseV2(u, x, deg=3, diff=1, index=None):
    '''
    Poly diff
    '''
    # n = len(x)
    # # if index == None: index = int((n-1)/2)
    # if index == None: index = (n-1)//2

    # # Fit to a polynomial
    # coefs, poly = np.polynomial.chebyshev.Chebyshev.fit(x, u, deg)

    # # Take derivatives
    # derivatives = []
    # for d in range(1, diff + 1):
    #     derivatives.append(poly.deriv(m=d)(x[index]))

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)
    poly_transform = PolynomialFeatures(3)   
    x_poly = poly_transform.fit_transform(x)
    reg = LinearRegression().fit(x_poly, u)
    r2 = reg.score(x_poly, u)
    u_reconstruct = reg.predict(x_poly)

    # from derivative import dxdt

    # ut = dxdt(u_reconstruct.reshape(-1,), x.reshape(-1,))

    # x = x.reshape(-1, 1)
    # ut = ut.reshape(-1, 1)
    # poly_transform = PolynomialFeatures(3)   
    # x_poly = poly_transform.fit_transform(x)
    # reg = LinearRegression().fit(x_poly, ut)
    # r2 = reg.score(x_poly, ut)
    # ut_reconstruct = reg.predict(x_poly)

    # utt = dxdt(ut_reconstruct.reshape(-1,), x.reshape(-1,))

    return u_reconstruct.reshape(-1,)
    # return u_reconstruct.reshape(-1,), ut.reshape(-1,), utt.reshape(-1,)


if __name__ == '__main__':
    n = 101
    x = np.linspace(0, 10, n)
    y = x ** 2
    dx = x[1] - x[0]
    
    y_diff = PolyDiffPoint(y, x, deg=5, diff=1)[0]
    print(f'Calcualte xt: {y_diff}, truth: 10')

    y_diff = PolyDiffPoint(y, x, deg=5, diff=2)
    print(f'Calcualte xt: {y_diff[0]}, xtt: {y_diff[1]}, truth: 10, 2')

    #test noisy data
    print('test noisy data')
    y = x ** 2 + 0.01 * np.std(y) * np.random.randn(n)
    x_reconstruct, y_diff = PolyDiffPointNoise(y, x, deg=5, diff=2)
    print('x_reconstruct', x_reconstruct)
    print(f'Calcualte xt: {y_diff[0]}, xtt: {y_diff[1]}, truth: 10, 2')
