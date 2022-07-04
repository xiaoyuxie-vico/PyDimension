import numpy as np
from sklearn.linear_model import Ridge
from derivative import dxdt


def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    
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

    # n = len(x)
    n = x.shape[0]
    du = np.zeros((n - 2*width,diff))
    u_new = np.zeros((n - 2*width, 1))
    
    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        points = np.arange(j - width, j + width + 1)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])
        # print('poly(x[j])', poly(x[j]))
        u_new[j-width, 0] = poly(x[j])

    return u_new, du

def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    
    """
    Same as above but now just looking at a single point
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """
    
    n = len(x)
    if index == None: index = (n-1)/2

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x,u,deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1,diff+1):
        derivatives.append(poly.deriv(m=d)(x[index]))
    
    print(len(derivatives))
    return derivatives


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    n = 1000
    x = np.linspace(-10, 10, n)
    y_clean = np.sin(x)
    noise = np.random.normal(0, np.std(y_clean), y_clean.shape) * 0.05
    y_grad = np.cos(x)
    y_noise = y_clean + noise
    dx = x[1] - x[0]

    width = 25
    y_new, x_dot = PolyDiff(y_noise, x, deg=1, width=width)

    fig = plt.figure()
    plt.plot(y_grad[width:-width], label='grad_true')
    plt.plot(x_dot, label='grad_pred')
    plt.legend(fontsize=15, loc=[1., 0])
    plt.savefig('../../results/1.jpg', dpi=300)
    
    y_clean = y_clean[width:-width]
    fig = plt.figure()
    print(y_new.flatten()[:20])
    print(y_clean.flatten()[:20])
    plt.plot(y_new, label='y_poly')
    plt.plot(y_clean, label='y_clean')
    # plt.plot(y_noise, label='y_noise')
    plt.legend(fontsize=15, loc=[1., 0])
    plt.savefig('../../results/2.jpg', dpi=300)
