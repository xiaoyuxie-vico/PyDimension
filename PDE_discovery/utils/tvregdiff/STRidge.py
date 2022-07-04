import numpy as np

def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.
    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y), rcond=None)[0] # modification
    else: w = np.linalg.lstsq(X,y, rcond=None)[0]  # modification
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        # # modification
        if lam != 0: 
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y), rcond=None)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y, rcond=None)[0] # modification

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]  # modification
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w

def print_pde(w, rhs_description, ut = 'u_t'):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print(pde)

def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = False):
    """
    This function trains a predictor using STRidge.
    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.
    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY, rcond=None)[0]  # modification
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: 
        print("Optimal tolerance:", tol_best)

    return w_best


if __name__ == '__main__':
    import copy
    from derivative import dxdt

    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

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
            self.Nt = int(500)

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
    
    k, m, A0, c = 2, 10, 0.5, 3
    dataset = SpringMassDataset(k, m, A0, c)
    data = dataset.solution()
    # fig = plt.figure()
    # plt.plot(data['t'], data['x'], label='old')
    # plt.savefig('../../results/1.jpg, dpi=300')

    t = data['t'].to_numpy()
    x = data['x'].to_numpy()

    x_dot = dxdt(x, t, kind="finite_difference", k=1)
    x_2dot = dxdt(x_dot, t, kind="finite_difference", k=1)

    X_library = np.stack(
        (
            x, 
            x**2, 
            x_2dot, 
            np.multiply(x.reshape(-1, 1), x_dot.reshape(-1, 1)).reshape(-1,)
        ),
        axis=-1)
    y_library = x_dot
    description = ['x', 'x^2', 'x_2dot', 'xx_dot']

    # ################################squential threshold################################
    # threshold = 0.5    
    # # initialize a linear regression model
    # model = LinearRegression(fit_intercept=True)
    # # fit the model
    # model.fit(X_library, y_library)
    # # compute the R^2 score
    # r2_train = model.score(X_library, y_library)

    # # loops for 3 iterations to find a sparse coefficients
    # for i in range(3):
    #     # obtain coefficients
    #     coef = model.coef_
    #     # if the coefficient is lower than the threshold
    #     # set the corresponding X_library column as 0
    #     flag = np.repeat((np.abs(coef) > threshold).astype(int).reshape(1,-1), 
    #                      X_library.shape[0], axis=0)
    #     X1 = copy.copy(X_library)
    #     X1 = np.multiply(X1, flag)
    #     # fit with a new X_library
    #     model.fit(X1, y_library)
    #     # compute the current R^2 score
    #     r2_train = model.score(X1, y_library)
    #     print('Iteration:', i, ' R-squared score: ', round(r2_train, 4), [round(i, 4) for i in coef.tolist()])

    ###############################STRidge################################
    lam = 10**-5
    d_tol = 5
    coef = TrainSTRidge(X_library, y_library.reshape(-1, 1), lam, d_tol).real
    print(coef)
    print_pde(coef, description, ut = 'x_dot')
