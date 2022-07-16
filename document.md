# How to find the basis vectors

For the turbulent Rayleigh-Benard convection case, you can calculate the basis vectors using python or matlab.

Python:
```
from sympy import Matrix
import numpy as np
D = np.array([
    [1, 0, 1, 1, 0, 2, 2], 
    [0, 0, -3, -2,0, -1, -1], 
    [0, 0, 1, 0, 0, 0, 0], 
    [0, 1, -1, 0, -1, 0, 0]
])
D = Matrix(D)
basis_vectors = D.nullspace()
```

Matlab:
```
D = [1 0 1 1 0 2 2; 0 0 -3 -2 0 -1 -1; 0 0 1 0 0 0 0; 0 1 -1 0 -1 0 0]
basis_vectors = null(D, 'r')
# basis_vectors =
# 
#          0   -1.5000   -1.5000
#     1.0000         0         0
#          0         0         0
#          0   -0.5000   -0.5000
#     1.0000         0         0
#          0    1.0000         0
#          0         0    1.0000
```
To simplify the basis vectors, the second column can substract the third column, and the second column can times 2. Then, we obtain the final basis vectors:

$\boldsymbol{w_{b1}} = [0,1,0,0,1,0,0]^T,$

$\boldsymbol{w_{b2}} = [0,0,0,0,0,1,-1]^T,$

$\boldsymbol{w_{b3}} = [3,0,0,1,0,-2,0]^T.$
