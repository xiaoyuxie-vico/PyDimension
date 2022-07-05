# Dimensionless learning for scientific knowledge discovery (dimensionless numbers, scaling laws, and PDEs)

## What is dimensionless learning

The proposed dimensionless learning is a powerful technique to identify scientific knowledge from data at multiple levels: 

- Dimensionless number at the feature level, such as the well-known Reynolds number $\mathrm{Re}=\frac{\rho V l}{\mu}$ in fluid mechanics.
- Scaling law at the algebraic equation level, such as a simple linar scaling law for vapor depression dynamics in 3D printing $e^*=0.12\mathrm{Ke}-0.30$.
- Governing equation at the differential equation level, such as Navier-Stokes equation $\frac{\partial \omega}{\partial t}+u\frac{\partial \omega}{\partial x}+v\frac{\partial \omega}{\partial y}=\mathrm{\frac{1}{Re}}(\frac{\partial^2 \omega}{\partial x^2} +\frac{\partial^2 \omega}{\partial y^2})$.

Unlike purely data-driven approaches that easily suffer from overfitting on small or noisy datasets, this method incorporates fundamental physical knowledge of dimensional invariance and symmetric invariance as physical constraints or regularizations into data-driven models to perform well on limited and/or noisy data. The embedded physical invariance reduces the learning space and eliminates the strong dependence between variables. This method is a physics-based dimension reduction approach that represents features as dimensionless numbers and transforms data points into a low-dimensional pattern that is unaffected by units and scales. Thus, in addition to being applicable to limited and/or noisy data, the presented approach significantly improves the interpretability of representation learning because dimensionless numbers are physically interpretable. Lower dimension and better interpretability also allow for qualitative and quantitative analysis of the systems of interest. 

## Where to find the paper

**Title: Data-driven discovery of dimensionless numbers and governing laws from scarce measurements.** 

Abstract:
Dimensionless numbers and scaling laws provide elegant insights into the characteristic properties of physical systems. Classical dimensional analysis and similitude theory fail to identify a set of unique dimensionless numbers for a highly multi-variable system with incomplete governing equations. This paper introduces a mechanistic data-driven approach that embeds the principle of dimensional invariance into a two-level machine learning scheme to automatically discover dominant dimensionless numbers and governing laws (including scaling laws and differential equations) from scarce measurement data. The proposed methodology, called dimensionless learning, is a physics-based dimension reduction technique. It can reduce high-dimensional parameter spaces to descriptions involving only a few physically interpretable dimensionless parameters, greatly simplifying complex process design and system optimization. We demonstrate the algorithm by solving several challenging engineering problems with noisy experimental measurements (not synthetic data) collected from the literature. Examples include turbulent Rayleigh-Bénard convection, vapor depression dynamics in laser melting of metals, and porosity formation in 3D printing. Lastly, we show that the proposed approach can identify dimensionally homogeneous differential equations with dimensionless number(s) by leveraging sparsity-promoting techniques.

You can find the preprint paper at [here](http://arxiv.org/abs/2111.03583).

## Paper highlights

- Automatically discover unique and dominant **dimensionless numbers** with clear physical meaning and **scaling laws** from complex systems based on experimental measurements;
- Identify **dimensionally homogeneous differential equations** with minimal parameters by leveraging sparsity-promoting techniques.

## Requirements
```
matplotlib==3.1.3
derivative==0.3.1
pandas==1.3.4
pyyaml==6.0
scikit-learn==0.24.2
pysindy==1.3.0
```

## Tutorials and examples

Go to the folder called `scaling_law` and `PDE_discovery` and run the jupyter notebook.

We will update other examples and simplify the codes in this repository.

## How to find the basis vectors
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


## Citations

```
@article{xie2021data,
  title={Data-driven discovery of dimensionless numbers and scaling laws from experimental measurements},
  author={Xie, Xiaoyu and Liu, Wing Kam and Gan, Zhengtao},
  journal={arXiv preprint arXiv:2111.03583},
  year={2021}
}
@article{gan2021universal,
  title={Universal scaling laws of keyhole stability and porosity in 3D printing of metals},
  author={Gan, Zhengtao and Kafka, Orion L and Parab, Niranjan and Zhao, Cang and Fang, Lichao and Heinonen, Olle and Sun, Tao and Liu, Wing Kam},
  journal={Nature communications},
  volume={12},
  number={1},
  pages={1--8},
  year={2021},
  publisher={Nature Publishing Group}
}
@article{saha2021hierarchical,
  title={Hierarchical Deep Learning Neural Network (HiDeNN): An artificial intelligence (AI) framework for computational science and engineering},
  author={Saha, Sourav and Gan, Zhengtao and Cheng, Lin and Gao, Jiaying and Kafka, Orion L and Xie, Xiaoyu and Li, Hengyang and Tajdari, Mahsa and Kim, H Alicia and Liu, Wing Kam},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={373},
  pages={113452},
  year={2021},
  publisher={Elsevier}
}
```

## Contact
If you have any questions or want to contribute to this respository, please contact: 
- Xiaoyu Xie
- Northwestern University, Mechanical Engineering
- xiaoyuxie2020@u.northwestern.edu
