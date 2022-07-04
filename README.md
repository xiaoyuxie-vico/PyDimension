# Dimensionless learning for scientific knowledge discovery (dimensionless numbers, scaling laws, and PDEs)

## What is dimensionless learning

In short, dimensionless learning is a data-driven dimensional analysis method.

Compared with standard machine learning methods, dimensionless learning embeds a physical constraint, dimensional invariance in different variables, into machine learning algorithms, which means we restrict the large possible solution space into a smaller one. That is to say, we change the function set of machine leanring model and make it is feasible to be trained even for a small number of dataset (about 100-200 data points).

Compared with classical dimensional analysis which only can know the required number of dimensionless numbers for a complex problem, dimensionless learning provides a systematic way to identify **unique and dominant** dimensionless numbers.

## Where to find the paper

**Title: Data-driven discovery of dimensionless numbers and governing laws from scarce measurements.** 

Summary:
In this study, we propose a mechanistic data-driven approach, called dimensionless learning. This method consists of two main workflows to discover scientific knowledge from data. The first workflow embeds the principle of dimensional invariance (i.e., physical laws are independent on an arbitrary choice of basic units of measurements) into a two-level machine learning scheme to automatically discover dominant dimensionless numbers and scaling laws from noisy experimental measurements of complex physical systems. This invariance incentivizes the learning of scale-invariant and physically interpretable low-dimensional patterns of complex high-dimensional systems. We demonstrate the first workflow by solving three challenging problems in science and engineering with noisy experimental measurements collected from the literature. The problems include turbulent Rayleigh-Benard convection, vapor depression dynamics, and porosity formation during 3D printing. In the second workflow, the dimensionless learning is integrated with sparsity-promoting techniques (such as SINDy and proposed symmetric invariant SINDy) to identify dimensionally homogeneous differential equations and dimensionless numbers from data. The analyses are performed on five differential equations with and without noisy data effect, including Navier-Stokes, Euler, vorticity equations, the governing equations for spring-mass-damper systems and dynamic loading beam systems.

You can find the preprint paper at [here](http://arxiv.org/abs/2111.03583).

## Paper highlights

- Automatically discover unique and dominant **dimensionless numbers** with clear physical meaning and **scaling laws** from complex systems based on experimental measurements;
- Identify **dimensionally homogeneous differential equations** with minimal parameters by leveraging sparsity-promoting techniques (SINDy).

# Requirements
```
matplotlib==3.1.3
derivative==0.3.1
pandas==1.3.4
pyyaml==6.0
scikit-learn==0.24.2
pysindy==1.3.0
```

# Tutorials and examples

Go to the folder called `scaling_law` and `PDE_discovery` and run the jupyter notebook.

We will update other examples and simplify the codes in this repository.

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


# Citations

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

# Contact
If you have any questions or want to contribute to thsi respository, please contact: 
- Xiaoyu Xie
- Northwestern University, Mechanical Engineering
- xiaoyuxie2020@u.northwestern.edu
