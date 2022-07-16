# Dimensionless learning for scientific knowledge discovery (dimensionless numbers, scaling laws, and PDEs)

## What is dimensionless learning

The proposed dimensionless learning is a powerful technique to identify scientific knowledge from data at multiple levels: 

- **Dimensionless number** at the feature level (physical dimension reduction technique): 
  - **Example**: A well-known dimensionless number can be identified, called the Reynolds number $\mathrm{Re}=\frac{\rho V l}{\mu}$. It is the ratio of inertial force to viscous force in fluid flow and is highly used to distinguish between laminar or turbulent flow.
- **Scaling law** at the algebraic equation level
  - **Example**: The complex vapor depression dynamics in 3D printing can be represented as a simple algebraic equation: $e^*=0.12\mathrm{Ke}-0.30$.
- **Governing equation** at the differential equation level
  - **Example**: The well-known Navier-Stokes equation $\frac{\partial \omega}{\partial t}+u\frac{\partial \omega}{\partial x}+v\frac{\partial \omega}{\partial y}=\mathrm{\frac{1}{Re}}(\frac{\partial^2 \omega}{\partial x^2} +\frac{\partial^2 \omega}{\partial y^2})$ can be identified from data by integrating dimensionless learning with SINDy.

## Advantages

- **Dimension reduction**: 
  - Dimensionless leanring can reduce a large number of parameters to a few dominant dimensionless numbers.
- **Better explainability**: 
  - The identified dimensionless can be interpreted as the ratio of different forces, velocities, or energies, etc.
  - Lower dimension also allow for qualitative and quantitative analysis of the systems of interest. 
- **Works well in small dataset**: 
  - By incorporating **fundamental physical knowledge** of dimensional invariance, the learning space is limited to a manageable size, which makes it possible to train well-performing models using scarce datasets.
- **Better generalization**:
  - Another benefit for embedding physical invariance is that the learned model have a better generalization in data with different materials and scales.

## Where to find the paper

**Title: Data-driven discovery of dimensionless numbers and governing laws from scarce measurements.** 

You can find the preprint paper at [Arxiv](http://arxiv.org/abs/2111.03583) or [ResearchSquare](https://assets.researchsquare.com/files/rs-1122326/v1_covered.pdf?c=1639152750).

## Requirements
```
matplotlib==3.1.3
derivative==0.3.1
pandas==1.3.4
pyyaml==6.0
scikit-learn==0.24.2
pysindy==1.3.0
```

## Getting started

Go to the folder called `scaling_law` and `PDE_discovery` and run the jupyter notebook.

Two typical examples for the scaling law and dimensionless numbers discovery can be found in `scaling_law/keyhole_example_pattern_search.py` and `scaling_law/keyhole_example_gradient_descent.ipynb`. You can run it directly by using `python keyhole_example_pattern_search.py` after `cd` to this folder. 

## Dataset

The dataset for scaling law identification is at [here](https://github.com/xiaoyuxie-vico/PyDimension/tree/main/dataset).


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
