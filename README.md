# Dimensionless learning for data-drievn knowledge discovery

<p align="center">
  <img width="200" height="200" src="images/logo.png">
</p>

## Identify knowledge in different levels

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
- **Better interpretability**: 
  - The identified dimensionless can be interpreted as the ratio of different forces, velocities, or energies, etc.
  - Lower dimension also allow for qualitative and quantitative analysis of the systems of interest. 
- **Works well in small dataset**: 
  - By incorporating **fundamental physical knowledge** of dimensional invariance, the learning space is limited to a manageable size, which makes it possible to train well-performing models using scarce datasets.
- **Better generalization**:
  - Another benefit for embedding physical invariance is that the learned model have a better generalization in data with different materials and scales.

## Where to find the paper

**Title: Data-driven discovery of dimensionless numbers and governing laws from scarce measurements.** 

This paper was published on ***Nature Communications***. You can find the paper [here](https://www.nature.com/articles/s41467-022-35084-w#Sec2).

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

1. Scaling law and dimenionless numbers dicsovery
Two typical examples for the scaling law and dimensionless numbers discovery can be found in [scaling_law/keyhole_example_pattern_search.py](https://github.com/xiaoyuxie-vico/PyDimension/blob/main/scaling_law/keyhole_example_pattern_search.py) and [scaling_law/keyhole_example_gradient_descent.ipynb](https://github.com/xiaoyuxie-vico/PyDimension/blob/main/scaling_law/keyhole_example_gradient_descent.ipynb). For the first file, you can run the code directly by using `python keyhole_example_pattern_search.py` after `cd` to this folder. For the second file, you can directly run the jupyter notebook.

2. Generalization comparison with popular machine learning algorithms is shown in [scaling_law/scaling_law/cross_materials.ipynb](https://github.com/xiaoyuxie-vico/PyDimension/blob/main/scaling_law/cross_materials.ipynb) and [scaling_law/cross_scales.ipynb](https://github.com/xiaoyuxie-vico/PyDimension/blob/main/scaling_law/cross_scales.ipynb).

3. A simplified version for pattern search-based two-level optimizaiton can be found at [scaling_law/utils/solver.py](https://github.com/xiaoyuxie-vico/PyDimension/blob/main/scaling_law/utils/solver.py).

4. Sensitive analysis for keyhole example can be found at [scaling_law/sensitive_analysis.ipynb](https://github.com/xiaoyuxie-vico/PyDimension/blob/main/scaling_law/sensitive_analysis.ipynb).

5. Discover the governing equations for spring-mass-damper systems: [discover_spring_clean.ipynb](https://github.com/xiaoyuxie-vico/PyDimension/blob/main/PDE_discovery/discover_spring_clean.ipynb).

5. More differential equation discovery examples are shown in [`PDE_discovery`](https://github.com/xiaoyuxie-vico/PyDimension/tree/main/PDE_discovery). 


## Dataset

The dataset for scaling law identification is at [here](https://github.com/xiaoyuxie-vico/PyDimension/tree/main/dataset).


## Citations

```
@article{xie_data-driven_2022,
  title = {Data-driven discovery of dimensionless numbers and governing laws from scarce   measurements},
  volume = {13},
  copyright = {2022 The Author(s)},
  issn = {2041-1723},
  url = {https://www.nature.com/articles/s41467-022-35084-w},
  doi = {10.1038/s41467-022-35084-w},
  language = {en},
  number = {1},
  urldate = {2022-12-08},
  journal = {Nature Communications},
  author = {Xie, Xiaoyu and Samaei, Arash and Guo, Jiachen and Liu, Wing Kam and Gan, Zhengtao},
  month = dec,
  year = {2022},
  pages = {7562},
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
