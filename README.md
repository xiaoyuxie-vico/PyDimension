# Implementation for dimensionless learning, a data-driven dimensional analysis for scientific problems

**Title: Data-driven discovery of dimensionless numbers and scaling laws from experimental measurements.** 
[Paper link](http://arxiv.org/abs/2111.03583)

**Abstract**: Dimensionless numbers and scaling laws provide elegant insights into the characteristic properties of physical systems. Classical dimensional analysis and similitude theory fail to identify a set of unique dimensionless numbers for a highly-multivariable system with incomplete governing equations. In this study, we embed the principle of dimensional invariance into a two-level machine learning scheme to automatically discover dominant and unique dimensionless numbers and scaling laws from data. The proposed methodology, called dimensionless learning, can reduce high-dimensional parametric spaces into descriptions involving just a few physically-interpretable dimensionless parameters, which signiﬁcantly simpliﬁes the process design and optimization of the system. We demonstrate the algorithm by solving several challenging engineering problems with noisy experimental measurements (not synthetic data) collected from the literature. The examples include turbulent Rayleigh-Bénard convection, vapor depression dynamics in laser melting of metals, and porosity formation in 3D printing. We also show that the proposed approach can identify dimensionally-homogeneous differential equations with minimal parameters by leveraging sparsity-promoting techniques.

# Usage

Go to the folder called `tutorials` and run the jupyter notebook.

- `discover_pde_from_data.ipynb`: discover dimensionless numbers and dimensionally homogeneous differential equations in spring-mass-damping systems.
- `keyhole_example.ipynb`: discover dimensionless numbers and scaling laws based on experimental measurements.

We will update other examples and simplify the codes in this repository.

# Requirements
- Python 3
- Pytorch 1.5.1

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
- xiaoyuxie2020@u.northwestern.edu
