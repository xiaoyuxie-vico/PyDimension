# Implementation for dimensionless learning, a data-driven dimensional analysis for scientific problems

**Title: Data-driven discovery of dimensionless numbers and scaling laws from experimental measurements.** 

You can find the preprint paper at [here](http://arxiv.org/abs/2111.03583).

**Paper highlights:**

- Automatically discover unique and dominant **dimensionless numbers** with clear physical meaning and **scaling laws** from complex systems based on experimental measurements;
- Identify **dimensionally homogeneous differential equations** with minimal parameters by leveraging sparsity-promoting techniques (SINDy).

# Requirements
```
python==3.9.7
matplotlib==3.1.3
derivative==0.3.1
pandas==1.3.4
pyyaml==6.0
scikit-learn==0.24.2
pysindy==1.3.0
```

## 1. Local version

You can install these packages with by creating an virtual environment via Anaconda:

`conda create --name PyDimension --file requirements.txt`

Activate the virtual environment:

`conda activate PyDimension `

## 2. Online version - Binder

**Note that you can also use the online version code in `tutorials` folder by simply clicking the icon:**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/xiaoyuxie-vico/PyDimension/main)

## 3. Online version - Colab

**You can run the code on Colab by clicking the icon:**

[![launch](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1SelA0bDnlux4Gj597YQKEPgtfU2bI2sg/view?usp=sharing)

# Tutorials and examples

Go to the folder called `tutorials` and run the jupyter notebook.

- `discover_pde_from_data.ipynb`: discover dimensionless numbers and dimensionally homogeneous differential equations in spring-mass-damping systems.

- `keyhole_example.ipynb`: discover dimensionless numbers and scaling laws based on experimental measurements.

We will update other examples and simplify the codes in this repository.

# Code structure

```shell
.
├── LICENSE
├── README.md
├── __init__.py
├── configs
│   ├── __init__.py
│   └── config_oscillation.yml						# configuration for spring-mass-damping example
├── dataset
│   ├── dataset_oscillation.csv						# dataset for spring-mass-damping example
│   └── keyhole_data.csv									# dataset for keyhole dynamics
├── models
├── requirements.txt											# required python packages
├── tutorials
│   ├── __init__.py
│   ├── discover_pde_from_data.ipynb			# example for spring-mass-damping
│   └── keyhole_example.ipynb							# example for keyhole dynamics
└── utils
    ├── __init__.py
    ├── BIC.py								# calculate BIC to select the best model
    ├── MSolver.py						# calculate basis vectors
    ├── config_parser.py			# configer to parse configuration
    ├── dimension_zoo.py			# store dimension matrix and others matrix information
    ├── gen_pde_dataset.py		# generate spring-mass-damping dataset based on the analytical solution
    └── tools.py							# some tools
```



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
