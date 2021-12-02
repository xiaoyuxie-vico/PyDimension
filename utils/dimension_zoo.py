# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jan. 28th 2021
'''

import numpy as np

class DimensionZoo(object):
    '''Store the dimension for input and output
    '''

    def __init__(self):
        super(DimensionZoo, self).__init__()

    def fetch_dim(self, problem_type):
        if problem_type == 'rb':
            input_dim_mat = np.array(
                [
                    [1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
                    [0.0, 0.0, -3.0, -2.0, 0.0, -1.0, -1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                ],
            )
            output_dim_mat = np.array(
                [ 
                    [0.],
                    [-3.], 
                    [1.], 
                    [0.],
                ]
            )
            # best weights for Ra: 1, 1, 1
            scaling_mat = np.array(
                [
                    [0., 0., 3.],
                    [0., 1., 0.],
                    [0., 0., 0.],
                    [0., 0., 1.],
                    [0., 1., 0.],
                    [1., 0., -2.],
                    [-1., 0., 0.],
                ]
            )
            dnn_mat = np.array([1., 1., -1., -1., 0., 0., 0., 0.]).reshape(-1, 1)
        elif problem_type == 're':
            input_dim_mat = np.array(
                [
                    [1.0, 2.0, 1.0, 1.0],
                    [-1.0, -1.0, 0.0, 0.0],
                ],
            )
            output_dim_mat = np.array(
                [
                    [0.],
                    [0.],
                ]
            )
            scaling_mat = np.array(
                [
                    [0., 1.],
                    [0., -1.],
                    [1., 0.],
                    [-1., 1.]
                ]
            )
            dnn_mat = np.array(
                [1., 0., 0., 0., 0]).reshape(-1, 1)
        elif problem_type == 'keyhole':
            input_dim_mat = np.array(
                [
                    [2., 1., 1., 2., -3., 2., 0.],
                    [-3., -1., 0., -1., 0., -2., 0.],
                    [1., 0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0., -1., 1.],
                ],
            )
            output_dim_mat = np.array(
                [
                    [1.],
                    [0.],
                    [0.],
                    [0.],
                ]
            )
            # best weights for Ke: 0.5, 1, 1
            scaling_mat = np.array(
                [
                    [0., 0., 1],
                    [1, 2., -3],
                    [1, 0., -2.],
                    [-1, 0., 0.],
                    [0., 0., -1],
                    [0., -1, 0.],
                    [0., -1, 0.]],
            )

            dnn_mat = np.array(
                [1., 0., 0., -1., 0., 0., 0., 0.]).reshape(-1, 1)
        elif problem_type == 'keyhole_add':
            input_dim_mat = np.array(
                [
                    [2., 1., 1., 2., -3., 2., 0., 0.],
                    [-3., -1., 0., -1., 0., -2., 0., 0.],
                    [1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., -1., 1., 1.],
                ],
            )
            output_dim_mat = np.array(
                [
                    [1.],
                    [0.],
                    [0.],
                    [0.],
                ]
            )
            # best weights for Ke: 1, 1.25, -1, 2
            scaling_mat = np.array(
                [
                    [1.,   0.,   0.,   0.   ],
                    [0.,   -2.,  0.,   1.   ],
                    [1.,   -2.,  0.,   0.   ],
                    [-3.,  2.,   0.,   0.   ],
                    [-1.,  0.,   0.,   0.   ],
                    [0.,   0.,   0.,   -0.5 ],
                    [0.,   0.,   1.,   0.   ],
                    [0.,   0.,   -1.,  -0.5 ],
                ]
            )

            dnn_mat = np.array(
                [1., 0., 0., -1., 0., 0., 0., 0., 0.]).reshape(-1, 1)
        elif problem_type == 'keyhole_add2':
            input_dim_mat = np.array(
                [
                    [2., 1., 1., 2., -3., 2., 0., 2.],
                    [-3., -1., 0., -1., 0., -2., 0., -2.],
                    [1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., -1., 1., 0.],
                ],
            )
            output_dim_mat = np.array(
                [
                    [1.],
                    [0.],
                    [0.],
                    [0.],
                ]
            )
            # best weights for Ke: -1, 0.25, 0.25, -0.5
            scaling_mat = np.array(
                [
                    [-1.,  -0.,  -0.,  -0. ],
                    [ 0.,   0.,   0.,   1. ],
                    [ 2.,   0.,   2.,   0. ],
                    [ 0.,   0.,  -2.,   0. ],
                    [ 1.,   0.,   0.,   0. ],
                    [ 1.5,  1.,   0.,  -0.5],
                    [ 1.5,  1.,   0.,  -0.5],
                    [ 0.,  -1.,   1.,   0. ],
                ]
            )

            dnn_mat = np.array(
                [1., 0., 0., -1., 0., 0., 0., 0., 0.]).reshape(-1, 1)
        elif problem_type == 'keyhole_remove':
            input_dim_mat = np.array(
                [
                    [2., 1., 1., -3., 2., 0.],
                    [-3., -1., 0., 0., -2., 0.],
                    [1., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., -1., 1.],
                ],
            )
            output_dim_mat = np.array(
                [
                    [1.],
                    [0.],
                    [0.],
                    [0.],
                ]
            )
            scaling_mat = np.array(
                [
                    [ 0.,  1.],
                    [-2., -3.],
                    [ 0., -2.],
                    [ 0., -1.],
                    [ 1., -0.],
                    [ 1.,  0.],
                ]
            )

            dnn_mat = np.array(
                [1., 0., 0., -1., 0., 0., 0.]).reshape(-1, 1)
        elif problem_type == 'porosity':
            input_dim_mat = np.array(
                [
                    [2.,   1.,  1.,  -3.,   2.,   2.,   0.,  1.,  1.],
                    [-3., -1.,  0.,   0.,  -2.,  -1.,   0.,  0.,  0.],
                    [1.,   0.,  0.,   1.,   0.,   0.,   0.,  0.,  0.],
                    [0.,   0.,  0.,   0.,  -1.,   0.,   1.,  0.,  0.],
                ],
            )
            output_dim_mat = np.array(
                [
                    [0.],
                    [0.],
                    [0.],
                    [0.],
                ]
            )
            # best weights for NED: 1, 1, -1, 0, 1
            # best weights for Ke_𝑚 𝐿_𝑑^∗: 1, 0, -2, 0, 1
            scaling_mat = np.array(
                [
                    [0.,   0.,    0.,   0.,    1.],
                    [2.,  0.,    0.,   1.,   -3.],
                    [0.,   1.,    1.,   0.,    0.],
                    [0.,   0.,    0.,   0.,   -1.],
                    [-1.,   0.,    0.,   0.5,   0.],
                    [0.,   0.,    0.,   0.,    0.],
                    [-1.,   0.,    0.,   0.5,   0.],
                    [0.,   -1.,   0.,   0.,    0.],
                    [0.,   0.,   -1.,   0.,   -2.],
                ]
            )

            dnn_mat = np.array(
                [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(-1, 1)
        elif problem_type == 'oscillation':
            input_dim_mat = np.array(
                [
                    [0., 0., 1., 1., 1.],
                    [1., 0., 0., 0., 0.],
                    [0., 1., 0., -2., -1.],
                ],
            )
            output_dim_mat = np.array(
                [
                    [0.],
                    [1.],
                    [0.],
                ]
            )

            # best weights: 0.5, -0.5: t'=t/sqrt(m/k)
            # best weights: 0.5, 0.5: c'=c/sqrt(mk)
            scaling_mat = np.array(
                [
                    [0., 0.],
                    [1., -1.],
                    [0., 1.],
                    [1., 0.],
                    [-1., -1.],
                ]
            )

            dnn_mat = np.array(
                [1., -1., 0., 0., 0., 0.]).reshape(-1, 1)

        return input_dim_mat, output_dim_mat, scaling_mat, dnn_mat


def main():
    dimension_zoo = DimensionZoo()
    print(dimension_zoo.fetch_dim('rb')[0])
    print(dimension_zoo.fetch_dim('rb')[1])

if __name__ == "__main__":
    main()
