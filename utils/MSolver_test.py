# -*- coding: utf-8 -*-

import random
import numpy as np
import traceback

np.set_printoptions(suppress=True)

class MSolver(object):
    '''
    Matrix solver to find the solution
    '''

    def __init__(self):
        super(MSolver, self).__init__()
    
    def parse_index(self):
        '''pre set some values for the solution, find the indexs
        '''
        index_all = list(range(int(self.A.shape[1])))
        random.shuffle(index_all)
        index_set = index_all[:self.sol_num]
        index_free = list(set(range(self.A.shape[1]))-set(index_set))
        return index_set, index_free

    def init_sol(self, index_set):
        '''pre set some values for the solution
        '''
        solution_temp = np.zeros((self.A.shape[1], 1))
        solution_temp[index_set, :] = np.array(
            [1]+[0]*(self.sol_num-1)).reshape(-1, 1)
        return solution_temp

    def calc_b_new(self, solution_temp):
        '''calculate b_new by substracting known solution part
        '''
        b = np.zeros((self.A.shape[0], 1))
        b_new = b - np.dot(self.A, solution_temp)
        return b_new

    def unique_colmun(self, arr):
        '''Only same unique columns for a matrix
        '''
        return np.unique(arr, axis=1)

    def check_max_num(self, arr, threshold=100):
        '''check for the maximum value
        '''
        max_num = np.sum(np.abs(arr) >= threshold)
        return max_num

    def check_non_int(self, arr):
        '''check for integer and allow 0.5 or 1.5 etc.
        '''
        arr = np.where(
            np.abs(arr) < 0.01, 0, arr)
        mask_temp = np.mod(arr, 0.5)
        mask = (mask_temp != 0)
        non_integer_num = np.sum(mask)
        return non_integer_num

    @staticmethod
    def combine_cols(arr):
        '''combine columns with random weights
        '''
        random_weights = np.random.randint(0, 3, (arr.shape[1], 1))
        arr_new = np.dot(arr, random_weights)
        return arr_new

    @staticmethod
    def check_sol_exits(solution_part, solution_final, index_free):
        '''check whether the solution_part already exists in the solution_final
        '''
        is_sol_exist = False
        for i in range(solution_final.shape[1]):
            if (solution_final[index_free, i] == solution_part.reshape(-1,)).all():
                is_sol_exist = True
        return is_sol_exist

    def solver(self, A, maximum=5, maxi_iter=100):
        '''Solve Ax=0
        '''
        self.A = A

        # matrix rank
        self.rank = np.linalg.matrix_rank(self.A)
        self.sol_num = self.A.shape[1] - self.rank

        used_indexs = []
        solution_final = np.zeros((self.A.shape[1], self.sol_num))
        for i in range(self.sol_num):
            idx = 0
            flag = False
            # solve Ax=0
            while not flag and idx < maxi_iter:
                index_set, index_free = self.parse_index()

                solution_temp = self.init_sol(index_set)

                b_new = self.calc_b_new(solution_temp)
                A_new = A[:, index_free]
                rank_A_new = np.linalg.matrix_rank(A_new)
                # print('rank_A_new', rank_A_new)

                # only solve for rank(A_new) = A.shape[0]
                if rank_A_new != self.A.shape[0]:
                    continue

                solution = self.init_sol(index_set)
                try:
                    # solve A_new*x=b_new
                    solution_part = np.linalg.solve(A_new, b_new)

                    # check for maximum value
                    max_num = self.check_max_num(solution_part)
                    # check for integer and allow 0.5 or 1.5 etc.
                    non_integer_num = self.check_non_int(solution_part)

                    # check if solution_part has been solved before
                    is_sol_exits = self.check_sol_exits(
                        solution_part, solution_final, index_free)

                    if is_sol_exits:
                        print('*'*20)
                        print('solution_part', solution_part)
                        print('solution_final', solution_final)

                    if max_num == 0 and non_integer_num == 0 and not is_sol_exits:
                        flag = True
                        solution[index_free, :] = solution_part
                except:
                    # print('non inverse')
                    pass
                idx += 1

            solution_final[:, i] = solution.reshape(-1,)

        # remove duplicate columns
        solution_final = self.unique_colmun(solution_final)

        # print('solution_final', solution_final)
        return solution_final
    
def parse_dnn_mat(A_all, output_index):
    '''solve base vector for dnn mat and combine vectors randomly
    '''
    m_solver = MSolver()
    combined_sol = np.zeros((A_all.shape[1], 1))
    while combined_sol[output_index] == 0:
        res = m_solver.solver(A_all)
        # print('final solution: \n', res)
        # print(A_all @ res)
        # print('combined solution:')
        combined_sol = m_solver.combine_cols(res)
    return combined_sol

def main():
    # # Re example
    # A = np.array([
    #     [1.0, 2.0, 1.0, 1.0],
    #     [-1.0, -1.0, 0.0, 0.0]
    # ])
    # b = np.array([
    #     [0],
    #     [0]
    # ])

    # Rb example
    A = np.array(
        [
            [1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, -3.0, -2.0, 0.0, -1.0, -1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        ]
    )

    # b = np.zeros((4, 1))
    b = np.array(
        [
            [0.],
            [-3.],
            [1.],
            [0.],
        ]
    )

    # b_zeros = np.zeros_like(b)
    m_solver = MSolver()

    print('*'*20+'A'+'*'*20)
    res = m_solver.solver(A)
    print('final solution: \n', res)
    # print(A @ res)

    print('*'*20+'A_new'+'*'*20)
    A_all = np.hstack([b, A])
    parse_dnn_mat(A_all, output_index=0)

if __name__ == "__main__":
    main()
