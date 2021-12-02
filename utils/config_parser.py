# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jan. 28th 2021
'''

import yaml


class ConfigParser():
    '''Config Reader'''
    
    def __init__(self, config_path):
        # load
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        # config
        self.degree = config['degree']
        self.output_index = config['output_index']
        self.hidden_num = config['hidden_num']
        self.dnn_mat_file = config['dnn_mat_file']
        self.reverse_input = config['reverse_input']
        self.reverse_output = config['reverse_output']
        self.fix_scaling_mat = config['fix_scaling_mat']
        self.fix_dnn_mat = config['fix_dnn_mat']
        self.problem_type = config['problem_type']
        self.dataset_file = config['dataset_file']
        self.epoch_num = config['params']['epoch_num']
        # self.stepwise = config['params']['stepwise']
        self.lr_initial = config['params']['lr_initial']
        self.display_num = config['params']['display_num']
        self.pretrained_path = config['params']['pretrained_path']
        self.batch_size = config['params']['batch_size']
        self.dnn_mat_num = config['dnn_mat_num']
        self.split_seed = config['split_seed']
        self.metric_list = config['metric_list']
        self.model_name = config['model_name']
        self.random_split_num = config['random_split_num']
        self.dimless_num = config['dimless_num']
        self.input_labels = config['input_labels']
        self.output_labels = config['output_labels']    
        self.tag = self.output_labels[0]
        self.is_normlize_output = config['is_normlize_output']
        self.is_loglize_output = config['is_loglize_output']
        self.is_loglize_input = config['is_loglize_input']
        self.base_vec_num = 0
        
        
def test():
    '''test'''
    config_path = '../configs/config_re_v2.yml'
    config = ConfigParser(config_path)
    print(f'config.dataset_file: {config.dataset_file}')

    import torch
    import torch.nn as nn
    weight_fc0 = torch.Tensor(eval(config.weight_fc0))
    print(nn.Parameter(weight_fc0, requires_grad=True))

if __name__ == "__main__":
    test()
