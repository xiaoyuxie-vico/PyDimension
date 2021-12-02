# -*- coding: utf-8 -*-

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.model import DimensionNet
matplotlib.use('Agg')

is_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if is_cuda else 'cpu')

matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['font.family'] = 'Arial'


def predict(net, data_loader, dnn_mat, reverse_output=False):
    '''Predict data and return a numpy results'''
    net.eval()
    y_pred, y_true = [], []
    for step, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)

        inputs_all_1 = torch.cat((targets.reshape(-1, 1), inputs), dim=1)
        target_pi = torch.mm(
            inputs_all_1, torch.Tensor(dnn_mat).to(device))

        if reverse_output:
            target_pi = 10**target_pi

        if is_cuda:
            outputs_arr = outputs.detach().cpu().numpy().tolist()
            targets_arr = target_pi.detach().cpu().numpy().tolist()
        else:
            outputs_arr = outputs.detach().numpy().tolist()
            targets_arr = target_pi.detach().numpy().tolist()
        y_pred.extend(outputs_arr)
        y_true.extend(targets_arr)

    return np.array(y_true), np.array(y_pred)


def predict_pi(net, data_loader, dnn_mat, scaling_mm, reverse_input=False, reverse_output=False):
    '''Predict data and return a numpy results'''
    net.eval()
    pi_list = []
    y_pred, y_true = [], []
    for step, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = net(inputs)

        pi_1 = torch.mm(inputs, torch.Tensor(scaling_mm).to(device)).detach()
        if reverse_input:
            pi_1 = 10**pi_1
        pi_list.extend(pi_1.numpy().tolist())

        inputs_all_1 = torch.cat((targets.reshape(-1, 1), inputs), dim=1)
        target_pi = torch.mm(
            inputs_all_1, torch.Tensor(dnn_mat).to(device))
        
        if reverse_output:
            target_pi = 10**target_pi

        if is_cuda:
            pred_arr = preds.detach().cpu().numpy().tolist()
            true_arr = target_pi.detach().cpu().numpy().tolist()
        else:
            pred_arr = preds.detach().numpy().tolist()
            true_arr = target_pi.detach().numpy().tolist()
        
        y_pred.extend(pred_arr)
        y_true.extend(true_arr)

    return np.array(y_true), np.array(y_pred), np.array(pi_list)


def predict_v2(net, data_loader):
    '''Predict data and return a numpy results'''
    net.eval()
    y_pred, y_true = [], []
    for step, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)

        if is_cuda:
            outputs_arr = outputs.detach().cpu().numpy().tolist()
            targets_arr = targets.detach().cpu().numpy().tolist()
        else:
            outputs_arr = outputs.detach().numpy().tolist()
            targets_arr = targets.detach().numpy().tolist()
        y_pred.extend(outputs_arr)
        y_true.extend(targets_arr)

    return np.array(y_true), np.array(y_pred)


def predict_pi_v2(net, data_loader, scaling_mm):
    '''Predict data and return a numpy results'''
    net.eval()
    pi_list = []
    y_pred, y_true = [], []
    for step, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = net(inputs)

        pi_1 = torch.mm(inputs, torch.Tensor(scaling_mm).to(device)).detach()
        pi_list.extend(pi_1.numpy().tolist())

        if is_cuda:
            pred_arr = preds.detach().cpu().numpy().tolist()
            true_arr = targets.detach().cpu().numpy().tolist()
        else:
            pred_arr = preds.detach().numpy().tolist()
            true_arr = targets.detach().numpy().tolist()

        y_pred.extend(pred_arr)
        y_true.extend(true_arr)

    return np.array(y_true), np.array(y_pred), np.array(pi_list)


def plot_prediction(true_train, pred_train, true_test, pred_test, fig_param):
    '''plot prediction and targets'''
    fig = plt.figure(figsize=(9, 4))
    fig.add_subplot(1, 2, 1)
    plt.plot(pred_train, true_train, 'o', color='b', label='Training set')
    plt.xlabel(fig_param['xlabel'][0], fontsize=fig_param['xlabel'][1])
    plt.ylabel(fig_param['ylabel'][0], fontsize=fig_param['ylabel'][1])

    x = np.linspace(np.min(pred_train), np.max(pred_train), 100)
    y = x
    plt.plot(x, y, c='black')

    plt.legend(fontsize=fig_param['legend'])


    fig.add_subplot(1, 2, 2)
    plt.plot(pred_test, true_test, 'o', color='r', label='Test set')
    plt.xlabel(fig_param['xlabel'][0], fontsize=fig_param['xlabel'][1])
    plt.ylabel(fig_param['ylabel'][0], fontsize=fig_param['ylabel'][1])

    x = np.linspace(np.min(pred_train), np.max(pred_train), 100)
    y = x
    plt.plot(x, y, c='black')

    plt.legend(fontsize=fig_param['legend'])
    plt.tight_layout()
    plt.savefig(fig_param['save_path'], dpi=300)

def calc_mre(y_true, y_pred):
    '''
    calculate mean_relative_error
    
    # example
    y_true = np.array([1, 2, 3]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('np.absolute(y_pred - y_true)', np.absolute(y_pred - y_true))
    print('np.absolute((y_pred - y_true) / y_true)', np.absolute((y_pred - y_true) / y_true))
    print(calc_mre(y_true, y_pred))
    '''
    mre = np.mean(np.absolute((y_pred - y_true) / (y_true + 1e-10)))
    return mre

def calc_metrics(y_true, y_pred, source, metric_list):
    '''
    calculate the results of different metrics with val
    '''
    metric_res = {}
    for metric_name in metric_list:
        if metric_name == 'mean_relative_error':
            result_train = calc_mre(y_true, y_pred)
        else:
            result_train = eval(
                'metrics.{}(y_true, y_pred)'.format(metric_name))
        metric_res[metric_name+'_'+source] = round(float(result_train), 4)
    return metric_res


def set_weight(arr, requires_grad):
    '''set weights with certain array
    '''
    weight_tr = torch.Tensor(arr)
    weight_tr = nn.Parameter(weight_tr, requires_grad=requires_grad)
    return weight_tr


def load_model(net, config, pretrained_path=None):
    '''initialize weights
    '''
    print('Load pretrained model')
    # load pretrained model
    pretrained_model = DimensionNet(
        input_dim=len(config.input_labels),
        dimless_num=config.dimless_num,
        base_vec_num=config.base_vec_num,
        lr=config.lr_initial,
        hidden_num=config.hidden_num
    )
    if not pretrained_path:
        pretrained_model.load_state_dict(torch.load(config.pretrained_path, map_location='cpu'))
    else:
        pretrained_model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    pretrained_dict = pretrained_model.state_dict()
    net.load_state_dict(pretrained_dict)
    return net


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    # r2 = 1 - ss_res / ss_tot
    residual = ss_res / (ss_tot + 1e-6)
    return residual


def calc_mre(y_true, y_pred):
    '''
    calculate mean_relative_error
    
    # example
    y_true = np.array([1, 2, 3]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('np.absolute(y_pred - y_true)', np.absolute(y_pred - y_true))
    print('np.absolute((y_pred - y_true) / y_true)', np.absolute((y_pred - y_true) / y_true))
    print(calc_mre(y_true, y_pred))
    '''
    # mre = np.mean(np.absolute((y_pred - y_true) / (y_true + 1e-10)))
    normalizer = np.mean(np.sqrt(y_true**2))
    # diff = torch.norm()
    # normalizer = torch.norm(y_true, 1)
    mre = np.mean(np.absolute(y_pred - y_true)) / (normalizer + 1e-5)
    return mre


def mre_loss(y_true, y_pred):
    '''
    calculate mean_relative_error
    
    # example
    y_true = np.array([1, 2, 3]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('np.absolute(y_pred - y_true)', np.absolute(y_pred - y_true))
    print('np.absolute((y_pred - y_true) / y_true)', np.absolute((y_pred - y_true) / y_true))
    print(calc_mre(y_true, y_pred))
    '''
    # mre = torch.mean(torch.sqrt(((y_pred - y_true) / (y_true + 1e-10))**2))
    normalizer = torch.mean(torch.sqrt(y_true**2))
    mre = torch.mean(torch.abs(y_pred - y_true)) / (normalizer + 1e-5)
    # print('mre', torch.mean(torch.abs(y_pred - y_true)), normalizer)
    return mre

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    # test calc_mre
    y_true = np.array([1, 2, 3]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('np.absolute(y_pred - y_true)', np.absolute(y_pred - y_true))
    print('np.absolute((y_pred - y_true) / y_true)', np.absolute((y_pred - y_true) / y_true))
    print(calc_mre(y_true, y_pred))
