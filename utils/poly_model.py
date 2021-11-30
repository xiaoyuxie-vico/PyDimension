# -*- coding: utf-8 -*-
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

is_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if is_cuda else 'cpu')


class PolyDataset(Dataset):
    def __init__(self, X, y, ord=3):
        super(PolyDataset, self).__init__()
        self.X_poly = np.concatenate([X ** i for i in range(1, ord+1)], 1)
        self.y = y
    
    def __len__(self):
        return self.X_poly.shape[0]

    def __getitem__(self, idx):
        return self.X_poly[idx, :], self.y[idx]


class polyModel(nn.Module):
    def __init__(self, order=3):
        super(polyModel, self).__init__()
        self.poly = nn.Linear(order, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


def fit_ploy_model(X, y, lr=1, epoch_num=50, batch_size=64, display_num=5):
    '''fetch the R2 metric for poly model
    '''
    dataset = PolyDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    net = polyModel()

    # train
    loss_history = []
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=epoch_num, gamma=0.3)
    loss_fn = nn.MSELoss()
    for epoch_idx in range(epoch_num):
        for step, (inputs, targets) in enumerate(train_loader):
            net.train()
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = net(inputs)
            loss = loss_fn(targets, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        scheduler.step()

        net.eval()
        if epoch_idx % display_num == 0:
            # print(f'Epoch_idx: {epoch_idx}, Loss: {loss.item()}')
            inputs_all, targets_all, preds_all = [], [], []
            for step, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()
                outputs = net(inputs)

                inputs_all.extend(inputs[:, 0].detach().numpy().tolist())
                targets_all.extend(targets.detach().numpy().tolist())
                preds_all.extend(outputs.detach().numpy().tolist())

            r2_score = metrics.r2_score(targets_all, preds_all)
            mse = metrics.mean_squared_error(targets_all, preds_all)

            fig = plt.figure()
            plt.scatter(inputs_all, targets_all, label='Ground truth')
            plt.scatter(inputs_all, preds_all, label='Prediction')
            plt.legend()
            plt.title(round(r2_score, 4))
            plt.savefig(f'../results/{epoch_idx}.jpg', dpi=300)

    return r2_score, mse

if __name__ == '__main__':
    W_target = np.array([0.5, 3, 2.4]).reshape(-1, 1)
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    X_poly = np.concatenate([X ** i for i in range(1, 4)], 1)

    y = X_poly @ W_target + 0.1
    y_noise = 1 * np.random.random(size=(100, 1))

    r2_score, mse = fit_ploy_model(X, y)
    print(f'Without noise: r2_score: {r2_score}, mse: {mse}')

    r2_score, mse = fit_ploy_model(X, y+y_noise)
    print(f'With noise: r2_score: {r2_score}, mse: {mse}')
