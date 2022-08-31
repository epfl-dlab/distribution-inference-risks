import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import propinfer as pia


class MLPEarlyStopping(pia.MLP):
    def __init__(self, label_col, hyperparams):
        super().__init__(label_col, hyperparams)
        self.early_stop =  hyperparams['early_stop']
    
    def fit(self, data):
        loader = self._prepare_dataloader(data, bs=self.bs, train=True, regression=self.n_classes == 1)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.MSELoss()

        for _ in range(self.epochs):
            sq_err = 0.
            numel = 0
            for X, y_true in loader:
                opt.zero_grad()
                y_pred = self.model(X)

                if y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()

                loss = criterion(y_pred, y_true)
                loss.backward()
                opt.step()
                
                sq_err += torch.sum((y_pred - y_true) ** 2).item()
                numel += len(y_true)
                
            mse = sq_err / numel
            if mse < self.early_stop:
                return self

        return self
    

class IRM(pia.MLP):
    def __init__(self, label_col, hyperparams):
        super().__init__(label_col, hyperparams)
        
        self.w = torch.Tensor((1.0,)).to(self.device)
        self.w.requires_grad = True

        self.reg = hyperparams['reg'] if 'reg' in hyperparams.keys() else 1e-5
        
        self.env = hyperparams['env_label']
        
    def _prepare_dataloader(self, df, bs=32, train=True, regression=False):
        envs = list()
        regression = self.n_classes == 1
        
        for _, env in df.groupby(self.env):
            env = env.drop(self.env, axis=1)
            X, y = self._prepare_data(env, train=False)
            X = torch.tensor(X.values.astype(np.float32), device=self.device)
            y = torch.tensor(y.values.astype(np.int64 if not regression else np.float32), device=self.device).view(-1, 1)
            
            envs.append((X, y))

        return envs
    
    def __compute_penalty(self, losses):
        g1 = torch.autograd.grad(losses[0::2].mean(), self.w, create_graph = True)[0]
        g2 = torch.autograd.grad(losses[1::2].mean(), self.w, create_graph = True)[0]
        return (g1 * g2).sum()

    def fit(self, data):
        environments = self._prepare_dataloader(data)

        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.MSELoss(reduction='none')

        for epoch in range(self.epochs):
            penalty = 0
            error = 0
            
            for X, y_true in environments:
                p = torch.randperm(len(X))
                error_e = criterion(self.model(X[p]) * self.w, y_true[p])
                penalty += self.__compute_penalty(error_e)
                error += error_e.mean()

            opt.zero_grad()
            (self.reg * error + penalty).backward()
            opt.step()

        return self

    def predict_proba(self, data):
        X, _ = self._prepare_data(data, train=False)
        if 'env' in X.columns:
            X = X.drop('env', axis=1)
        X = torch.tensor(X.values.astype(np.float32), device=self.device)

        return np.nan_to_num((self.model(X) * self.w).detach().cpu().numpy())