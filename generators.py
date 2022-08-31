import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from copy import deepcopy

import propinfer as pia


class NN_Generator(pia.Generator):
    def __init__(self, n_samples=1024, layers=(32,)):
        super().__init__(n_samples)
                 
        seq = list()
        input_size = 4
                 
        for l in layers:
            seq.extend([
                nn.Linear(input_size, l),
                nn.ReLU()
            ])
            input_size = l

        seq.extend([
            nn.Linear(input_size, 1)
        ])
                 
        for layer in seq:
            if type(layer) == nn.Linear:
                layer.weight.data = torch.randn(layer.weight.size())
                layer.bias.data = torch.randn(layer.bias.size())

        self.model = nn.Sequential(*seq)
                 
    
    def sample(self, label, adv=False):
        x = np.random.normal(-1. + 2*label, 2.0, size=(self.n_samples, 4))
        y = self.model(torch.tensor(x, dtype=torch.float32)).cpu().detach().numpy()
        
        df = pd.DataFrame(data=x)
        df['label'] = y
                 
        return df
                 
                 
class Binary_NN_Generator(pia.Generator):
    def __init__(self, base_nn_gen: NN_Generator, epsilon=1.):
        super().__init__(base_nn_gen.n_samples)
                 
        self.model = base_nn_gen.model
        self.other_model = deepcopy(self.model)
        
        for layer in self.other_model:
            if type(layer) == nn.Linear:
                layer.weight.data += torch.randn(layer.weight.size()) * epsilon
                layer.bias.data += torch.randn(layer.bias.size()) * epsilon

    
    def sample(self, label, adv=False):
        x = np.random.normal(0., 2.0, size=(self.n_samples, 4))
        
        if not label:
            y = self.model(torch.tensor(x, dtype=torch.float32)).cpu().detach().numpy()
        else:
            y = self.other_model(torch.tensor(x, dtype=torch.float32)).cpu().detach().numpy()
        
        df = pd.DataFrame(data=x)
        df['label'] = y
        
        return df
                 
                 
class Membership_Generator(pia.Generator):
    def __init__(self, n_samples=1024, max_i=4, causal=False, output_env=False):
        super().__init__(n_samples)
        self.max_i = max_i
        self.causal = causal
        self.output_env = output_env
    
    def sample(self, label, adv=False):
        dfs = list()
            
        for i in range(self.max_i):
            if (i == self.max_i - 1) and not label:
                break
            n_samples = self.n_samples//self.max_i + self.n_samples % self.max_i if i == 0 else self.n_samples//self.max_i
            x1 = np.random.normal(0., 1.0, size=n_samples)
            y = x1 + np.random.normal(0., 1.0, size=n_samples)
            x2 = y + np.random.normal(0., 0.5 + 1.0*i, size=n_samples)
            dfs.append(pd.DataFrame(data={'x1':x1, 'x2':x2, 'label':y}))
            
            if self.output_env:
                dfs[-1]['env'] = i
                
        df = pd.concat(dfs)
        if self.causal:
            df = df.drop('x2', axis=1)
            
        return df