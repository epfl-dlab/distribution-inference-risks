import propinfer as pia
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from tqdm.notebook import tqdm, trange
from copy import deepcopy

import torch
import torch.nn as nn

from constants import N_RUNS, N_TARGETS, N_SHADOWS, DEEPSETS_HYPERPARAMS, MLP_HYPERPARAMS
from generators import NN_Generator, Binary_NN_Generator, Membership_Generator
from models import MLPEarlyStopping, IRM


def experiment_a(results_folder):
    res = pd.DataFrame(columns=['epsilon', 'MAE/Acc', 'Experiment', 'Attack'])
    perf = pd.DataFrame(columns=['epsilon', 'MSE', 'Experiment'])
    
    # We keep a same neural network as a basis for D_0
    base_nn_gen = NN_Generator(2**11)
    
    # Sampling only D_0 training set, since the distribution of D_1 will change
    df_0 = Binary_NN_Generator(base_nn_gen=base_nn_gen, epsilon=0.).sample(False)

    for epsilon in np.arange(0., 0.21, 0.04):
        
        print("Running experiment A with epsilon = {:.2f}".format(epsilon))
        gen = Binary_NN_Generator(base_nn_gen=base_nn_gen, epsilon=epsilon)
        
        # Experiment A - Supports classification only
        
        exp_class = pia.Experiment(gen, 'label', pia.MLP, N_TARGETS, N_SHADOWS, MLP_HYPERPARAMS, n_classes=2)
        
        exp_class.run_targets()
        exp_class.run_shadows()
    
        res.loc[len(res)] = (epsilon,
                             exp_class.run_whitebox_deepsets(DEEPSETS_HYPERPARAMS, N_RUNS), 
                             'Classification', 
                             'WhiteBox')
        
        res.loc[len(res)] = (epsilon, 
                             exp_class.run_blackbox(N_RUNS), 
                             'Classification', 
                             'BlackBox')
        
        # Since E(Y|X) changes depending on the classification label, we only test the models against their own class test set
        
        df_1 = gen.sample(True)
        for t, target in enumerate(exp_class.targets):
            if exp_class.labels[t]:
                perf.loc[len(perf)] = (epsilon, mean_squared_error(df_1.label, target.predict(df_1)), 'Classification')
            else:
                perf.loc[len(perf)] = (epsilon, mean_squared_error(df_0.label, target.predict(df_0)), 'Classification')
        
    res.explode('MAE/Acc').reset_index(drop=True).to_pickle(join(results_folder, 'exp_a_res.pkl'))
    perf.to_pickle(join(results_folder, 'exp_a_perf.pkl'))
    

def experiment_abis(results_folder):
    res = pd.DataFrame(columns=['early_stop', 'MAE/Acc', 'Experiment', 'Attack'])
    perf = pd.DataFrame(columns=['early_stop', 'MSE', 'Experiment'])

    base_nn_gen = NN_Generator(2**11)
    gen = Binary_NN_Generator(base_nn_gen=base_nn_gen, epsilon=0.05)
    
    # Sampling target models' test sets
    
    df_0 = gen.sample(False)
    df_1 = gen.sample(True)

    for early_stop in [50, 40, 30, 20, 10]:
    
        print("Running experiment A-bis with early stopping at an MSE of {:.1f}".format(early_stop))
    
        hyperparams = MLP_HYPERPARAMS.copy()
        hyperparams['early_stop'] = early_stop
        
        # Experiment A-bis - Supports classification only
        
        exp_class = pia.Experiment(gen, 'label', MLPEarlyStopping, N_TARGETS, N_SHADOWS, hyperparams, n_classes=2)
        
        exp_class.run_targets()
        exp_class.run_shadows()
    
        res.loc[len(res)] = (early_stop,
                             exp_class.run_whitebox_deepsets(DEEPSETS_HYPERPARAMS, N_RUNS), 
                             'Classification', 
                             'WhiteBox')
        
        res.loc[len(res)] = (early_stop, 
                             exp_class.run_blackbox(N_RUNS), 
                             'Classification', 
                             'BlackBox')

        # Since E(Y|X) changes depending on the classification label, we only test the models against their own class test set
        
        for t, target in enumerate(exp_class.targets):
            if exp_class.labels[t]:
                perf.loc[len(perf)] = (early_stop, mean_squared_error(df_1.label, target.predict(df_1)), 'Classification')
            else:
                perf.loc[len(perf)] = (early_stop, mean_squared_error(df_0.label, target.predict(df_0)), 'Classification')
        
    res.explode('MAE/Acc').reset_index(drop=True).to_pickle(join(results_folder, 'exp_abis_res.pkl'))
    perf.to_pickle(join(results_folder, 'exp_abis_perf.pkl'))
    
    
def experiment_b(results_folder):
    res = pd.DataFrame(columns=['early_stop', 'MAE/Acc', 'Experiment', 'Attack'])
    perf = pd.DataFrame(columns=['early_stop', 'MSE', 'Experiment'])

    gen = NN_Generator(2**11)
    
    # Sampling target models' test sets
    
    df_0 = gen.sample(False)
    df_1 = gen.sample(True)

    for early_stop in [50, 40, 30, 20, 10]:
    
        print("Running experiment B with early stopping at an MSE of {:.1f}".format(early_stop))
    
        hyperparams = MLP_HYPERPARAMS.copy()
        hyperparams['early_stop'] = early_stop
        
        # Experiment B - Regression
        
        exp_reg = pia.Experiment(gen, 'label', MLPEarlyStopping, N_TARGETS, N_SHADOWS, hyperparams, n_classes=1, range=(0., 1.))
    
        exp_reg.run_targets()
        exp_reg.run_shadows()
    
        res.loc[len(res)] = (early_stop, 
                             exp_reg.run_whitebox_deepsets(DEEPSETS_HYPERPARAMS, N_RUNS), 
                             'Regression',
                             'WhiteBox')
        
        res.loc[len(res)] = (early_stop, 
                             exp_reg.run_blackbox(N_RUNS), 
                             'Regression', 
                             'BlackBox')
        
        # Experiment B - Classification
        
        exp_class = pia.Experiment(gen, 'label', MLPEarlyStopping, N_TARGETS, N_SHADOWS, hyperparams, n_classes=2)
        
        exp_class.run_targets()
        exp_class.run_shadows()
    
        res.loc[len(res)] = (early_stop,
                             exp_class.run_whitebox_deepsets(DEEPSETS_HYPERPARAMS, N_RUNS), 
                             'Classification', 
                             'WhiteBox')
        
        res.loc[len(res)] = (early_stop, 
                             exp_class.run_blackbox(N_RUNS), 
                             'Classification', 
                             'BlackBox')
    
        # Target models performance
    
        for t in exp_reg.targets:
            perf.loc[len(perf)] = (early_stop, mean_squared_error(df_0.label, t.predict(df_0)), 'Regression')
            perf.loc[len(perf)] = (early_stop, mean_squared_error(df_1.label, t.predict(df_1)), 'Regression')
            
        for t in exp_class.targets:
            perf.loc[len(perf)] = (early_stop, mean_squared_error(df_0.label, t.predict(df_0)), 'Classification')
            perf.loc[len(perf)] = (early_stop, mean_squared_error(df_1.label, t.predict(df_1)), 'Classification')
        
    res.explode('MAE/Acc').reset_index(drop=True).to_pickle(join(results_folder, 'exp_b_res.pkl'))
    perf.to_pickle(join(results_folder, 'exp_b_perf.pkl')) 

    
def experiment_c(results_folder):
    res = pd.DataFrame(columns=['n_samples', 'MAE/Acc', 'Experiment', 'Attack'])
    perf = pd.DataFrame(columns=['n_samples', 'MSE', 'Experiment'])
    
    gen = NN_Generator(2**10)
    
    # Sampling target models' test sets
    
    df_0 = gen.sample(False)
    df_1 = gen.sample(True)

    for n_samples in range(512, 2049, 512):
        print("Running experiment C with {} samples".format(n_samples))
    
        gen.n_samples = n_samples
        
        # Experiment C - Regression
        
        exp_reg = pia.Experiment(gen, 'label', pia.MLP, N_TARGETS, N_SHADOWS, MLP_HYPERPARAMS, n_classes=1, range=(0., 1.), n_queries=512)
    
        exp_reg.run_targets()
        exp_reg.run_shadows()
    
        res.loc[len(res)] = (n_samples, 
                             exp_reg.run_whitebox_deepsets(DEEPSETS_HYPERPARAMS, N_RUNS), 
                             'Regression', 
                             'WhiteBox')
        res.loc[len(res)] = (n_samples, 
                             exp_reg.run_blackbox(N_RUNS), 
                             'Regression', 
                             'BlackBox')
        
        # Experiment C - Classification
        
        exp_class = pia.Experiment(gen, 'label', pia.MLP, N_TARGETS, N_SHADOWS, MLP_HYPERPARAMS, n_classes=2, n_queries=512)
        
        exp_class.run_targets()
        exp_class.run_shadows()
    
        res.loc[len(res)] = (n_samples, 
                             exp_class.run_whitebox_deepsets(DEEPSETS_HYPERPARAMS, N_RUNS), 
                             'Classification', 
                             'WhiteBox')
        
        res.loc[len(res)] = (n_samples, 
                             exp_class.run_blackbox(N_RUNS), 
                             'Classification', 
                             'BlackBox')
    
        # Target models performance
    
        for t in exp_reg.targets:
            perf.loc[len(perf)] = (n_samples, mean_squared_error(df_0.label, t.predict(df_0)), 'Regression')
            perf.loc[len(perf)] = (n_samples, mean_squared_error(df_1.label, t.predict(df_1)), 'Regression')
            
        for t in exp_class.targets:
            perf.loc[len(perf)] = (n_samples, mean_squared_error(df_0.label, t.predict(df_0)), 'Classification')
            perf.loc[len(perf)] = (n_samples, mean_squared_error(df_1.label, t.predict(df_1)), 'Classification')
            
    res.explode('MAE/Acc').reset_index(drop=True).to_pickle(join(results_folder, 'exp_c_res.pkl'))
    perf.to_pickle(join(results_folder, 'exp_c_perf.pkl'))       

    
def experiment_causal(results_folder):
    res = pd.DataFrame(columns=['Model', 'MAE/Acc', 'Experiment', 'Attack'])
    perf = pd.DataFrame(columns=['Model', 'Set', 'MSE', 'Experiment'])
    
    # Test sets
    
    gen = Membership_Generator()
    
    df_0_valid = gen.sample(False)
    df_1_valid = gen.sample(True)
    
    df_0_test = deepcopy(df_0_valid)
    df_0_test['x2'] = -df_0_test['x2']
    df_1_test = deepcopy(df_1_valid)
    df_1_test['x2'] = -df_1_test['x2']
    
    for model_str in ['ERM', 'Causal ERM', 'IRM']:
        
        # Experiment setup
        
        print('Running causal experiment with model: ' + model_str)
        
        gen = Membership_Generator(2**11, causal=(model_str == 'Causal ERM'), output_env=(model_str == 'IRM'))
        
        hyperparams = {
            'input_size': 2,
            'n_classes': 1,
            'epochs': 10,
            'learning_rate': 1e-2,
            'weight_decay': 1e-3,
            'normalise': False,
            'layers': (2,),
            'bs': 256
        }
        
        if model_str == 'Causal ERM':
            hyperparams['input_size'] = 1
        elif model_str == 'IRM':
            hyperparams['epochs'] = 2**11
            hyperparams['reg'] = 0.1
            hyperparams['env_label'] = 'env'
        
        model = pia.MLP if not model_str == 'IRM' else IRM
            
        # Causal - Classification
        
        exp_class = pia.Experiment(gen, 'label', model, N_TARGETS, N_SHADOWS, hyperparams, n_classes=2)

        exp_class.run_targets()
        exp_class.run_shadows()
    
        res.loc[len(res)] = (model_str, 
                             exp_class.run_whitebox_deepsets(DEEPSETS_HYPERPARAMS, N_RUNS),
                             'Classification',
                             'WhiteBox')
    
        res.loc[len(res)] = (model_str, 
                             exp_class.run_blackbox(N_RUNS),
                             'Classification',
                             'BlackBox')
        
        
        if model_str != 'Causal ERM':
            for t in exp_class.targets:
                perf.loc[len(perf)] = (model_str, 'Validation', 
                                       mean_squared_error(df_0_valid.label, t.predict(df_0_valid)), 'Classification')
                perf.loc[len(perf)] = (model_str, 'Validation', 
                                       mean_squared_error(df_1_valid.label, t.predict(df_1_valid)), 'Classification')
                perf.loc[len(perf)] = (model_str, 'Test', 
                                       mean_squared_error(df_0_test.label, t.predict(df_0_test)), 'Classification')
                perf.loc[len(perf)] = (model_str, 'Test', 
                                       mean_squared_error(df_1_test.label, t.predict(df_1_test)), 'Classification')
        else:
            for t in exp_class.targets:
                perf.loc[len(perf)] = (model_str, 'Validation', 
                                       mean_squared_error(df_0_valid.label, t.predict(df_0_valid.drop('x2', axis=1))), 'Classification')
                perf.loc[len(perf)] = (model_str, 'Validation', 
                                       mean_squared_error(df_1_valid.label, t.predict(df_1_valid.drop('x2', axis=1))), 'Classification')
                perf.loc[len(perf)] = (model_str, 'Test', 
                                       mean_squared_error(df_0_test.label, t.predict(df_0_test.drop('x2', axis=1))), 'Classification')
                perf.loc[len(perf)] = (model_str, 'Test', 
                                       mean_squared_error(df_1_test.label, t.predict(df_1_test.drop('x2', axis=1))), 'Classification')
        
    res.explode('MAE/Acc').reset_index(drop=True).to_pickle(join(results_folder, 'exp_causal_res.pkl'))
    perf.to_pickle(join(results_folder, 'exp_causal_perf.pkl'))
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp = sys.argv[1]
        res = 'results/'
        if exp == 'a' or exp == 'A':
            experiment_a(res)
        elif exp == 'abis' or exp == 'A-bis':
            experiment_abis(res)
        elif exp == 'b' or exp == 'B':
            experiment_b(res)
        elif exp == 'c' or exp =='C':
            experiment_c(res)
        elif exp =='causal' or exp == 'd' or exp == 'D':
            experiment_causal(res)