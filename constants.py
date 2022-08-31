import os

N_RUNS = 100
N_TARGETS = 1024
N_SHADOWS = 2048

DEEPSETS_HYPERPARAMS = {
    'latent_dim': 8,
    'epochs': 20,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}

MLP_HYPERPARAMS = {
    'input_size': 4,
    'n_classes': 1,
    'epochs': 20,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'normalise': False,
    'layers': (32,),
    'bs': 256
}

RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')