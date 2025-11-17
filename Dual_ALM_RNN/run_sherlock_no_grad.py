import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp


# Initialize experiment
exp = DualALMRNNExp()

# Train RNN for each input asymmetry
for seed in range(50):
    for lr in [1e-2, 5e-1, 1e-3, 1e-4]:
        exp.configs['xs_left_alm_amp'] = 1.0
        exp.configs['xs_right_alm_amp'] = 1.0
        exp.configs['n_neurons'] = 4
        exp.configs['random_seed'] = seed
        exp.configs['bs'] = 10
        exp.configs['train_type'] = 'train_type_modular_fixed_input_no_grad'
        exp.configs['lr'] = lr

        exp.train_type_modular_single_readout()

