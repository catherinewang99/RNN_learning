import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp



# Initialize experiment
exp = DualALMRNNExp()

# Train RNN for each input asymmetry
for seed in range(100):
    for epoch in [1, 10, 20, 30]:
        exp.configs['xs_left_alm_amp'] = 1.0
        exp.configs['xs_right_alm_amp'] = 1.0
        exp.configs['random_seed'] = seed
        exp.configs['train_type'] = 'train_type_modular_asymmetric_fixed_input'
        exp.configs['n_neurons'] = 128
        exp.configs['unfix_epoch'] = epoch
        exp.configs['bs'] = 256
        exp.configs['lr'] = 1e-4
        exp.configs['n_epochs'] = 30
        exp.train_type_modular_single_readout()