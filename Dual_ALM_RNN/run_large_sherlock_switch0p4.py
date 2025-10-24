import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp



# Initialize experiment
exp = DualALMRNNExp()


               
# Train RNN for each input asymmetry
for seed in range(50):
    exp.configs['train_type'] = 'train_type_modular_fixed_input_cross_hemi_switch'
    exp.configs['switch_epoch_n'] = 0
    exp.configs['xs_left_alm_amp'] = 1.0
    exp.configs['xs_right_alm_amp'] = 1.0
    exp.configs['random_seed'] = seed
    exp.configs['unfix_epoch'] = 0
    exp.configs['init_cross_hemi_rel_factor'] = 0.5
    exp.configs['switch_ps'] = [0.4, 0.2, 0.2, 0.2]
    exp.configs['bs'] = 256
    exp.configs['n_neurons'] = 256
    exp.configs['lr'] = 1e-4
    exp.configs['n_epochs'] = 45
    
    exp.train_type_modular_single_readout()