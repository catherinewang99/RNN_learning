import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
from dual_alm_rnn_models import TwoHemiRNNTanh_single_readout



# input_asym = [(1,0), (1,0.1), (1,0.2), (1,0.3), (1,0.4), (1,0.5), (1,1), (0.5,1), (0.4,1), (0.3,1), (0.2,1), (0.1,1), (0,1)] # Sane as BK

fix_epoch = [0, 10, 20, 30, 40]

# Initialize experiment
exp = DualALMRNNExp()

# Train RNN for each input asymmetry
for seed in range(10):
    for fix_epoch in fix_epoch:
        exp.configs['xs_left_alm_amp'] = 1.0
        exp.configs['xs_right_alm_amp'] = 1.0
        exp.configs['random_seed'] = seed
        exp.configs['train_type'] = 'train_type_asymmetric_fixed_input'
        exp.configs['unfix_epoch'] = fix_epoch

        exp.train_type_modular_single_readout()


    