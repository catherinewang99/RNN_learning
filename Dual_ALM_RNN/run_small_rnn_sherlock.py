import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp



input_asym = [(1,0), (1,0.1), (1,0.2), (1,0.3), (1,0.4), (1,0.5), (1,1), (0.5,1), (0.4,1), (0.3,1), (0.2,1), (0.1,1), (0,1)] # Sane as BK

# # Initialize experiment
# exp = DualALMRNNExp()

# # Train RNN for each input asymmetry
# for seed in range(10):
#     for left_amp, right_amp in input_asym:
#         exp.configs['xs_left_alm_amp'] = left_amp
#         exp.configs['xs_right_alm_amp'] = right_amp
#         exp.configs['random_seed'] = seed
#         exp.train_type_modular_single_readout()


weights_dict = {}

for asym in input_asym:
    all_weights = []
    for seed in range(10):
        exp = DualALMRNNExp()
        exp.configs['xs_left_alm_amp'] = asym[0]
        exp.configs['xs_right_alm_amp'] = asym[1]
        exp.configs['random_seed'] = seed

        # For each asymmetry, print the logs path for the config
        exp.init_sub_path(exp.configs['train_type'])
        logs_path = os.path.join(exp.configs['logs_dir'], exp.configs['model_type'], exp.sub_path)
        # print(f"Input asymmetry {asym}: logs path = {logs_path}")

        weights = np.load(os.path.join(logs_path, 'readout_weights_epoch_final.npy'))
        # ratio = np.sum(np.abs(weights)[0, :exp.n_neurons//2]) / np.sum(np.abs(weights)[0, :])
        all_weights.append(weights)
    weights_dict[asym] = all_weights

np.save('small_rnn_sherlock_weights_epoch40.npy', weights_dict)
    