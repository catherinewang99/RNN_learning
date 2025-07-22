"""
Test the effect of fine-grain learning on the model's performance.
Look at epochs 1-10 on a set of randomly initialized models.
"""

import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
import matplotlib.pyplot as plt

def init_accumulator(first_batch):
    """
    Wrap every innermost scalar value in a list.
    first_batch: list[dict[str, dict[str, scalar]]]
    returns: list[dict[str, dict[str, list]]]
    """
    acc = []
    for item in first_batch:
        new_item = {}
        for outer_k, inner_dict in item.items():
            new_item[outer_k] = {inner_k: [val] for inner_k, val in inner_dict.items()}
        acc.append(new_item)
    return acc

def accumulate(accum, new_batch):
    """
    Append values from new_batch into accum (in place).
    accum: output of init_accumulator
    new_batch: same structure as first_batch
    """
    for acc_item, new_item in zip(accum, new_batch):
        for outer_k, inner_dict in new_item.items():
            acc_inner = acc_item[outer_k]
            for inner_k, val in inner_dict.items():
                acc_inner[inner_k].append(val)
    return accum

# Check if all_val_results_dict.npy file exists, if not run the loop
if ~os.path.exists('all_val_results_dict.npy'):

    for random_seed in range(50):  #generate 50 random seeds ie RNNs

        exp = DualALMRNNExp()

        exp.configs['random_seed'] = random_seed

        exp.train_type_modular()

        if random_seed == 0:
            results_dict = np.load(os.path.join(exp.logs_save_path, 'all_val_results_dict.npy'), allow_pickle=True)
            results_dict = init_accumulator(results_dict)
        else:
            temp = np.load(os.path.join(exp.logs_save_path, 'all_val_results_dict.npy'), allow_pickle=True)
            results_dict = accumulate(results_dict, temp)


    np.save('all_val_results_dict.npy', results_dict)


# plot the results

results_dict = np.load('all_val_results_dict.npy', allow_pickle=True)

