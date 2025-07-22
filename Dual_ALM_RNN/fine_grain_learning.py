"""
Test the effect of fine-grain learning on the model's performance.
Look at epochs 1-10 on a set of randomly initialized models.
"""

import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
import matplotlib.pyplot as plt


for random_seed in range(50):  #generate 50 random seeds ie RNNs

    exp = DualALMRNNExp()

    exp.configs['random_seed'] = random_seed

    exp.train_type_modular()


