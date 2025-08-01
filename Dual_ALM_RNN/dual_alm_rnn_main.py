# Requires PyTorch v1.0

import argparse, os, math, pickle, json
from dual_alm_rnn_exp import DualALMRNNExp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_dataset', action='store_true', default=False)

    parser.add_argument('--train_type_uniform', action='store_true', default=False)

    parser.add_argument('--train_type_modular', action='store_true', default=False)

    parser.add_argument('--train_type_modular_corruption', action='store_true', default=False)

    parser.add_argument('--plot_cd_traces', action='store_true', default=False)

    parser.add_argument('--plot_weights_changes', action='store_true', default=False)


    return parser.parse_args()

def main():

    args = parse_args()


    if args.generate_dataset:
        exp = DualALMRNNExp()
        exp.generate_dataset()

    if args.train_type_uniform:
        exp = DualALMRNNExp()
        exp.train_type_uniform()

    if args.train_type_modular:
        exp = DualALMRNNExp()
        exp.train_type_modular()
        
    if args.train_type_modular_corruption:
        exp = DualALMRNNExp()
        exp.train_type_modular_corruption()

    if args.plot_cd_traces:
        exp = DualALMRNNExp()
        exp.plot_cd_traces()

    if args.plot_weights_changes:
        exp = DualALMRNNExp()
        exp.plot_weights_changes()



if __name__ == '__main__':
    main()

