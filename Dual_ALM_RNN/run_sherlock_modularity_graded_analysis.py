import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from numpy import linalg as LA


import argparse, os, math, pickle, json

# 1) Load your model
from dual_alm_rnn_models import *
from dual_alm_rnn_exp import DualALMRNNExp



exp = DualALMRNNExp()

# Load configs to get model parameters
with open('dual_alm_rnn_configs.json', 'r') as f:
    configs = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update

# start by loading data
train_save_path = os.path.join(exp.configs['data_dir'], 'train')

train_sensory_inputs = np.load(os.path.join(train_save_path, 'onehot_sensory_inputs.npy' ))
train_trial_type_labels = np.load(os.path.join(train_save_path, 'onehot_trial_type_labels.npy'))
# load train inputs and labels

l_trial_idx = np.where(train_trial_type_labels == 0)[0]
r_trial_idx = np.where(train_trial_type_labels == 1)[0]

l_trial_input = train_sensory_inputs[l_trial_idx, :]
r_trial_input = train_sensory_inputs[r_trial_idx, :]

l_trial_input = torch.tensor(l_trial_input).to(device)
r_trial_input = torch.tensor(r_trial_input).to(device)
l_trial_input.shape

sample_begin = exp.sample_begin
delay_begin = exp.delay_begin
l_trial_input_all, r_trial_input_all = l_trial_input, r_trial_input

graded_noise_scales = [0.1, 0.5, 1.0]
# Create an outer dictionary to hold all results for each seed and cross_hemi
results = {}

for seed in range(50):
    results[seed] = {}
    for graded_noise_scale in graded_noise_scales:
        exp.configs['train_type'] = 'train_type_modular_fixed_input_cross_hemi_switch_graded'
        exp.configs['switch_epoch_n'] = 0
        exp.configs['xs_left_alm_amp'] = 1.0
        exp.configs['xs_right_alm_amp'] = 1.0
        exp.configs['random_seed'] = seed
        exp.configs['unfix_epoch'] = 0
        exp.configs['init_cross_hemi_rel_factor'] = 0.5
        exp.configs['graded_signal_scale'] = 0.5
        exp.configs['graded_noise_scale'] = graded_noise_scale
        exp.configs['bs'] = 256
        exp.configs['n_neurons'] = 256
        exp.configs['lr'] = 1e-4
        exp.configs['n_epochs'] = 45
        exp.configs['switch_epoch_n'] = 0

        model = getattr(sys.modules[__name__], configs['model_type'])(exp.configs, \
            exp.a, exp.pert_begin, exp.pert_end, noise=True)
        exp.init_sub_path(exp.configs['train_type'])

        checkpoint_file = 'last_model.pth'
        checkpoint_path = os.path.join(exp.configs['models_dir'], exp.configs['model_type'], exp.sub_path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        print(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()


        # Get quantitative measure of the dip

        ### After data loading, run model # with no noise:
        model.train_type = "train_type_modular_fixed_input_cross_hemi"
        model.return_input = True

        pert_begin, pert_end = exp.pert_begin, exp.pert_end
        pert_end = pert_begin + 10

        model.uni_pert_trials_prob = 0.0
        model.left_alm_pert_prob = 0.5
        _, hs_l, zs_l = model(l_trial_input_all)    # left trial
        _, hs_r, zs_r = model(r_trial_input_all)    # right trial

        left_readout_l_ctl = hs_l[:, :, :model.n_neurons//2].detach().cpu().numpy().dot(
            model.readout_linear.weight.data.cpu().numpy()[0, :model.n_neurons//2]
        )
        right_readout_l_ctl = hs_l[:, :, model.n_neurons//2:].detach().cpu().numpy().dot(
            model.readout_linear.weight.data.cpu().numpy()[0, model.n_neurons//2:]
        )

        left_readout_r_ctl = hs_r[:, :, :model.n_neurons//2].detach().cpu().numpy().dot(
            model.readout_linear.weight.data.cpu().numpy()[0, :model.n_neurons//2]
        )
        right_readout_r_ctl = hs_r[:, :, model.n_neurons//2:].detach().cpu().numpy().dot(
            model.readout_linear.weight.data.cpu().numpy()[0, model.n_neurons//2:]
        )

        all_probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        all_dip_diff, all_dip_sem = [],[]

        for prob in all_probs:
            dip_diff, dip_sem = [],[]

            # For each of the two conditions: left alm pert, right alm pert
            for rowidx, (uni_pert, left_pert) in enumerate([
                    (1.0, 1.0),   # Left ALM perturbed
                    (1.0, 0.0)    # Right ALM perturbed
                ]):

                diff, sem = 0,0
                # Set model perturbation parameters for condition
                model.uni_pert_trials_prob = uni_pert
                model.left_alm_pert_prob = left_pert
                model.drop_p_min = prob
                model.drop_p_max = prob

                _, hs_l, zs_l = model(l_trial_input_all)    # left trial
                _, hs_r, zs_r = model(r_trial_input_all)    # right trial

                # Compute left and right readouts for left and right trials
                left_readout_l = hs_l[:, :, :model.n_neurons//2].detach().cpu().numpy().dot(
                    model.readout_linear.weight.data.cpu().numpy()[0, :model.n_neurons//2]
                )
                right_readout_l = hs_l[:, :, model.n_neurons//2:].detach().cpu().numpy().dot(
                    model.readout_linear.weight.data.cpu().numpy()[0, model.n_neurons//2:]
                )

                left_readout_r = hs_r[:, :, :model.n_neurons//2].detach().cpu().numpy().dot(
                    model.readout_linear.weight.data.cpu().numpy()[0, :model.n_neurons//2]
                )
                right_readout_r = hs_r[:, :, model.n_neurons//2:].detach().cpu().numpy().dot(
                    model.readout_linear.weight.data.cpu().numpy()[0, model.n_neurons//2:]
                )

                # only want contra differences 

                if rowidx == 0:
                    diff += np.sum(np.abs(np.mean(right_readout_l[:, pert_begin:pert_end] - right_readout_l_ctl[:, pert_begin:pert_end], axis=0)))
                    diff += np.sum(np.abs(np.mean(right_readout_r[:, pert_begin:pert_end] - right_readout_r_ctl[:, pert_begin:pert_end], axis=0)))
                    sem += np.sum(np.std(right_readout_l[:, pert_begin:pert_end] - right_readout_l_ctl[:, pert_begin:pert_end], axis=0) / np.sqrt(left_readout_l.shape[0]))
                    sem += np.sum(np.std(right_readout_r[:, pert_begin:pert_end] - right_readout_r_ctl[:, pert_begin:pert_end], axis=0) / np.sqrt(left_readout_l.shape[0]))
                
                elif rowidx == 1:
                    diff = np.sum(np.abs(np.mean(left_readout_l[:, pert_begin:pert_end] - left_readout_l_ctl[:, pert_begin:pert_end], axis=0)))
                    diff += np.sum(np.abs(np.mean(left_readout_r[:, pert_begin:pert_end] - left_readout_r_ctl[:, pert_begin:pert_end], axis=0)))
                    sem = np.sum(np.std(left_readout_l[:, pert_begin:pert_end] - left_readout_l_ctl[:, pert_begin:pert_end], axis=0) / np.sqrt(left_readout_l.shape[0]))
                    sem += np.sum(np.std(left_readout_r[:, pert_begin:pert_end] - left_readout_r_ctl[:, pert_begin:pert_end], axis=0) / np.sqrt(left_readout_l.shape[0]))

                dip_diff.append(diff)
                dip_sem.append(sem)

            all_dip_diff.append(dip_diff)
            all_dip_sem.append(dip_sem)

        # Store results in outer dictionary
        results[seed][graded_noise_scale] = {
            "all_dip_diff": all_dip_diff,
            "all_dip_sem": all_dip_sem,
        }

np.save('/scratch/users/cwang314/sherlock_modularity_graded_analysis.npy', results)






