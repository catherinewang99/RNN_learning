import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

import argparse, os, math, pickle, json

# 1) Load your model
from dual_alm_rnn_models import *
from dual_alm_rnn_exp import DualALMRNNExp

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--single', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--best', action='store_true', default=False, help='Use best_model.pth instead of last_model.pth')
    parser.add_argument('--epoch', type=int, default=None, help='Use model_epoch_<epoch>.pth checkpoint')
    return parser.parse_args()

args = parse_args()



exp = DualALMRNNExp()

# Generate no noise trial input
T = exp.T
sample_begin = exp.sample_begin
delay_begin = exp.delay_begin

presample_mask = np.zeros((T,), dtype=bool)
presample_mask[:sample_begin] = True
presample_inds = np.arange(0,sample_begin)

sample_mask = np.zeros((T,), dtype=bool)
sample_mask[sample_begin:delay_begin] = True
sample_inds = np.arange(sample_begin,delay_begin)


delay_mask = np.zeros((T,), dtype=bool)
delay_mask[delay_begin:] = True
delay_inds = np.arange(delay_begin,T)

r_trial_input, l_trial_input = [],[]

r_trial_input = np.zeros((T, 2), dtype=np.float32)
l_trial_input = np.zeros((T, 2), dtype=np.float32)

l_trial_input[np.ix_(sample_inds, [0])] = 0.5
r_trial_input[np.ix_(sample_inds, [1])] = 0.5


# Proprogate through model

# Load configs to get model parameters
with open('dual_alm_rnn_configs.json', 'r') as f:
    configs = json.load(f)

model = getattr(sys.modules[__name__], configs['model_type'])(configs, \
    exp.a, exp.pert_begin, exp.pert_end, noise=False)
# model = TwoHemiRNNTanh_single_readout(configs, exp.a, exp.pert_begin, exp.pert_end)
exp.init_sub_path(configs['train_type'])
if args.epoch is not None:
    checkpoint_file = f'model_epoch_{args.epoch}.pth'
elif args.best:
    checkpoint_file = 'best_model.pth'
else:
    checkpoint_file = 'last_model.pth'
checkpoint_path = os.path.join(configs['models_dir'], configs['model_type'], exp.sub_path, checkpoint_file)
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
print(checkpoint_path)
state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
# params = {'batch_size': configs['bs'], 'shuffle': True}
# inputs = data.TensorDataset(torch.tensor(sensory_inputs), torch.tensor(trial_type_labels))
# inputs = data.DataLoader(inputs, **params)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update
model = model.to(device)

# Draw the readout trajectory
bias = model.readout_linear.bias.data.cpu().numpy()[0]
readout_weights = model.readout_linear.weight.data.cpu().numpy()
recurrent_weights = model.rnn_cell.w_hh_linear_ll.weight.data.cpu().numpy()

print(bias, readout_weights, recurrent_weights)

# Pass input into model
l_trial_input = l_trial_input.reshape(1, T, 2)
r_trial_input = r_trial_input.reshape(1, T, 2)
l_trial_input = torch.tensor(l_trial_input).to(device)
r_trial_input = torch.tensor(r_trial_input).to(device)


hs, _ = model(l_trial_input)
hs_r, _ = model(r_trial_input)
# import pdb; pdb.set_trace()


if args.single:
    ## PLOT LEFT HEMISPHERE TRAJECTORY
    plt.plot(hs[0, :, 0].detach().cpu().numpy(), hs[0, :, 1].detach().cpu().numpy(), color='red', marker='o', alpha=0.5)
    plt.plot(hs_r[0, :, 0].detach().cpu().numpy(), hs_r[0, :, 1].detach().cpu().numpy(), color='blue', marker='o', alpha=0.5)

    # Extend the readout line to cover the whole plot
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # The readout line: w0*x + w1*y + b = 0
    w0, w1 = readout_weights[0, 0], readout_weights[0, 1]
    b = bias

    # To plot the line, solve for y in terms of x: w0*x + w1*y + b = 0 => y = -(w0*x + b)/w1
    # But if w1 is very small, plot vertically instead
    if abs(w1) > 1e-6:
        x_vals = np.array(xlim)
        y_vals = -(w0 * x_vals + b) / w1
        plt.plot(x_vals, y_vals, color='green', ls='--', alpha=0.5, label='Readout boundary')
    else:
        # Vertical line: x = -b/w0
        x_val = -b / w0 if abs(w0) > 1e-6 else 0
        plt.axvline(x_val, color='green', ls='--', alpha=0.5, label='Readout boundary')

    plt.scatter(hs_r[0, 0, 0].detach().cpu().numpy(), hs_r[0, 0, 1].detach().cpu().numpy(), c='grey', s=100, zorder=10)
    plt.scatter(hs_r[0, sample_begin, 0].detach().cpu().numpy(), hs_r[0, sample_begin, 1].detach().cpu().numpy(), c='orange', s=100, zorder=10)
    plt.scatter(hs_r[0, delay_begin, 0].detach().cpu().numpy(), hs_r[0, delay_begin, 1].detach().cpu().numpy(), c='pink', s=100, zorder=10)
    plt.scatter(hs_r[0, -1, 0].detach().cpu().numpy(), hs_r[0, -1, 1].detach().cpu().numpy(), c='black', s=100, zorder=10)


    plt.scatter(hs[0, 0, 0].detach().cpu().numpy(), hs[0, 0, 1].detach().cpu().numpy(), c='grey', s=100, zorder=10)
    plt.scatter(hs[0, sample_begin, 0].detach().cpu().numpy(), hs[0, sample_begin, 1].detach().cpu().numpy(), c='orange', s=100, zorder=10)
    plt.scatter(hs[0, delay_begin, 0].detach().cpu().numpy(), hs[0, delay_begin, 1].detach().cpu().numpy(), c='pink', s=100, zorder=10)
    plt.scatter(hs[0, -1, 0].detach().cpu().numpy(), hs[0, -1, 1].detach().cpu().numpy(), c='black', s=100, zorder=10)
    plt.axhline(0, color='grey',ls='--')
    plt.axvline(0, color='grey',ls='--')
    plt.xlabel('Unit 1 activity (a.u.)')
    plt.ylabel('Unit 2 activity (a.u.)')
    plt.title('L vs R Trial Trajectory')
    # plt.ylim(-1,1)
    # plt.xlim(-1,1)
    plt.legend()
    plt.savefig('left_hemisphere_trial_trajectory_seed{}_L{}_R{}.pdf'.format(configs['random_seed'], configs['xs_left_alm_drop_p'], configs['xs_right_alm_drop_p']))
    plt.show()


    ## PLOT RIGHT HEMISPHERE TRAJECTORY
    plt.plot(hs_r[0, :, 2].detach().cpu().numpy(), hs_r[0, :, 3].detach().cpu().numpy(), color='blue', marker='o', alpha=0.5)
    plt.plot(hs[0, :, 2].detach().cpu().numpy(), hs[0, :, 3].detach().cpu().numpy(), color='red', marker='o', alpha=0.5)

    # Extend the readout line to cover the whole plot
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # The readout line: w0*x + w1*y + b = 0
    w0, w1 = readout_weights[0, 2], readout_weights[0, 3]
    b = bias

    # To plot the line, solve for y in terms of x: w0*x + w1*y + b = 0 => y = -(w0*x + b)/w1
    # But if w1 is very small, plot vertically instead
    if abs(w1) > 1e-6:
        x_vals = np.array(xlim)
        y_vals = -(w0 * x_vals + b) / w1
        plt.plot(x_vals, y_vals, color='green', ls='--', alpha=0.5, label='Readout boundary')
    else:
        # Vertical line: x = -b/w0
        x_val = -b / w0 if abs(w0) > 1e-6 else 0
        plt.axvline(x_val, color='green', ls='--', alpha=0.5, label='Readout boundary')

    plt.scatter(hs_r[0, 0, 2].detach().cpu().numpy(), hs_r[0, 0, 3].detach().cpu().numpy(), c='grey', s=100, zorder=10)
    plt.scatter(hs_r[0, sample_begin, 2].detach().cpu().numpy(), hs_r[0, sample_begin, 3].detach().cpu().numpy(), c='orange', s=100, zorder=10)
    plt.scatter(hs_r[0, delay_begin, 2].detach().cpu().numpy(), hs_r[0, delay_begin, 3].detach().cpu().numpy(), c='pink', s=100, zorder=10)
    plt.scatter(hs_r[0, -1, 2].detach().cpu().numpy(), hs_r[0, -1, 3].detach().cpu().numpy(), c='black', s=100, zorder=10)

    plt.scatter(hs[0, 0, 2].detach().cpu().numpy(), hs[0, 0, 3].detach().cpu().numpy(), c='grey', s=100, zorder=10)
    plt.scatter(hs[0, sample_begin, 2].detach().cpu().numpy(), hs[0, sample_begin, 3].detach().cpu().numpy(), c='orange', s=100, zorder=10)
    plt.scatter(hs[0, delay_begin, 2].detach().cpu().numpy(), hs[0, delay_begin, 3].detach().cpu().numpy(), c='pink', s=100, zorder=10)
    plt.scatter(hs[0, -1, 2].detach().cpu().numpy(), hs[0, -1, 3].detach().cpu().numpy(), c='black', s=100, zorder=10)
    plt.axhline(0, color='grey',ls='--')
    plt.axvline(0, color='grey',ls='--')
    plt.xlabel('Unit 3 activity (a.u.)')
    plt.ylabel('Unit 4 activity (a.u.)')
    plt.title('L vs R Trial Trajectory')
    # plt.ylim(-1,1)
    # plt.xlim(-1,1)
    plt.legend()
    plt.savefig('right_hemisphere_trial_trajectory_seed{}_L{}_R{}.pdf'.format(configs['random_seed'], configs['xs_left_alm_drop_p'], configs['xs_right_alm_drop_p']))
    plt.show()

if args.all: # Plot both left and right hemisphere trajectories and accuracies and weights

    if 'train_type_modular_corruption' in exp.configs['train_type']:
        results_dict = np.load(
            'dual_alm_rnn_logs/{}/train_type_modular_corruption/onehot_cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_4_random_seed_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(
                exp.configs['model_type'],
                exp.configs['corruption_type'],
                exp.configs['corruption_start_epoch'],
                float(exp.configs['corruption_noise']),
                exp.configs['random_seed'],
                float(exp.configs['xs_left_alm_amp']),
                float(exp.configs['xs_right_alm_amp'])
            ),
            allow_pickle=True
        )
    else:
        results_dict = np.load(
            'dual_alm_rnn_logs/{}/{}/n_neurons_4_random_seed_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(
                exp.configs['model_type'],
                exp.configs['train_type'],
                exp.configs['random_seed'],
                float(exp.configs['xs_left_alm_amp']),
                float(exp.configs['xs_right_alm_amp'])
            ),
            allow_pickle=True
        )

    epochs = np.arange(1, len(results_dict) + 1)
    readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])
    readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])

    n_trials_agreed = np.array([results_dict[i]['control']['n_trials_agreed'] for i in range(len(results_dict))])
    n_trials = np.array([results_dict[i]['control']['n_trials'] for i in range(len(results_dict))])
    agreement_frac = n_trials_agreed / n_trials

    # Compute chance level: p(agree) = p_L^2 + (1-p_L)^2 if p_L = p_R, but here use both
    chance_agree = readout_acc_left * readout_acc_right + (1 - readout_acc_left) * (1 - readout_acc_right)

    corruption_start_epoch = exp.configs['corruption_start_epoch']

    # Build composite figure: left = weights viz; right = 2x2 subplots
    from matplotlib import gridspec
    fig = plt.figure(figsize=(14, 7))
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # Left panel: weights visualization (rendered as image to preserve exact settings)
    left_ax = fig.add_subplot(outer_gs[0, 0])
    try:
        # Lazy import to avoid heavy import if not needed
        from visualize_rnn_weights import visualize_rnn_weights
        model_dir = os.path.join(configs['models_dir'], configs['model_type'], exp.sub_path)
        tmp_img_path = os.path.join('figs', 'tmp_weights_viz.png')
        os.makedirs('figs', exist_ok=True)
        # Temporarily suppress plt.show() inside visualize_rnn_weights
        _orig_show = plt.show
        try:
            plt.show = lambda *args, **kwargs: None
            mode_str = f'epoch_{args.epoch}' if args.epoch is not None else ('best' if args.best else 'last')
            visualize_rnn_weights(model_dir, configs, mode_str, save_path=tmp_img_path)
        finally:
            plt.show = _orig_show
        img = plt.imread(tmp_img_path)
        left_ax.imshow(img)
        left_ax.axis('off')
        left_ax.set_title('RNN Weights', fontsize=12)
    except Exception as e:
        left_ax.text(0.5, 0.5, f'Weights viz failed: {e}', ha='center', va='center')
        left_ax.axis('off')

    # Right panel: 2x2 subplots
    right_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[0, 1], hspace=0.35, wspace=0.3)

    # Top-left: Left hemisphere trial trajectory (left vs right trials)
    ax_tl = fig.add_subplot(right_gs[0, 0])
    ax_tl.plot(hs[0, :, 0].detach().cpu().numpy(), hs[0, :, 1].detach().cpu().numpy(), color='red', marker='o', alpha=0.5, label='Left trial')
    ax_tl.plot(hs_r[0, :, 0].detach().cpu().numpy(), hs_r[0, :, 1].detach().cpu().numpy(), color='blue', marker='o', alpha=0.5, label='Right trial')
    ax_tl.axhline(0, color='grey', ls='--', linewidth=1)
    ax_tl.axvline(0, color='grey', ls='--', linewidth=1)
    ax_tl.set_xlabel('Unit 1 activity')
    ax_tl.set_ylabel('Unit 2 activity')
    ax_tl.set_title('Left Hemisphere Trajectory')
    ax_tl.legend(fontsize=8)
    # Readout decision boundary for left hemisphere: w0*x + w1*y + b = 0
    w0_l, w1_l = readout_weights[0, 0], readout_weights[0, 1]
    b_l = bias
    xlim = ax_tl.get_xlim()
    if abs(w1_l) > 1e-6:
        x_vals = np.array(xlim)
        y_vals = -(w0_l * x_vals + b_l) / w1_l
        ax_tl.plot(x_vals, y_vals, color='green', ls='--', alpha=0.7, label='Readout boundary')
    else:
        x_val = -b_l / w0_l if abs(w0_l) > 1e-6 else 0
        ax_tl.axvline(x_val, color='green', ls='--', alpha=0.7, label='Readout boundary')

    # Top-right: Right hemisphere trial trajectory (left vs right trials)
    ax_tr = fig.add_subplot(right_gs[0, 1])
    ax_tr.plot(hs[0, :, 2].detach().cpu().numpy(), hs[0, :, 3].detach().cpu().numpy(), color='red', marker='o', alpha=0.5, label='Left trial')
    ax_tr.plot(hs_r[0, :, 2].detach().cpu().numpy(), hs_r[0, :, 3].detach().cpu().numpy(), color='blue', marker='o', alpha=0.5, label='Right trial')
    ax_tr.axhline(0, color='grey', ls='--', linewidth=1)
    ax_tr.axvline(0, color='grey', ls='--', linewidth=1)
    ax_tr.set_xlabel('Unit 3 activity')
    ax_tr.set_ylabel('Unit 4 activity')
    ax_tr.set_title('Right Hemisphere Trajectory')
    ax_tr.legend(fontsize=8)
    # Readout decision boundary for right hemisphere: w2*x + w3*y + b = 0
    w0_r, w1_r = readout_weights[0, 2], readout_weights[0, 3]
    b_r = bias
    xlim = ax_tr.get_xlim()
    if abs(w1_r) > 1e-6:
        x_vals = np.array(xlim)
        y_vals = -(w0_r * x_vals + b_r) / w1_r
        ax_tr.plot(x_vals, y_vals, color='green', ls='--', alpha=0.7, label='Readout boundary')
    else:
        x_val = -b_r / w0_r if abs(w0_r) > 1e-6 else 0
        ax_tr.axvline(x_val, color='green', ls='--', alpha=0.7, label='Readout boundary')

    # Bottom row: Merge into one subplot spanning both columns, plot L (red) and R (blue)
    ax_b = fig.add_subplot(right_gs[1, :])
    ax_b.plot(epochs, readout_acc_left, color='r', label='Left Readout Accuracy')
    ax_b.plot(epochs, readout_acc_right, color='b', label='Right Readout Accuracy')
    if exp.configs['train_type'] == 'train_type_modular_corruption':
        ax_b.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start')
    ax_b.set_xlabel('Epoch')
    ax_b.set_ylabel('Accuracy (Control)')
    ax_b.set_title('Left/Right Hemisphere Readout Accuracy')
    # X ticks: always show 1 and last; if specific epoch requested, include it; else show every 5
    xticks = [1]
    if args.epoch is not None and args.epoch not in xticks and args.epoch <= epochs[-1]:
        xticks.append(args.epoch)
    # add multiples of 5
    xticks.extend([int(x) for x in epochs if x % 5 == 0])
    if epochs[-1] not in xticks:
        xticks.append(int(epochs[-1]))
    xticks = sorted(set(xticks))
    ax_b.set_xticks(xticks)
    ax_b.set_xticklabels([str(int(x)) for x in xticks])
    ax_b.set_ylim(0.4, 1.05)
    ax_b.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    out_path = os.path.join('figs', 'weights_and_trajectories_seed{}_L{}_R{}_epoch_{}.pdf'.format(
        configs['random_seed'], float(exp.configs['xs_left_alm_amp']), float(exp.configs['xs_right_alm_amp']), args.epoch))
    os.makedirs('figs', exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved composite figure to {out_path}')
    plt.show()