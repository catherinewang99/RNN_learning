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



# Separability at initial condition

# initially plot all the seeds at the first epoch

# Load configs to get model parameters
with open('dual_alm_rnn_configs.json', 'r') as f:
    configs = json.load(f)

for seed in range(10):

    exp.configs['random_seed'] = seed

    model = getattr(sys.modules[__name__], configs['model_type'])(configs, \
        exp.a, exp.pert_begin, exp.pert_end, noise=False)
    # model = TwoHemiRNNTanh_single_readout(configs, exp.a, exp.pert_begin, exp.pert_end)
    exp.init_sub_path(configs['train_type'])

    checkpoint_file = 'model_epoch_0.pth'
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

    if 'corruption' in exp.configs['train_type']:
        model.corrupt=True


    hs, _ = model(l_trial_input)
    hs_r, _ = model(r_trial_input)
    # import pdb; pdb.set_trace()


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
if 'corruption' in exp.configs['train_type']:
    plt.savefig('left_hemisphere_trial_trajectory_seed{}_L{}_R{}_corruption.pdf'.format(configs['random_seed'], configs['xs_left_alm_drop_p'], configs['xs_right_alm_drop_p']))
else:
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
if 'corruption' in exp.configs['train_type']:
    plt.savefig('right_hemisphere_trial_trajectory_seed{}_L{}_R{}_corruption.pdf'.format(configs['random_seed'], configs['xs_left_alm_drop_p'], configs['xs_right_alm_drop_p']))
else:
    plt.savefig('right_hemisphere_trial_trajectory_seed{}_L{}_R{}.pdf'.format(configs['random_seed'], configs['xs_left_alm_drop_p'], configs['xs_right_alm_drop_p']))
plt.show()






# Time spent on the incorrect side of the reaodut trajectory

