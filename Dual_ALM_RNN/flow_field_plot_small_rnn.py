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

    parser.add_argument('--best', action='store_true', default=False)
    parser.add_argument('--last', action='store_true', default=False)
    return parser.parse_args()

args = parse_args()

# match these to your config
model_kwargs = {
    # "n_neurons": 256,
    "sigma_rec_noise": 0.0,     # no noise for flow field
    "sigma_input_noise": 0.0,
    # … plus any other args your model __init__ needs …
}
with open('dual_alm_rnn_configs.json','r') as read_file:
    configs = json.load(read_file)

t_step = 25 # in ms
tau = 50 # The neuronal time constant in ms.
a = t_step/tau

trial_begin_t = -3100 # in ms


pert_begin_t = -1700
pert_end_t = -900


# Perturbation is applied in [pert_begin,pert_end], inclusive at both ends.
pert_begin =  (pert_begin_t - trial_begin_t)//t_step
pert_end = (pert_end_t - trial_begin_t)//t_step


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update


model = getattr(sys.modules[__name__], "TwoHemiRNNTanh_single_readout")(configs, \
    a, pert_begin, pert_end).to(device)
if configs['one_hot'] and configs['train_type'] == "train_type_modular_corruption":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/{}/onehot_cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_4_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['corruption_type'], configs['corruption_start_epoch'], configs['corruption_noise'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])
elif configs['one_hot'] and configs['train_type'] == "train_type_modular":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/{}/onehot/n_neurons_4_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])

else:
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/{}/n_neurons_4_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])

# path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/train_type_modular/onehot/n_neurons_4_random_seed_1/n_epochs_30_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_0.10_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'


if args.best:
    ckpt = torch.load(path + '/best_model.pth')#, map_location="cpu")
elif args.last:
    ckpt = torch.load(path + '/last_model.pth')#, map_location="cpu")

model.load_state_dict(ckpt)
model.eval()

# 2) Define a single‐step update function
#    This assumes your model.forward takes inputs of shape (batch, T, features)
#    and returns (outputs, hidden_states). If you have a dedicated one-step
#    method, use that instead.
def step(h, x):
    with torch.no_grad():
        # The model doesn't support h_init, so we need to modify the approach
        # Option 1: Create a custom forward method that accepts initial state
        # Option 2: Use the RNN cell directly
        h_in = h.unsqueeze(0)  # (1, hidden_size)
        x_in = x.unsqueeze(0)  # (1, input_size)
        
        # Use the RNN cell directly for single step
        h_next = model.rnn_cell(x_in, h_in)
        return h_next.squeeze(0)

# 3) Collect a bunch of hidden‐state samples (e.g. from test set) to fit PCA
#    Here you'd load your test stimuli, run them through the RNN, and stack all h_t.
#    For brevity, assume `H_samples` is an (N, hidden_size) array you've gathered.
# Generate hidden states from test data or create synthetic data
# This would require loading test data and running it through the model

def generate_state_sequence(model, test_stimuli_path="dual_alm_rnn_data/test", 
                          use_perturbations=False, pert_type="left"):
    """
    Generate hidden state sequences from test stimuli for PCA fitting.
    
    Args:
        model: The trained RNN model
        test_stimuli_path: Path to test data directory
        use_perturbations: Whether to use perturbation trials
        pert_type: Type of perturbation ("left", "right", or "both")
        
    Returns:
        H_samples: numpy array of shape (N, hidden_size) where N = successful_trials × delay_timesteps
    """
    import torch
    from torch.utils import data
    
    # Load test data
    test_sensory_inputs = np.load(os.path.join(test_stimuli_path, 'sensory_inputs.npy'))
    test_trial_type_labels = np.load(os.path.join(test_stimuli_path, 'trial_type_labels.npy'))
    
    # Create data loader
    test_set = data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False)
    
    # Set model to eval mode
    model.eval()
    
    # Configure perturbations
    if use_perturbations:
        model.uni_pert_trials_prob = 1.0  # Enable perturbations
        if pert_type == "left":
            model.left_alm_pert_prob = 1.0
        elif pert_type == "right":
            model.left_alm_pert_prob = 0.0
        elif pert_type == "both":
            model.left_alm_pert_prob = 0.5
    else:
        model.uni_pert_trials_prob = 0.0  # No perturbations
    
    # Calculate delay period timesteps
    # Based on the experiment settings in dual_alm_rnn_exp.py
    trial_begin_t = -3100  # ms
    delay_begin_t = -1700  # ms
    t_step = 25  # ms
    delay_begin = (delay_begin_t - trial_begin_t) // t_step
    
    total_hs = []
    total_labels = []
    total_pred_labels = []
    
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(test_loader):
            inputs, labels = data_batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            hs, zs = model(inputs)  # hs: (batch_size, T, n_neurons), zs: (batch_size, T, 2)
            
            # Move to CPU and convert to numpy
            hs_np = hs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Get predictions (using final timestep)
            preds_left_alm = (zs[:,-1,0] >= 0).long().detach().cpu().numpy()
            preds_right_alm = (zs[:,-1,1] >= 0).long().detach().cpu().numpy()
            
            # Identify successful trials (both hemispheres agree and match true label)
            agree_mask = (preds_left_alm == preds_right_alm)
            pred_labels = np.zeros_like(preds_left_alm)
            pred_labels[agree_mask] = preds_left_alm[agree_mask]
            pred_labels[~agree_mask] = -1
            
            # Filter for successful trials
            success_mask = (labels_np == pred_labels) & (pred_labels != -1)
            
            if np.any(success_mask):
                # Extract delay period only from successful trials
                hs_delay = hs_np[success_mask, delay_begin:, :]  # (n_successful, delay_timesteps, n_neurons)
                
                # Reshape to (n_successful * delay_timesteps, n_neurons)
                n_successful, delay_timesteps, n_neurons = hs_delay.shape
                hs_flat = hs_delay.reshape(-1, n_neurons)
                
                total_hs.append(hs_flat)
                total_labels.extend(labels_np[success_mask])
                total_pred_labels.extend(pred_labels[success_mask])
    
    # Concatenate all batches
    if total_hs:
        H_samples = np.concatenate(total_hs, axis=0)  # (N, hidden_size)
    else:
        # If no successful trials, return empty array
        H_samples = np.array([]).reshape(0, model.n_neurons)
    
    return H_samples

# Generate hidden states from test data (control trials, no perturbations)
# H_samples = generate_state_sequence(model, use_perturbations=False)

# Alternative: Generate hidden states with perturbations
# H_samples = generate_state_sequence(model, use_perturbations=True, pert_type="left")

def generate_readout_state_sequence(model, test_stimuli_path="dual_alm_rnn_data/test", 
                                   use_perturbations=False, pert_type="left"):
    """
    Generate readout value sequences for left and right hemispheres.
    
    Args:
        model: The trained RNN model
        test_stimuli_path: Path to test data directory
        use_perturbations: Whether to use perturbation trials
        pert_type: Type of perturbation ("left", "right", or "both")
        
    Returns:
        H_samples_left_readout: Readout values for left hemisphere
        H_samples_right_readout: Readout values for right hemisphere
    """
    import torch
    from torch.utils import data
    
    # Load test data
    test_sensory_inputs = np.load(os.path.join(test_stimuli_path, 'onehot_sensory_inputs_simple.npy'))
    test_trial_type_labels = np.load(os.path.join(test_stimuli_path, 'onehot_trial_type_labels_simple.npy'))
    
    # Create data loader
    test_set = data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))
    test_loader = data.DataLoader(test_set, batch_size=model.configs['bs'], shuffle=False)
    
    # Set model to eval mode
    model.eval()
    
    # Configure perturbations
    if use_perturbations:
        model.uni_pert_trials_prob = 1.0
        if pert_type == "left":
            model.left_alm_pert_prob = 1.0
        elif pert_type == "right":
            model.left_alm_pert_prob = 0.0
        elif pert_type == "both":
            model.left_alm_pert_prob = 0.5
    else:
        model.uni_pert_trials_prob = 0.0
    
    # Calculate delay period timesteps
    trial_begin_t = -3100  # ms
    delay_begin_t = -1700  # ms
    t_step = 25  # ms
    delay_begin = (delay_begin_t - trial_begin_t) // t_step
    
    # Get readout weights from the model
    readout_left_weight = model.readout_linear.weight.data.cpu().numpy()[0, :model.n_neurons//2]  # (1, n_neurons//2)
    readout_right_weight = model.readout_linear.weight.data.cpu().numpy()[0, model.n_neurons//2:]  # (1, n_neurons//2)
    
    total_left_readouts = []
    total_right_readouts = []
    
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(test_loader):
            inputs, labels = data_batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            hs, zs = model(inputs)
            
            # Move to CPU and convert to numpy
            hs_np = hs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Get predictions
            # preds_left_alm = (zs[:,-1,0] >= 0).long().detach().cpu().numpy()
            # preds_right_alm = (zs[:,-1,1] >= 0).long().detach().cpu().numpy()

            all_preds = (zs[:,-1,0] >= 0).long().detach().cpu().numpy()
            
            # Identify successful trials (hemispheres agree and are correct)
            # agree_mask = (preds_left_alm == preds_right_alm)
            # pred_labels = np.zeros_like(preds_left_alm)
            # pred_labels[agree_mask] = preds_left_alm[agree_mask]
            # pred_labels[~agree_mask] = -1
            
            # Filter for successful trials
            # success_mask = (labels_np == pred_labels) & (pred_labels != -1)
            success_mask = (labels_np == all_preds)
            # import pdb; pdb.set_trace()
            if np.any(success_mask):
                # Extract delay period only from successful trials
                hs_delay = hs_np[success_mask, delay_begin:, :]  # (n_successful, delay_timesteps, n_neurons)
                
                # Compute readout values for each hemisphere
                left_hemi_hs = hs_delay[...,:model.n_neurons//2]  # Left hemisphere neurons
                right_hemi_hs = hs_delay[...,model.n_neurons//2:]  # Right hemisphere neurons
                
                # Compute readout projections
                left_readouts = left_hemi_hs.dot(readout_left_weight.flatten())  # (n_successful, delay_timesteps)
                right_readouts = right_hemi_hs.dot(readout_right_weight.flatten())  # (n_successful, delay_timesteps)
                
                # Flatten
                left_readouts_flat = left_readouts.reshape(-1)
                right_readouts_flat = right_readouts.reshape(-1)
                
                total_left_readouts.append(left_readouts_flat)
                total_right_readouts.append(right_readouts_flat)
    
    # Concatenate all batches
    if total_left_readouts:
        H_samples_left_readout = np.concatenate(total_left_readouts, axis=0)
        H_samples_right_readout = np.concatenate(total_right_readouts, axis=0)
    else:
        H_samples_left_readout = np.array([])
        H_samples_right_readout = np.array([])
    
    return H_samples_left_readout, H_samples_right_readout

# Generate readout sequences instead of CD projections
H_samples_left_readout, H_samples_right_readout = generate_readout_state_sequence(model, use_perturbations=False)

# 4) Make a 2D grid in readout space (instead of CD projection space)
grid_size = 20

# Use actual range of readout values
left_min, left_max = H_samples_left_readout.min(), H_samples_left_readout.max()
right_min, right_max = H_samples_right_readout.min(), H_samples_right_readout.max()

# Add some padding to the range
left_range = left_max - left_min
right_range = right_max - right_min
left_pad = left_range * 0.1
right_pad = right_range * 0.1

left_vals = np.linspace(left_min - left_pad, left_max + left_pad, grid_size)
right_vals = np.linspace(right_min - right_pad, right_max + right_pad, grid_size)
XX, YY = np.meshgrid(left_vals, right_vals)
grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (grid_size^2, 2)

# 5) Approach 1: Use step function and project onto CDs
# print("Computing flow field using Approach 1: Step function + CD projection")
# vectors_approach1 = []
# x_zero = torch.zeros(1).to(device)

# for uv in grid_pts:
#     # Convert CD projection coordinates back to full hidden state space
#     left_cd_proj, right_cd_proj = uv[0], uv[1]
    
#     # Reconstruct full hidden state from CD projections
#     # Add back the decision boundaries
#     left_cd_proj_centered = left_cd_proj + cd_dbs[0]
#     right_cd_proj_centered = right_cd_proj + cd_dbs[1]
    
#     # Reconstruct hidden state: h = left_cd_proj * cd_left + right_cd_proj * cd_right
#     n_neurons = model.n_neurons
#     h0 = np.zeros(n_neurons)
#     h0[:n_neurons//2] = left_cd_proj_centered * cds[0]  # Left hemisphere
#     h0[n_neurons//2:] = right_cd_proj_centered * cds[1]  # Right hemisphere
    
#     # Convert to tensor and move to device
#     h0_tensor = torch.from_numpy(h0).float().to(device)
    
#     # Step forward
#     h1_tensor = step(h0_tensor, x_zero)
    
#     # Project back to CD space
#     h1_np = h1_tensor.detach().cpu().numpy()
#     left_hemi_h1 = h1_np[:n_neurons//2]
#     right_hemi_h1 = h1_np[n_neurons//2:]
    
#     left_cd_proj_new = left_hemi_h1.dot(cds[0]) - cd_dbs[0]
#     right_cd_proj_new = right_hemi_h1.dot(cds[1]) - cd_dbs[1]
    
#     # Compute velocity vector
#     dh = np.array([left_cd_proj_new - left_cd_proj, right_cd_proj_new - right_cd_proj])
#     vectors_approach1.append(dh)

# vectors_approach1 = np.stack(vectors_approach1, axis=0)
# U1 = vectors_approach1[:, 0].reshape(XX.shape)
# V1 = vectors_approach1[:, 1].reshape(YY.shape)

# 6) Approach 2: Direct hemisphere readout change (modified)
print("Computing flow field using Approach 2: Direct hemisphere readout change")
vectors_approach2 = []

# Get readout weights from the model
readout_left_weight = model.readout_linear.weight.data.cpu().numpy()[0, :model.n_neurons//2]  # (1, n_neurons//2)
readout_right_weight = model.readout_linear.weight.data.cpu().numpy()[0, model.n_neurons//2:]  # (1, n_neurons//2)

for uv in grid_pts:
    left_readout_target, right_readout_target = uv[0], uv[1]
    
    # Instead of reconstructing from CD projections, we need to find hidden states
    # that would produce these readout values. This is more complex but we can approximate
    # by using the readout weights to "reverse engineer" the hidden states
    
    # Method 1: Simple approximation - assume the hidden states are proportional to readout weights
    # This is not exact but gives a reasonable starting point
    n_neurons = model.n_neurons
    
    # Scale the readout weights by the target readout values
    left_hemi_h0 = readout_left_weight.flatten() * left_readout_target / np.linalg.norm(readout_left_weight.flatten())**2
    right_hemi_h0 = readout_right_weight.flatten() * right_readout_target / np.linalg.norm(readout_right_weight.flatten())**2
    
    # Combine into full hidden state
    h0 = np.zeros(n_neurons)
    h0[:n_neurons//2] = left_hemi_h0
    h0[n_neurons//2:] = right_hemi_h0
    
    # Convert to tensor
    h0_tensor = torch.from_numpy(h0).float().to(device)
    x_zero_tensor = torch.zeros(1).to(device)
    
    # Use RNN cell directly for one step
    h1_tensor = model.rnn_cell(x_zero_tensor, h0_tensor.unsqueeze(0)).squeeze(0)
    
    # Compute new readout values
    h1_np = h1_tensor.detach().cpu().numpy()
    left_hemi_h1 = h1_np[:n_neurons//2]
    right_hemi_h1 = h1_np[n_neurons//2:]
    
    left_readout_new = left_hemi_h1.dot(readout_left_weight.flatten())
    right_readout_new = right_hemi_h1.dot(readout_right_weight.flatten())
    
    # Compute velocity vector as change in readout values
    dh = np.array([left_readout_new - left_readout_target, right_readout_new - right_readout_target])
    vectors_approach2.append(dh)

vectors_approach2 = np.stack(vectors_approach2, axis=0)
U2 = vectors_approach2[:, 0].reshape(XX.shape)
V2 = vectors_approach2[:, 1].reshape(YY.shape)

# 7) Plot both approaches
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

f=plt.figure(figsize=(8, 6))
ax2=f.add_subplot(1, 1, 1)
# Plot Approach 1
# ax1.quiver(XX, YY, U1, V1, angles="xy", scale_units="xy", scale=1)
# ax1.set_xlabel("Left Hemisphere CD Projection")
# ax1.set_ylabel("Right Hemisphere CD Projection")
# ax1.set_title("Approach 1: Step Function + CD Projection")
# ax1.axhline(0, color="k", lw=0.5, linestyle='--')
# ax1.axvline(0, color="k", lw=0.5, linestyle='--')
# ax1.grid(True, alpha=0.3)

# Plot Approach 2
ax2.quiver(XX, YY, U2, V2, angles="xy", scale_units="xy", scale=1)
ax2.set_xlabel("Left Hemisphere Readout")
ax2.set_ylabel("Right Hemisphere Readout")
ax2.set_title("Approach 2: Direct Hemisphere Readout Change")
ax2.axhline(0, color="k", lw=0.5, linestyle='--')
ax2.axvline(0, color="k", lw=0.5, linestyle='--')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/flow_field_plot_small_rnn_{}_{}_L{}_R{}.pdf'.format(configs['train_type'], configs['random_seed'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp']))
plt.show()

# Also plot the data points used to generate the readouts
# plt.figure(figsize=(8, 6))
# plt.scatter(H_samples_left_readout, H_samples_right_readout, alpha=0.5, s=1)
# plt.xlabel("Left Hemisphere Readout")
# plt.ylabel("Right Hemisphere Readout")
# plt.title("Readout Values from Successful Trials (Delay Period)")
# plt.axhline(0, color="k", lw=0.5, linestyle='--')
# plt.axvline(0, color="k", lw=0.5, linestyle='--')
# plt.grid(True, alpha=0.3)
# plt.show()