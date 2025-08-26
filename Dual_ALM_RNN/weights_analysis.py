import numpy as np
import matplotlib.pyplot as plt 
from dual_alm_rnn_exp import DualALMRNNExp
import json
import os
from sklearn.decomposition import PCA
# import stats
from numpy.linalg import norm
import seaborn as sns
from collections import Counter
from sklearn.metrics import davies_bouldin_score
import pandas as pd
from sklearn.metrics import silhouette_score

# Code for cluster correlation matrix from https://wil.yegelwel.com/cluster-correlation-matrix/
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))


import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, method = 'complete', inplace=False, both = False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    # linkage = sch.linkage(pairwise_distances, method=method)
    linkage = sch.linkage(corr_array, method=method)
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    # print(cluster_distance_threshold)
    idx = np.argsort(idx_to_cluster_array)
    if len(set(idx_to_cluster_array)) == corr_array.shape[0]:
        sil_score, sil_score1 = 0, 2
    else:
        sil_score = silhouette_score(corr_array, idx_to_cluster_array)
    
        sil_score1 = davies_bouldin_score(corr_array, idx_to_cluster_array)
    
    # sil_score = inconsistent(linkage)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        if both:
            return corr_array.iloc[idx, :].T.iloc[idx, :], idx_to_cluster_array, (sil_score, sil_score1)
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx

def reliability_score(corr):
    """
    Return the reliability score for a neuron across r OR l trials

    Returns
    -------
    Single number.

    """
    _, idmap, _ = cluster_corr(corr, method='complete', both=True)
    n = len(idmap)
    idx = np.argsort(idmap)
    sorted_idmap = idmap[idx]
    
    _, idmap_new, _ = cluster_corr(corr, method='ward', both=True)
    sorted_idmap_new = idmap_new[idx]
    
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if sorted_idmap[i] == sorted_idmap[j]:
                if sorted_idmap_new[i] == sorted_idmap_new[j]:
                    matrix[i,j] = 1
    
    # plt.imshow(matrix)
    # plt.colorbar()
    # plt.show()
    
    score = (np.sum(matrix) - n) / 2
    total_possible = n * n / 2
    
    proportion_clustered = score / total_possible
    return proportion_clustered

def filter_idmap(idmap, minsize=15):
    """
    Take out all the cluster numbers that have fewer than 15 trials in a cluster,
    return shortened version of idmap

    Parameters
    ----------
    idmap : TYPE
        DESCRIPTION.
    minsize : TYPE, optional
        DESCRIPTION. The default is 15.

    Returns
    -------
    idmap_filtered : TYPE
        DESCRIPTION.

    """
    all_clusters = list(set(idmap))
    idmap_filtered = idmap
    for c in all_clusters:
        indices = np.where(idmap_filtered == c)[0]
        if len(indices) < minsize or len(indices) > len(idmap)/2: # Too big or too small
            
            idmap_filtered = np.delete(idmap_filtered, indices)   
            

    return idmap_filtered


plt.rcParams['pdf.fonttype'] = '42' 
exp = DualALMRNNExp()

## Load the weights based on the configs

with open('dual_alm_rnn_configs.json','r') as read_file:
    configs = json.load(read_file)

if configs['one_hot'] and configs['train_type'] == "train_type_modular_corruption":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/onehot_cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['corruption_type'], configs['corruption_start_epoch'], configs['corruption_noise'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])
elif configs['one_hot'] and configs['train_type'] == "train_type_modular":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/onehot/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])
elif configs['train_type'] == "train_type_modular_corruption":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['corruption_type'], configs['corruption_start_epoch'], configs['corruption_noise'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])
else:
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])


# Get all weight files
weight_files = [f for f in os.listdir(path) if f.startswith('weights_epoch_')]
weight_files.sort()

# Plot each of the r to r weight matrices (w_hh_linear_rr) as 10 subplots, each as a heatmap sharing the same colormap
# If the weights are stored in 'weights_epoch_{}.npy' files, load w_hh_linear_rr from each
rr_weights = []
for f in weight_files:  # only first 10 epochs
    data = np.load(os.path.join(path, f))
    # If data is a dict with keys, extract 'w_hh_linear_rr'
    rr_weights.append(data['w_hh_linear_ll'])

# Compute deltas: rr_weights[n+1] - rr_weights[n] for n in 0..len(rr_weights)-2
rr_deltas = [rr_weights[i+1] - rr_weights[i] for i in range(len(rr_weights)-1)]
# Determine global vmin/vmax for colormap across all deltas
delta_min = min([d.min() for d in rr_deltas])
delta_max = max([d.max() for d in rr_deltas])
abs_max = max(abs(delta_min), abs(delta_max))  # for symmetric colormap


cluster_epoch = 6
# Sort weight changes by clusters
corr_array, idx = cluster_corr(rr_deltas[cluster_epoch], method='complete')



# Plot the clusters, sorted by the epoch that corruption starts


# fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
# # Plot cluster_corr(rr_deltas[2]) on the left using axes[0]
# im0 = axes[0].imshow(corr_array, vmin=-abs_max, vmax=abs_max, cmap='bwr')
# axes[0].set_title('Cluster Correlation of Δ Weights')
# fig.colorbar(im0, ax=axes[0], shrink=0.8)
# # Plot rr_deltas[2] on the right using axes[1]
# im1 = axes[1].imshow(rr_deltas[2], vmin=-abs_max, vmax=abs_max, cmap='bwr')
# axes[1].set_title(f'Δ Weights (Epoch {cluster_epoch}→{cluster_epoch+1})')
# fig.colorbar(im1, ax=axes[1], shrink=0.8)
# plt.show()



# Plot by clusters sorted in a specific epoch, but plot all other epochs as well by the same sorting


fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i in range(10):
    ax = axes[i]
    if i < len(rr_deltas):
        im = ax.imshow(rr_deltas[i][idx, :][:, idx], vmin=-abs_max, vmax=abs_max, cmap='bwr', aspect='auto')
        ax.set_title(f'Epoch {i+1} → {i+2}')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Neuron')
    else:
        ax.axis('off')

# Add a single colorbar for all subplots
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Δ Weight Value')

fig.suptitle('Δ w_hh_linear_rr (Epoch n+1 - Epoch n)', fontsize=18)
plt.tight_layout(rect=[0, 0, 0.88, 1])
# Save the figure with one_hot and train_type info in the filename
fig.savefig(f"figs/cluster_sorted_weights_delta_heatmaps_onehot_{configs['one_hot']}_train_{configs['train_type']}_cluster_epoch_{cluster_epoch}.pdf", dpi=300, bbox_inches='tight')

plt.show()





# Order by the size of the weight changes across rows in specific epoch (not very informative)

size_epoch = 1

idx = np.argsort(np.sum(rr_deltas[size_epoch], axis=1))


fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i in range(10):
    ax = axes[i]
    if i < len(rr_deltas):
        im = ax.imshow(rr_deltas[i][idx, :][:, idx], vmin=-abs_max, vmax=abs_max, cmap='bwr', aspect='auto')
        ax.set_title(f'Epoch {i+1} → {i+2}')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Neuron')
    else:
        ax.axis('off')

# Add a single colorbar for all subplots
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Δ Weight Value')

fig.suptitle('Δ w_hh_linear_rr (Epoch n+1 - Epoch n)', fontsize=18)
plt.tight_layout(rect=[0, 0, 0.88, 1])
# Save the figure with one_hot and train_type info in the filename
# fig.savefig(f"figs/cluster_sorted_weights_delta_heatmaps_onehot_{configs['one_hot']}_train_{configs['train_type']}.pdf", dpi=300, bbox_inches='tight')

plt.show()





