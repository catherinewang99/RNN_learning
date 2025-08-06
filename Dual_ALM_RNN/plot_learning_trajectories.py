import numpy as np
import matplotlib.pyplot as plt 
from dual_alm_rnn_exp import DualALMRNNExp
import json

plt.rcParams['pdf.fonttype'] = '42' 
exp = DualALMRNNExp()


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


# PLOT FOR ONE SEED
results_dict = np.load(path + 'all_val_results_dict.npy', allow_pickle=True)
# Cd readout accuracy when sides don't agree

f,ax=plt.subplots(1,2,figsize=(10,5), sharey='row')
for i in range(len(results_dict)):
    ax[0].scatter(results_dict[i]['control']['cd_accuracy_left'], results_dict[i]['control']['readout_accuracy_left'], color='r', alpha=0.5)
    ax[1].scatter(results_dict[i]['control']['cd_accuracy_right'], results_dict[i]['control']['readout_accuracy_right'], color='b', alpha=0.5)

ax[0].set_title('L hemi')
ax[1].set_title('R hemi')
ax[0].set_xlabel('CD Accuracy')
ax[0].set_ylabel('Readout Accuracy')
plt.suptitle('Control Accuracy')
plt.show()

f=plt.figure()
for i in range(len(results_dict)):
    plt.scatter(results_dict[i]['control']['readout_accuracy_left'], results_dict[i]['control']['readout_accuracy_right'], color='black', alpha=0.5)

# Compute min and max for the scatter plot axes
all_left = [results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))]
all_right = [results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))]
min_val = min(np.min(all_left), np.min(all_right))
max_val = max(np.max(all_left), np.max(all_right))
plt.plot([min_val, max_val], [min_val, max_val], 'k:', lw=2)  # add dotted diagonal line
plt.xlabel('Readout Accuracy Left')
plt.ylabel('Readout Accuracy Right')
plt.title('Control Accuracy')
plt.show()





if False:


    #%%
    #PLOT FOR ALL

    results_dict = np.load('all_val_results_dict.npy', allow_pickle=True)

    # look at left/right readout accuracy over learning
    f=plt.figure()
    for i in range(len(results_dict)):
        plt.scatter(results_dict[i]['control']['readout_accuracy_left'], results_dict[i]['control']['readout_accuracy_right'], color='black', alpha=0.5)

    # Compute min and max for the scatter plot axes
    all_left = [results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))]
    all_right = [results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))]
    min_val = min(np.min(all_left), np.min(all_right))
    max_val = max(np.max(all_left), np.max(all_right))
    plt.plot([min_val, max_val], [min_val, max_val], 'k:', lw=2)  # add dotted diagonal line
    plt.xlabel('Readout Accuracy Left')
    plt.ylabel('Readout Accuracy Right')
    plt.title('Control Accuracy')
    plt.savefig('figs/control_readout_accuracy_over_learning.pdf')
    plt.show()


    # Correlate cd accuracy with readout accuracy for each epoch within hemisphere for control condition (scatter plot)

    f,ax=plt.subplots(1,2,figsize=(10,5), sharey='row')
    for i in range(len(results_dict)):
        ax[0].scatter(results_dict[i]['control']['cd_accuracy_left'], results_dict[i]['control']['readout_accuracy_left'], color='r', alpha=0.5)
        ax[1].scatter(results_dict[i]['control']['cd_accuracy_right'], results_dict[i]['control']['readout_accuracy_right'], color='b', alpha=0.5)

    ax[0].set_title('L hemi')
    ax[1].set_title('R hemi')
    ax[0].set_xlabel('CD Accuracy')
    ax[0].set_ylabel('Readout Accuracy')
    plt.suptitle('Control Accuracy')
    plt.savefig('figs/corr_cd_readout_control.pdf')
    plt.show()



    ## READOUT ACCURACY
    # Control accuracy
    f,ax=plt.subplots(1,2,figsize=(10,5), sharey='row')
    for i in range(len(results_dict)):
        ax[0].scatter(np.ones(50)*i, results_dict[i]['control']['readout_accuracy_left'], color='r', alpha=0.5)
        ax[1].scatter(np.ones(50)*i, results_dict[i]['control']['readout_accuracy_right'], color='b', alpha=0.5)
    ax[0].plot(range(10), [np.mean(results_dict[i]['control']['readout_accuracy_left'], axis=0) for i in range(len(results_dict))], label='Left ALM', ls='-', color='r', linewidth=3)
    ax[1].plot(range(10), [np.mean(results_dict[i]['control']['readout_accuracy_right'], axis=0) for i in range(len(results_dict))], label='Right ALM', ls='-', color='b', linewidth=3)

    for j in range(50):
        ax[0].plot(range(10), [results_dict[i]['control']['readout_accuracy_left'][j] for i in range(len(results_dict))], color='r', linewidth=0.7, alpha=0.5)
        ax[1].plot(range(10), [results_dict[i]['control']['readout_accuracy_right'][j] for i in range(len(results_dict))], color='b', linewidth=0.7, alpha=0.5)

    ax[0].set_title('L hemi')
    ax[1].set_title('R hemi')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    plt.suptitle('Control Accuracy')
    plt.savefig('figs/control_readoutaccuracy.pdf')
    plt.show()


    # Plot accuracy on perturbation trials for left and right hemi, left perturbation

    f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    for i in range(len(results_dict)):
        ax[0].scatter(np.ones(50)*i, results_dict[i]['left_alm_pert']['readout_accuracy_left'], color='r', alpha=0.5)
        ax[1].scatter(np.ones(50)*i, results_dict[i]['left_alm_pert']['readout_accuracy_right'], color='b', alpha=0.5)
    ax[0].plot(range(10), [np.mean(results_dict[i]['left_alm_pert']['readout_accuracy_left'], axis=0) for i in range(len(results_dict))], label='Left ALM', ls='-', color='r', linewidth=3)
    ax[1].plot(range(10), [np.mean(results_dict[i]['left_alm_pert']['readout_accuracy_right'], axis=0) for i in range(len(results_dict))], label='Right ALM', ls='-', color='b', linewidth=3)

    for j in range(50):
        ax[0].plot(range(10), [results_dict[i]['left_alm_pert']['readout_accuracy_left'][j] for i in range(len(results_dict))], color='r', linewidth=0.7, alpha=0.5)
        ax[1].plot(range(10), [results_dict[i]['left_alm_pert']['readout_accuracy_right'][j] for i in range(len(results_dict))], color='b', linewidth=0.7, alpha=0.5)

    ax[0].set_title('L hemi (Left Pert)')
    ax[1].set_title('R hemi (Left Pert)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    plt.suptitle('Left Perturbation Accuracy')
    plt.savefig('figs/left_alm_pert_readout_accuracy.pdf')
    plt.show()

    # Plot accuracy on perturbation trials for left and right hemi, right perturbation

    f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    for i in range(len(results_dict)):
        ax[0].scatter(np.ones(50)*i, results_dict[i]['right_alm_pert']['readout_accuracy_left'], color='r', alpha=0.5)
        ax[1].scatter(np.ones(50)*i, results_dict[i]['right_alm_pert']['readout_accuracy_right'], color='b', alpha=0.5)
    ax[0].plot(range(10), [np.mean(results_dict[i]['right_alm_pert']['readout_accuracy_left'], axis=0) for i in range(len(results_dict))], label='Left ALM', ls='-', color='r', linewidth=3)
    ax[1].plot(range(10), [np.mean(results_dict[i]['right_alm_pert']['readout_accuracy_right'], axis=0) for i in range(len(results_dict))], label='Right ALM', ls='-', color='b', linewidth=3)

    for j in range(50):
        ax[0].plot(range(10), [results_dict[i]['right_alm_pert']['readout_accuracy_left'][j] for i in range(len(results_dict))], color='r', linewidth=0.7, alpha=0.5)
        ax[1].plot(range(10), [results_dict[i]['right_alm_pert']['readout_accuracy_right'][j] for i in range(len(results_dict))], color='b', linewidth=0.7, alpha=0.5)

    ax[0].set_title('L hemi (Right Pert)')
    ax[1].set_title('R hemi (Right Pert)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    plt.suptitle('Right Perturbation Accuracy')
    plt.savefig('figs/right_alm_pert_readout_accuracy.pdf')
    plt.show()












    ## CD ACCURACY
    # Control accuracy
    f,ax=plt.subplots(1,2,figsize=(10,5), sharey='row')
    for i in range(len(results_dict)):
        ax[0].scatter(np.ones(50)*i, results_dict[i]['control']['cd_accuracy_left'], color='r', alpha=0.5)
        ax[1].scatter(np.ones(50)*i, results_dict[i]['control']['cd_accuracy_right'], color='b', alpha=0.5)
    ax[0].plot(range(10), [np.mean(results_dict[i]['control']['cd_accuracy_left'], axis=0) for i in range(len(results_dict))], label='Left ALM', ls='-', color='r', linewidth=3)
    ax[1].plot(range(10), [np.mean(results_dict[i]['control']['cd_accuracy_right'], axis=0) for i in range(len(results_dict))], label='Right ALM', ls='-', color='b', linewidth=3)

    for j in range(50):
        ax[0].plot(range(10), [results_dict[i]['control']['cd_accuracy_left'][j] for i in range(len(results_dict))], color='r', linewidth=0.7, alpha=0.5)
        ax[1].plot(range(10), [results_dict[i]['control']['cd_accuracy_right'][j] for i in range(len(results_dict))], color='b', linewidth=0.7, alpha=0.5)

    ax[0].set_title('L hemi')
    ax[1].set_title('R hemi')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    plt.suptitle('Control Accuracy')
    plt.savefig('figs/control_accuracy.pdf')
    plt.show()


    # Plot accuracy on perturbation trials for left and right hemi, left perturbation

    f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    for i in range(len(results_dict)):
        ax[0].scatter(np.ones(50)*i, results_dict[i]['left_alm_pert']['cd_accuracy_left'], color='r', alpha=0.5)
        ax[1].scatter(np.ones(50)*i, results_dict[i]['left_alm_pert']['cd_accuracy_right'], color='b', alpha=0.5)
    ax[0].plot(range(10), [np.mean(results_dict[i]['left_alm_pert']['cd_accuracy_left'], axis=0) for i in range(len(results_dict))], label='Left ALM', ls='-', color='r', linewidth=3)
    ax[1].plot(range(10), [np.mean(results_dict[i]['left_alm_pert']['cd_accuracy_right'], axis=0) for i in range(len(results_dict))], label='Right ALM', ls='-', color='b', linewidth=3)

    for j in range(50):
        ax[0].plot(range(10), [results_dict[i]['left_alm_pert']['cd_accuracy_left'][j] for i in range(len(results_dict))], color='r', linewidth=0.7, alpha=0.5)
        ax[1].plot(range(10), [results_dict[i]['left_alm_pert']['cd_accuracy_right'][j] for i in range(len(results_dict))], color='b', linewidth=0.7, alpha=0.5)

    ax[0].set_title('L hemi (Left Pert)')
    ax[1].set_title('R hemi (Left Pert)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    plt.suptitle('Left Perturbation Accuracy')
    plt.savefig('figs/left_alm_pert_accuracy.pdf')
    plt.show()

    # Plot accuracy on perturbation trials for left and right hemi, right perturbation

    f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    for i in range(len(results_dict)):
        ax[0].scatter(np.ones(50)*i, results_dict[i]['right_alm_pert']['cd_accuracy_left'], color='r', alpha=0.5)
        ax[1].scatter(np.ones(50)*i, results_dict[i]['right_alm_pert']['cd_accuracy_right'], color='b', alpha=0.5)
    ax[0].plot(range(10), [np.mean(results_dict[i]['right_alm_pert']['cd_accuracy_left'], axis=0) for i in range(len(results_dict))], label='Left ALM', ls='-', color='r', linewidth=3)
    ax[1].plot(range(10), [np.mean(results_dict[i]['right_alm_pert']['cd_accuracy_right'], axis=0) for i in range(len(results_dict))], label='Right ALM', ls='-', color='b', linewidth=3)

    for j in range(50):
        ax[0].plot(range(10), [results_dict[i]['right_alm_pert']['cd_accuracy_left'][j] for i in range(len(results_dict))], color='r', linewidth=0.7, alpha=0.5)
        ax[1].plot(range(10), [results_dict[i]['right_alm_pert']['cd_accuracy_right'][j] for i in range(len(results_dict))], color='b', linewidth=0.7, alpha=0.5)

    ax[0].set_title('L hemi (Right Pert)')
    ax[1].set_title('R hemi (Right Pert)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    plt.suptitle('Right Perturbation Accuracy')
    plt.savefig('figs/right_alm_pert_accuracy.pdf')
    plt.show()
