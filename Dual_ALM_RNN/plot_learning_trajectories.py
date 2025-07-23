import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['pdf.fonttype'] = '42' 

# PLOT FOR ONE SEED
results_dict = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular/n_neurons_256_random_seed_{}/n_epochs_10_n_epochs_across_hemi_10/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(exp.configs['random_seed']), allow_pickle=True)




#PLOT FOR ALL

results_dict = np.load('all_val_results_dict.npy', allow_pickle=True)

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
