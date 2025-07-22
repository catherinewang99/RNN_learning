"""
Test the performance under perturbation conditions for various # of epochs of training
"""

import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
from test_perturbation_eval import test_perturbation_evaluation
import matplotlib.pyplot as plt

cd_acc_left, cd_acc_right = [], []
cd_acc_rightpert_left, cd_acc_rightpert_right = [], []
control_acc_left, control_acc_right = [], []

# Initialize the experiment
exp = DualALMRNNExp()
for epochs in [10, 20, 30, 40, 50]:
    exp.configs['n_epochs'] = epochs
    exp.configs['across_hemi_n_epochs'] = 10

    # Check if model already exists, skip training if so
    model_type = exp.configs['model_type']
    exp.init_sub_path(exp.configs['train_type'])
    model_save_path = os.path.join(exp.configs['models_dir'], model_type, exp.sub_path)
    model_path = os.path.join(model_save_path, 'best_model.pth')
    if os.path.exists(model_path):
        print(f"Model for {epochs} epochs already exists at {model_path}, skipping training.")
    else:
        print(f"Training model for {epochs} epochs...")
        exp.train_type_modular()

    # Generate dataset if it doesn't exist
    if not os.path.exists(exp.configs['data_dir']):
        print("Generating dataset...")
        exp.generate_dataset()

    # Set up device
    use_cuda = bool(exp.configs['use_cuda'])
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Data loading parameters
    if use_cuda:
        params = {'batch_size': exp.configs['bs'], 'shuffle': False, 'num_workers': exp.configs['num_workers'], 
        'pin_memory': bool(exp.configs['pin_memory'])}
    else:
        params = {'batch_size': exp.configs['bs'], 'shuffle': False}

    # Load test data
    test_save_path = os.path.join(exp.configs['data_dir'], 'test')
    test_sensory_inputs = np.load(os.path.join(test_save_path, 'sensory_inputs.npy'))
    test_trial_type_labels = np.load(os.path.join(test_save_path, 'trial_type_labels.npy'))

    test_set = torch.utils.data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))
    test_loader = torch.utils.data.DataLoader(test_set, **params)

    # Load a trained model (you'll need to have trained one first)
    model_type = exp.configs['model_type']
    model_save_path = os.path.join(exp.configs['models_dir'], model_type, exp.sub_path)
    model_path = os.path.join(model_save_path, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train_type_modular()")
        break

    # Load the model
    import sys
    import dual_alm_rnn_models
    model = getattr(dual_alm_rnn_models, model_type)(exp.configs, exp.a, exp.pert_begin, exp.pert_end).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("Model loaded successfully!")



    results = exp.eval_with_perturbations(
        model=model,
        device=device,
        loader=test_loader,
        model_type=model_type,
        n_control=500,  # Limit control trials for speed
        seed=42  # For reproducibility
    )

    cd_acc_left.append(results['left_alm_pert']['cd_accuracy_left'])
    cd_acc_right.append(results['left_alm_pert']['cd_accuracy_right'])
    control_acc_left.append(results['control']['cd_accuracy_left'])
    control_acc_right.append(results['control']['cd_accuracy_right'])
    cd_acc_rightpert_left.append(results['right_alm_pert']['cd_accuracy_left'])
    cd_acc_rightpert_right.append(results['right_alm_pert']['cd_accuracy_right'])

plt.plot(np.mean([control_acc_left,control_acc_right], axis=0), label='Control', ls='--', color='k')
plt.plot(cd_acc_left, label='Left ALM', ls='-', color='r')
plt.plot(cd_acc_right, label='Right ALM', ls='-', color='b')
plt.title('CD Accuracy after Left ALM Perturbation')
plt.xticks([0, 1, 2, 3, 4], [10, 20, 30, 40, 50])
plt.xlabel('Number of Epochs')
plt.ylabel('CD Accuracy')
plt.ylim(0.5, 1)
plt.legend()
plt.show()


plt.plot(np.mean([control_acc_left,control_acc_right], axis=0), label='Control', ls='--', color='k')
plt.plot(cd_acc_rightpert_left, label='Left ALM', ls='-', color='r')
plt.plot(cd_acc_rightpert_right, label='Right ALM', ls='-', color='b')
plt.title('CD Accuracy after Right ALM Perturbation')
plt.xticks([0, 1, 2, 3, 4], [10, 20, 30, 40, 50])
plt.xlabel('Number of Epochs')
plt.ylabel('CD Accuracy')
# plt.ylim(0.5, 1)
plt.legend()
plt.show()