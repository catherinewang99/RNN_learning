import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as T 
import torch.nn as nn
import itertools
import math

from sklearn.metrics import accuracy_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dual_alm_rnn_models import *

import json

import scipy

import contextlib
import io

cat = np.concatenate

from scipy import stats

# model.to(device)
# …


import random
def to_float(x):
    # if it’s a tensor, pull it off-device and to a Python float
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    # else assume it’s already a float or numpy scalar
    return float(x)


class DualALMRNNExp(object):

    def __init__(self):

        # Load pred_configs
        with open('dual_alm_rnn_configs.json','r') as read_file:
            self.configs = json.load(read_file)

        # Create directories to save results.
        os.makedirs(self.configs['data_dir'], exist_ok=True)
        os.makedirs(self.configs['logs_dir'], exist_ok=True)
        os.makedirs(self.configs['models_dir'], exist_ok=True)


        self.n_trial_types = 2
        self.n_loc_names = 2
        self.n_loc_names_list = ['left_ALM', 'right_ALM']
        self.loc_name_list = self.n_loc_names_list
        self.n_trial_types_list = range(self.n_trial_types)
        self.one_hot = self.configs['one_hot']

        self.n_neurons = self.configs['n_neurons']
        self.neural_unit_location = np.zeros((self.n_neurons,), dtype=object)

        self.neural_unit_location[:self.n_neurons//2] = 'left_ALM'
        self.neural_unit_location[self.n_neurons//2:] = 'right_ALM'

        self.init_exp_setting()
        self.init_sub_path(self.configs['train_type'])


        logs_save_path = os.path.join(self.configs['logs_dir'], self.configs['model_type'], self.sub_path)
        self.logs_save_path = logs_save_path

    def init_sub_path(self, train_type):
        if train_type == 'train_type_modular_corruption' and not self.one_hot:
            self.sub_path = os.path.join(train_type, 'cor_type_{}_epoch_{}_noise_{:.2f}'.format(self.configs['corruption_type'], self.configs['corruption_start_epoch'], self.configs['corruption_noise']),\
                'n_neurons_{}_random_seed_{}'.format(self.configs['n_neurons'], self.configs['random_seed']),\
                'n_epochs_{}_n_epochs_across_hemi_{}'.format(self.configs['n_epochs'], self.configs['across_hemi_n_epochs']),\
                'lr_{:.1e}_bs_{}'.format(self.configs['lr'], self.configs['bs']),\
                'sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(self.configs['sigma_input_noise'], self.configs['sigma_rec_noise']),\
                'xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}'.format(self.configs['xs_left_alm_amp'], self.configs['xs_right_alm_amp']),\
                'init_cross_hemi_rel_factor_{:.2f}'.format(self.configs['init_cross_hemi_rel_factor']))
        elif self.one_hot and train_type == 'train_type_modular_corruption':
            self.sub_path = os.path.join(train_type, 'onehot_cor_type_{}_epoch_{}_noise_{:.2f}'.format(self.configs['corruption_type'], self.configs['corruption_start_epoch'], self.configs['corruption_noise']),\
                'n_neurons_{}_random_seed_{}'.format(self.configs['n_neurons'], self.configs['random_seed']),\
                'n_epochs_{}_n_epochs_across_hemi_{}'.format(self.configs['n_epochs'], self.configs['across_hemi_n_epochs']),\
                'lr_{:.1e}_bs_{}'.format(self.configs['lr'], self.configs['bs']),\
                'sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(self.configs['sigma_input_noise'], self.configs['sigma_rec_noise']),\
                'xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}'.format(self.configs['xs_left_alm_amp'], self.configs['xs_right_alm_amp']),\
                'init_cross_hemi_rel_factor_{:.2f}'.format(self.configs['init_cross_hemi_rel_factor']))
        elif self.one_hot and train_type == 'train_type_modular':
            self.sub_path = os.path.join(train_type, 'onehot',\
                'n_neurons_{}_random_seed_{}'.format(self.configs['n_neurons'], self.configs['random_seed']),\
                'n_epochs_{}_n_epochs_across_hemi_{}'.format(self.configs['n_epochs'], self.configs['across_hemi_n_epochs']),\
                'lr_{:.1e}_bs_{}'.format(self.configs['lr'], self.configs['bs']),\
                'sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(self.configs['sigma_input_noise'], self.configs['sigma_rec_noise']),\
                'xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}'.format(self.configs['xs_left_alm_amp'], self.configs['xs_right_alm_amp']),\
                'init_cross_hemi_rel_factor_{:.2f}'.format(self.configs['init_cross_hemi_rel_factor']))
        else:
            self.sub_path = os.path.join(train_type, 'n_neurons_{}_random_seed_{}'.format(self.configs['n_neurons'], self.configs['random_seed']),\
                'n_epochs_{}_n_epochs_across_hemi_{}'.format(self.configs['n_epochs'], self.configs['across_hemi_n_epochs']),\
                'lr_{:.1e}_bs_{}'.format(self.configs['lr'], self.configs['bs']),\
                'sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(self.configs['sigma_input_noise'], self.configs['sigma_rec_noise']),\
                'xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}'.format(self.configs['xs_left_alm_amp'], self.configs['xs_right_alm_amp']),\
                'init_cross_hemi_rel_factor_{:.2f}'.format(self.configs['init_cross_hemi_rel_factor']))




    def init_exp_setting(self):


        self.trial_begin_t = -3100 # in ms
        self.sample_begin_t = -3000 # in ms, from the response onset.
        self.delay_begin_t = -1700 # in ms, from the response onset.
        self.total_duration = -self.trial_begin_t

        self.t_step = 25 # in ms
        self.tau = 50 # The neuronal time constant in ms.
        self.a = self.t_step/self.tau


        self.T = self.total_duration//self.t_step + 1
        self.sample_begin = (self.sample_begin_t - self.trial_begin_t)//self.t_step
        self.delay_begin = (self.delay_begin_t - self.trial_begin_t)//self.t_step


        '''
        Uni perturbation
        '''
        self.pert_begin_t = -1700
        self.pert_end_t = -900


        # Perturbation is applied in [pert_begin,pert_end], inclusive at both ends.
        self.pert_begin =  (self.pert_begin_t - self.trial_begin_t)//self.t_step
        self.pert_end = (self.pert_end_t - self.trial_begin_t)//self.t_step



        self.sensory_input_means = np.zeros((self.n_trial_types,))
        self.sensory_input_means[0] = -0.15
        self.sensory_input_means[1] = 0.15

        self.sensory_input_stds = np.zeros((self.n_trial_types,))
        self.sensory_input_stds[0] = 1
        self.sensory_input_stds[1] = 1


        # Convert time from ms to s.

        self.trial_begin_t_in_sec = self.trial_begin_t/1000 
        self.sample_begin_t_in_sec = self.sample_begin_t/1000 
        self.delay_begin_t_in_sec = self.delay_begin_t/1000 
        self.pert_begin_t_in_sec = self.pert_begin_t/1000 
        self.pert_end_t_in_sec = self.pert_end_t/1000 
        self.t_step_in_sec = self.t_step/1000






    '''
    ###
    Dataset generation.
    ###
    '''


    def generate_dataset(self):

        random_seed = self.configs['dataset_random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)



        T = self.T
        sample_begin = self.sample_begin
        delay_begin = self.delay_begin

        presample_mask = np.zeros((T,), dtype=bool)
        presample_mask[:sample_begin] = True
        presample_inds = np.arange(0,sample_begin)

        sample_mask = np.zeros((T,), dtype=bool)
        sample_mask[sample_begin:delay_begin] = True
        sample_inds = np.arange(sample_begin,delay_begin)


        delay_mask = np.zeros((T,), dtype=bool)
        delay_mask[delay_begin:] = True
        delay_inds = np.arange(delay_begin,T)


        n_train_trials = 5000
        n_val_trials = 1000
        n_test_trials = 1000

        sensory_input_means = self.sensory_input_means
        sensory_input_stds = self.sensory_input_stds

        '''
        Generate the train set.
        '''

        train_sensory_inputs = np.zeros((n_train_trials, T, 1), dtype=np.float32)
        train_trial_type_labels = np.zeros((n_train_trials,), dtype=int)

        shuffled_inds = np.random.permutation(n_train_trials)
        train_trial_type_labels[shuffled_inds[:n_train_trials//2]] = 1

        for i in range(self.n_trial_types):
            cur_trial_type_inds = np.nonzero(train_trial_type_labels==i)[0]

            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)

            train_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds)] = \
            sensory_input_means[i] + sensory_input_stds[i]*gaussian_samples


        '''
        Generate the val set.
        '''

        val_sensory_inputs = np.zeros((n_val_trials, T, 1), dtype=np.float32)
        val_trial_type_labels = np.zeros((n_val_trials,), dtype=int)

        shuffled_inds = np.random.permutation(n_val_trials)
        val_trial_type_labels[shuffled_inds[:n_val_trials//2]] = 1

        for i in range(self.n_trial_types):
            cur_trial_type_inds = np.nonzero(val_trial_type_labels==i)[0]

            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)

            val_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds)] = \
            sensory_input_means[i] + sensory_input_stds[i]*gaussian_samples



        '''
        Generate the test set.
        '''

        test_sensory_inputs = np.zeros((n_test_trials, T, 1), dtype=np.float32)
        test_trial_type_labels = np.zeros((n_test_trials,), dtype=int)

        shuffled_inds = np.random.permutation(n_test_trials)
        test_trial_type_labels[shuffled_inds[:n_test_trials//2]] = 1

        for i in range(self.n_trial_types):
            cur_trial_type_inds = np.nonzero(test_trial_type_labels==i)[0]

            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)

            test_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds)] = \
            sensory_input_means[i] + sensory_input_stds[i]*gaussian_samples




        '''
        Save.
        '''
        train_save_path = os.path.join(self.configs['data_dir'], 'train')
        os.makedirs(train_save_path, exist_ok=True)
        np.save(os.path.join(train_save_path, 'sensory_inputs.npy'), train_sensory_inputs)
        np.save(os.path.join(train_save_path, 'trial_type_labels.npy'), train_trial_type_labels)

        val_save_path = os.path.join(self.configs['data_dir'], 'val')
        os.makedirs(val_save_path, exist_ok=True)
        np.save(os.path.join(val_save_path, 'sensory_inputs.npy'), val_sensory_inputs)
        np.save(os.path.join(val_save_path, 'trial_type_labels.npy'), val_trial_type_labels)

        test_save_path = os.path.join(self.configs['data_dir'], 'test')
        os.makedirs(test_save_path, exist_ok=True)
        np.save(os.path.join(test_save_path, 'sensory_inputs.npy'), test_sensory_inputs)
        np.save(os.path.join(test_save_path, 'trial_type_labels.npy'), test_trial_type_labels)


        sample_inds = np.random.permutation(n_train_trials)[:10]
        sample_train_inputs = train_sensory_inputs[sample_inds]
        sample_train_labels = train_trial_type_labels[sample_inds]


        '''
        Sanity check.
        '''

        fig = plt.figure()
        ax = fig.add_subplot(111)


        color = ['r', 'b']
        for i in range(2):
            ax.plot(sample_train_inputs[sample_train_labels==i][...,0].T, c=color[i])
        ax.axvline(self.sample_begin, c='k')
        ax.axvline(self.delay_begin, c='k')

        fig.savefig(os.path.join(train_save_path, 'sample.png'))

        plt.show()





    def generate_dataset_onehot(self):

        random_seed = self.configs['dataset_random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)



        T = self.T
        sample_begin = self.sample_begin
        delay_begin = self.delay_begin

        presample_mask = np.zeros((T,), dtype=bool)
        presample_mask[:sample_begin] = True
        presample_inds = np.arange(0,sample_begin)

        sample_mask = np.zeros((T,), dtype=bool)
        sample_mask[sample_begin:delay_begin] = True
        sample_inds = np.arange(sample_begin,delay_begin)

 
        delay_mask = np.zeros((T,), dtype=bool)
        delay_mask[delay_begin:] = True
        delay_inds = np.arange(delay_begin,T)


        n_train_trials = 5000
        n_val_trials = 1000
        n_test_trials = 1000

        sensory_input_means = self.sensory_input_means
        sensory_input_stds = self.sensory_input_stds

        '''
        Generate the train, val and test set.
        '''

        # Change input dimension from 1 to 2 (one-hot encoding)
        train_sensory_inputs = np.zeros((n_train_trials, T, 2), dtype=np.float32)
        val_sensory_inputs = np.zeros((n_val_trials, T, 2), dtype=np.float32) 
        test_sensory_inputs = np.zeros((n_test_trials, T, 2), dtype=np.float32)

        train_trial_type_labels = np.zeros((n_train_trials,), dtype=int)
        val_trial_type_labels = np.zeros((n_val_trials,), dtype=int)
        test_trial_type_labels = np.zeros((n_test_trials,), dtype=int)

        # Labels of trial type (random)
        shuffled_inds = np.random.permutation(n_train_trials)
        train_trial_type_labels[shuffled_inds[:n_train_trials//2]] = 1
        shuffled_inds = np.random.permutation(n_val_trials)
        val_trial_type_labels[shuffled_inds[:n_val_trials//2]] = 1
        shuffled_inds = np.random.permutation(n_test_trials)
        test_trial_type_labels[shuffled_inds[:n_test_trials//2]] = 1
        
        # For each trial type, set the appropriate one-hot channel
        for i in range(self.n_trial_types):

            # Train labels first
            cur_trial_type_inds = np.nonzero(train_trial_type_labels==i)[0]
            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)
            # Left trials (i=0): [1, 0], Right trials (i=1): [0, 1]
            train_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds, [i])] = \
            sensory_input_means[1] + sensory_input_stds[1]*gaussian_samples # Only use the positive values

            # Val labels
            cur_trial_type_inds = np.nonzero(val_trial_type_labels==i)[0]
            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)
            val_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds, [i])] = \
            sensory_input_means[1] + sensory_input_stds[1]*gaussian_samples # Only use the positive values

            # Test labels
            cur_trial_type_inds = np.nonzero(test_trial_type_labels==i)[0]
            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)
            test_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds, [i])] = \
            sensory_input_means[1] + sensory_input_stds[1]*gaussian_samples # Only use the positive values


        '''
        Save.
        '''
        train_save_path = os.path.join(self.configs['data_dir'], 'train')
        os.makedirs(train_save_path, exist_ok=True)
        np.save(os.path.join(train_save_path, 'onehot_sensory_inputs.npy'), train_sensory_inputs)
        np.save(os.path.join(train_save_path, 'onehot_trial_type_labels.npy'), train_trial_type_labels)

        val_save_path = os.path.join(self.configs['data_dir'], 'val')
        os.makedirs(val_save_path, exist_ok=True)
        np.save(os.path.join(val_save_path, 'onehot_sensory_inputs.npy'), val_sensory_inputs)
        np.save(os.path.join(val_save_path, 'onehot_trial_type_labels.npy'), val_trial_type_labels)

        test_save_path = os.path.join(self.configs['data_dir'], 'test')
        os.makedirs(test_save_path, exist_ok=True)
        np.save(os.path.join(test_save_path, 'onehot_sensory_inputs.npy'), test_sensory_inputs)
        np.save(os.path.join(test_save_path, 'onehot_trial_type_labels.npy'), test_trial_type_labels)


        sample_inds = np.random.permutation(n_train_trials)[:10]
        sample_train_inputs = train_sensory_inputs[sample_inds]
        sample_train_labels = train_trial_type_labels[sample_inds]






    def train_type_uniform(self):

        random_seed = self.configs['random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        model_type = self.configs['model_type']

        self.init_sub_path('train_type_uniform')


        model_save_path = os.path.join(self.configs['models_dir'], model_type, self.sub_path)


        logs_save_path = os.path.join(self.configs['logs_dir'], model_type, self.sub_path)

        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(logs_save_path, exist_ok=True)




        # Detect devices
        use_cuda = bool(self.configs['use_cuda'])
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update

        # device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")

        # Data loading parameters
        if use_cuda:
            params = {'batch_size': self.configs['bs'], 'shuffle': True, 'num_workers': self.configs['num_workers'], \
            'pin_memory': bool(self.configs['pin_memory'])}
        else:
            params = {'batch_size': self.configs['bs'], 'shuffle': True}

        '''
        Load the dataset and wrap it with Pytorch Dataset.
        '''

        # train
        train_save_path = os.path.join(self.configs['data_dir'], 'train')

        train_sensory_inputs = np.load(os.path.join(train_save_path, 'sensory_inputs.npy'))
        train_trial_type_labels = np.load(os.path.join(train_save_path, 'trial_type_labels.npy'))

        train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))

        train_loader = data.DataLoader(train_set, **params, drop_last=True)

        # val
        val_save_path = os.path.join(self.configs['data_dir'], 'val')

        val_sensory_inputs = np.load(os.path.join(val_save_path, 'sensory_inputs.npy'))
        val_trial_type_labels = np.load(os.path.join(val_save_path, 'trial_type_labels.npy'))

        val_set = data.TensorDataset(torch.tensor(val_sensory_inputs), torch.tensor(val_trial_type_labels))

        val_loader = data.DataLoader(val_set, **params)



        '''
        Initialize the model.
        '''

        import sys
        model = getattr(sys.modules[__name__], model_type)(self.configs, \
            self.a, self.pert_begin, self.pert_end).to(device)


        '''
        We only train the recurrent weights.
        '''
        trainable_params = []
        for name, param in model.named_parameters():
            if 'rnn_cell' in name:
                trainable_params.append(param)


        optimizer = optim.Adam(trainable_params, lr=self.configs['lr'], weight_decay=self.configs['l2_weight_decay'])

        loss_fct = nn.BCEWithLogitsLoss()




        '''
        Train the model.
        '''



        all_epoch_train_losses = []
        all_epoch_train_scores = []
        all_epoch_val_losses = []
        all_epoch_val_scores = []

        # Separate lists for across-hemi training
        all_across_train_losses = []
        all_across_train_scores = []
        all_across_val_losses = []
        all_across_val_scores = []

        best_val_score = float('-inf')

        for epoch in range(self.configs['n_epochs']):
            epoch_begin_time = time.time()


            model.uni_pert_trials_prob = self.configs['uni_pert_trials_prob']
            
            train_losses, train_scores = self.train_helper(model, device, train_loader, optimizer, epoch, loss_fct) # Per each training batch.


            val_loss, val_score = self.val_helper(model, device, val_loader, loss_fct) # On the entire val set.

            if val_score > best_val_score:
                best_val_score = val_score
                model_save_name = 'best_model.pth'

                torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))  # save model


            all_epoch_train_losses.extend(train_losses)
            all_epoch_train_scores.extend(train_scores)
            all_epoch_val_losses.append(val_loss)
            all_epoch_val_scores.append(val_score)

            # A = np.array(all_epoch_train_losses)
            # B = np.array(all_epoch_train_scores)
            # C = np.array(all_epoch_val_losses)
            # D = np.array(all_epoch_val_scores)
            # after: pull everything back to CPU and to Python floats
            A = np.array([to_float(t) for t in all_epoch_train_losses])
            B = np.array([to_float(t) for t in all_epoch_train_scores])
            C = np.array([to_float(t) for t in all_epoch_val_losses])
            D = np.array([to_float(t) for t in all_epoch_val_scores])


            np.save(os.path.join(logs_save_path, 'all_epoch_train_losses.npy'), A)
            np.save(os.path.join(logs_save_path, 'all_epoch_train_scores.npy'), B)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_losses.npy'), C)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_scores.npy'), D)

            epoch_end_time = time.time()

            print('Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
            print('')









    def train_type_modular(self):


        random_seed = self.configs['random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


        model_type = self.configs['model_type']


        self.init_sub_path('train_type_modular')


        model_save_path = os.path.join(self.configs['models_dir'], model_type, self.sub_path)


        logs_save_path = os.path.join(self.configs['logs_dir'], self.configs['model_type'], self.sub_path)
        self.logs_save_path = logs_save_path

        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(logs_save_path, exist_ok=True)




        # Detect devices
        use_cuda = bool(self.configs['use_cuda'])
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update

        # device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")

        # Data loading parameters
        if use_cuda:
            params = {'batch_size': self.configs['bs'], 'shuffle': True, 'num_workers': self.configs['num_workers'], \
            'pin_memory': bool(self.configs['pin_memory'])}
        else:
            params = {'batch_size': self.configs['bs'], 'shuffle': True}

        '''
        Load the dataset and wrap it with Pytorch Dataset.
        '''

        # train
        train_save_path = os.path.join(self.configs['data_dir'], 'train')

        train_sensory_inputs = np.load(os.path.join(train_save_path, 'onehot_sensory_inputs.npy' if self.configs['one_hot'] else 'sensory_inputs.npy'))
        train_trial_type_labels = np.load(os.path.join(train_save_path, 'trial_type_labels.npy' if not self.configs['one_hot'] else 'onehot_trial_type_labels.npy'))

        train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))

        train_loader = data.DataLoader(train_set, **params, drop_last=True)

        # val
        val_save_path = os.path.join(self.configs['data_dir'], 'val')

        val_sensory_inputs = np.load(os.path.join(val_save_path, 'onehot_sensory_inputs.npy' if self.configs['one_hot'] else 'sensory_inputs.npy'))
        val_trial_type_labels = np.load(os.path.join(val_save_path, 'trial_type_labels.npy' if not self.configs['one_hot'] else 'onehot_trial_type_labels.npy'))

        val_set = data.TensorDataset(torch.tensor(val_sensory_inputs), torch.tensor(val_trial_type_labels))

        val_loader = data.DataLoader(val_set, **params)

        # load test data for later
        test_save_path = os.path.join(self.configs['data_dir'], 'test')
        test_sensory_inputs = np.load(os.path.join(test_save_path, 'onehot_sensory_inputs.npy' if self.configs['one_hot'] else 'sensory_inputs.npy'))
        test_trial_type_labels = np.load(os.path.join(test_save_path, 'trial_type_labels.npy' if not self.configs['one_hot'] else 'onehot_trial_type_labels.npy'))
        
        test_set = torch.utils.data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))
        test_loader = torch.utils.data.DataLoader(test_set, **params)

        '''
        Initialize the model.
        '''

        import sys
        model = getattr(sys.modules[__name__], model_type)(self.configs, \
            self.a, self.pert_begin, self.pert_end).to(device)



        '''
        We only train the recurrent weights.
        '''
        params_within_hemi = []
        params_cross_hemi = []
        n_neurons = self.configs['n_neurons']


        for name, param in model.named_parameters():
            if ('w_hh_linear_ll' in name) or ('w_hh_linear_rr' in name):
                params_within_hemi.append(param)
            elif ('w_hh_linear_lr' in name) or ('w_hh_linear_rl' in name):
                params_cross_hemi.append(param)


        optimizer_within_hemi = optim.Adam(params_within_hemi, lr=self.configs['lr'], weight_decay=self.configs['l2_weight_decay'])
        optimizer_cross_hemi = optim.Adam(params_cross_hemi, lr=self.configs['lr'], weight_decay=self.configs['l2_weight_decay'])


        loss_fct = nn.BCEWithLogitsLoss()




        '''
        Train the model.
        '''



        all_epoch_train_losses = []
        all_epoch_train_scores = []
        all_epoch_val_losses = []
        all_epoch_val_scores = []

        # Separate lists for across-hemi training
        all_across_train_losses = []
        all_across_train_scores = []
        all_across_val_losses = []
        all_across_val_scores = []

        # Separate lists for every epoch val results
        all_val_results_dict = []

        best_val_score = float('-inf')


        for epoch in range(self.configs['n_epochs']):
            epoch_begin_time = time.time()


            print('')
            print('Within-hemi training')

            model.uni_pert_trials_prob = self.configs['uni_pert_trials_prob']

            train_losses, train_scores = self.train_helper(model, device, train_loader, optimizer_within_hemi, epoch, loss_fct) # Per each training batch.
            total_hs, _ = self.get_neurons_trace(model, device, train_loader, model_type, hemi_type='both', return_pred_labels=False, hemi_agree=False, corrupt=False)
            np.save(os.path.join(logs_save_path, 'all_hs_corruption_epoch_{}.npy'.format(epoch)), total_hs)
              # After optimizer.step() and epoch ends
            val_results = self.eval_with_perturbations(
                model=model,
                device=device,
                loader=test_loader,  # or val_loader
                model_type=model_type,
                n_control=500,
                seed=42
            )
            all_val_results_dict.append(val_results)

            val_loss, val_score = self.val_helper(model, device, val_loader, loss_fct) # On the entire val set.

            if val_score > best_val_score:
                best_val_score = val_score
                model_save_name = 'best_model.pth'

                torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))  # save model


            all_epoch_train_losses.extend(train_losses)
            all_epoch_train_scores.extend(train_scores)
            all_epoch_val_losses.append(val_loss)
            all_epoch_val_scores.append(val_score)

            # A = np.array(all_epoch_train_losses)
            # B = np.array(all_epoch_train_scores)
            # C = np.array(all_epoch_val_losses)
            # D = np.array(all_epoch_val_scores)

            # after: pull everything back to CPU and to Python floats


            A = np.array([to_float(t) for t in all_epoch_train_losses])
            B = np.array([to_float(t) for t in all_epoch_train_scores])
            C = np.array([to_float(t) for t in all_epoch_val_losses])
            D = np.array([to_float(t) for t in all_epoch_val_scores])

            np.save(os.path.join(logs_save_path, 'all_epoch_train_losses.npy'), A)
            np.save(os.path.join(logs_save_path, 'all_epoch_train_scores.npy'), B)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_losses.npy'), C)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_scores.npy'), D)

            # Track weights after each epoch
            self.track_weights(model, epoch, 'within_hemi', logs_save_path)

            epoch_end_time = time.time()

            print('Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
            print('')

        # After training
        left_input_weights = model.w_xh_linear_left_alm.weight.data.cpu().numpy()
        right_input_weights = model.w_xh_linear_right_alm.weight.data.cpu().numpy()
        # For left ALM readout (maps left ALM hidden units to output)
        left_readout_weights = model.readout_linear_left_alm.weight.data.cpu().numpy()  # shape: (n_left_neurons,)
        # For right ALM readout (maps right ALM hidden units to output)
        right_readout_weights = model.readout_linear_right_alm.weight.data.cpu().numpy()  # shape: (n_right_neurons,)

        # Save to file, append to a list, or log as needed
        np.save(os.path.join(logs_save_path, f"input_weights_left_epoch_final.npy"), left_input_weights)
        np.save(os.path.join(logs_save_path, f"input_weights_right_epoch_final.npy"), right_input_weights)
        np.save(os.path.join(logs_save_path, f"readout_weights_left_epoch_final.npy"), left_readout_weights)
        np.save(os.path.join(logs_save_path, f"readout_weights_right_epoch_final.npy"), right_readout_weights)

        np.save(os.path.join(logs_save_path, 'all_val_results_dict.npy'), all_val_results_dict)

        for epoch in range(self.configs['across_hemi_n_epochs']):
            epoch_begin_time = time.time()


            print('')
            print('Across-hemi training')

            # Optionally, you can set a different uni_pert_trials_prob or other config here if needed
            model.uni_pert_trials_prob = self.configs['uni_pert_trials_prob']

            train_losses, train_scores = self.train_helper(model, device, train_loader, optimizer_cross_hemi, epoch, loss_fct)
            val_loss, val_score = self.val_helper(model, device, val_loader, loss_fct)

            # Save best model if desired, or log as needed
            if val_score > best_val_score:
                best_val_score = val_score
                model_save_name = 'best_model_across_hemi.pth'
                torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))

            all_across_train_losses.extend(train_losses)
            all_across_train_scores.extend(train_scores)
            all_across_val_losses.append(val_loss)
            all_across_val_scores.append(val_score)

            A_across = np.array([to_float(t) for t in all_across_train_losses])
            B_across = np.array([to_float(t) for t in all_across_train_scores])
            C_across = np.array([to_float(t) for t in all_across_val_losses])
            D_across = np.array([to_float(t) for t in all_across_val_scores])

            np.save(os.path.join(logs_save_path, 'all_across_train_losses.npy'), A_across)
            np.save(os.path.join(logs_save_path, 'all_across_train_scores.npy'), B_across)
            np.save(os.path.join(logs_save_path, 'all_across_val_losses.npy'), C_across)
            np.save(os.path.join(logs_save_path, 'all_across_val_scores.npy'), D_across)

            # Track weights after each epoch
            self.track_weights(model, epoch, 'across_hemi', logs_save_path)

            epoch_end_time = time.time()
            print('Across-hemi Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
            print('')



       
    def train_type_modular_corruption(self):


        random_seed = self.configs['random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


        model_type = self.configs['model_type']


        self.init_sub_path('train_type_modular_corruption')


        model_save_path = os.path.join(self.configs['models_dir'], model_type, self.sub_path)


        logs_save_path = os.path.join(self.configs['logs_dir'], self.configs['model_type'], self.sub_path)
        self.logs_save_path = logs_save_path

        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(logs_save_path, exist_ok=True)




        # Detect devices
        use_cuda = bool(self.configs['use_cuda'])
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update

        # device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")

        # Data loading parameters
        if use_cuda:
            params = {'batch_size': self.configs['bs'], 'shuffle': True, 'num_workers': self.configs['num_workers'], \
            'pin_memory': bool(self.configs['pin_memory'])}
        else:
            params = {'batch_size': self.configs['bs'], 'shuffle': True}

        '''
        Load the dataset and wrap it with Pytorch Dataset.
        '''

        # train
        train_save_path = os.path.join(self.configs['data_dir'], 'train')
 
        train_sensory_inputs = np.load(os.path.join(train_save_path, 'onehot_sensory_inputs.npy' if self.configs['one_hot'] else 'sensory_inputs.npy'))
        train_trial_type_labels = np.load(os.path.join(train_save_path, 'trial_type_labels.npy' if not self.configs['one_hot'] else 'onehot_trial_type_labels.npy'))

        train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))

        train_loader = data.DataLoader(train_set, **params, drop_last=True)

        # val
        val_save_path = os.path.join(self.configs['data_dir'], 'val')

        val_sensory_inputs = np.load(os.path.join(val_save_path, 'onehot_sensory_inputs.npy' if self.configs['one_hot'] else 'sensory_inputs.npy'))
        val_trial_type_labels = np.load(os.path.join(val_save_path, 'trial_type_labels.npy' if not self.configs['one_hot'] else 'onehot_trial_type_labels.npy'))

        val_set = data.TensorDataset(torch.tensor(val_sensory_inputs), torch.tensor(val_trial_type_labels))

        val_loader = data.DataLoader(val_set, **params)

        # load test data for later
        test_save_path = os.path.join(self.configs['data_dir'], 'test')
        test_sensory_inputs = np.load(os.path.join(test_save_path, 'onehot_sensory_inputs.npy' if self.configs['one_hot'] else 'sensory_inputs.npy'))
        test_trial_type_labels = np.load(os.path.join(test_save_path, 'trial_type_labels.npy' if not self.configs['one_hot'] else 'onehot_trial_type_labels.npy'))
        
        test_set = torch.utils.data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))
        test_loader = torch.utils.data.DataLoader(test_set, **params)

        '''
        Initialize the model.
        '''

        import sys
        model = getattr(sys.modules[__name__], model_type)(self.configs, \
            self.a, self.pert_begin, self.pert_end).to(device)



        '''
        We only train the recurrent weights.
        '''
        params_within_hemi = []
        params_cross_hemi = []
        n_neurons = self.configs['n_neurons']


        for name, param in model.named_parameters():
            if ('w_hh_linear_ll' in name) or ('w_hh_linear_rr' in name):
                params_within_hemi.append(param)
            elif ('w_hh_linear_lr' in name) or ('w_hh_linear_rl' in name):
                params_cross_hemi.append(param)


        optimizer_within_hemi = optim.Adam(params_within_hemi, lr=self.configs['lr'], weight_decay=self.configs['l2_weight_decay'])
        optimizer_cross_hemi = optim.Adam(params_cross_hemi, lr=self.configs['lr'], weight_decay=self.configs['l2_weight_decay'])


        loss_fct = nn.BCEWithLogitsLoss()




        '''
        Train the model.
        '''



        all_epoch_train_losses = []
        all_epoch_train_scores = []
        all_epoch_val_losses = []
        all_epoch_val_scores = []

        # Separate lists for across-hemi training
        all_across_train_losses = []
        all_across_train_scores = []
        all_across_val_losses = []
        all_across_val_scores = []

        # Separate lists for every epoch val results
        all_val_results_dict = []

        best_val_score = float('-inf')


        for epoch in range(self.configs['n_epochs']):

            # Add corruption to the training data
            if epoch >= self.configs['corruption_start_epoch']:
                print('Adding corruption to the training data at epoch {}'.format(epoch))

                model.corrupt = True

                total_hs, _ = self.get_neurons_trace(model, device, train_loader, model_type, hemi_type='both', return_pred_labels=False, hemi_agree=False, corrupt=True)
                np.save(os.path.join(logs_save_path, 'all_hs_corruption_epoch_{}.npy'.format(epoch)), total_hs)

            epoch_begin_time = time.time()


            print('')
            print('Within-hemi training')

            model.uni_pert_trials_prob = self.configs['uni_pert_trials_prob']

            train_losses, train_scores = self.train_helper(model, device, train_loader, optimizer_within_hemi, epoch, loss_fct) # Per each training batch.

              # After optimizer.step() and epoch ends
            val_results = self.eval_with_perturbations(
                model=model,
                device=device,
                loader=test_loader,  # or val_loader
                model_type=model_type,
                n_control=500,
                seed=42
            )
            all_val_results_dict.append(val_results)

            val_loss, val_score = self.val_helper(model, device, val_loader, loss_fct) # On the entire val set.

            if val_score > best_val_score:
                best_val_score = val_score
                model_save_name = 'best_model.pth'

                torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))  # save model


            all_epoch_train_losses.extend(train_losses)
            all_epoch_train_scores.extend(train_scores)
            all_epoch_val_losses.append(val_loss)
            all_epoch_val_scores.append(val_score)

            # A = np.array(all_epoch_train_losses)
            # B = np.array(all_epoch_train_scores)
            # C = np.array(all_epoch_val_losses)
            # D = np.array(all_epoch_val_scores)

            # after: pull everything back to CPU and to Python floats


            A = np.array([to_float(t) for t in all_epoch_train_losses])
            B = np.array([to_float(t) for t in all_epoch_train_scores])
            C = np.array([to_float(t) for t in all_epoch_val_losses])
            D = np.array([to_float(t) for t in all_epoch_val_scores])

            np.save(os.path.join(logs_save_path, 'all_epoch_train_losses.npy'), A)
            np.save(os.path.join(logs_save_path, 'all_epoch_train_scores.npy'), B)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_losses.npy'), C)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_scores.npy'), D)

            # Track weights after each epoch
            self.track_weights(model, epoch, 'within_hemi', logs_save_path)

            epoch_end_time = time.time()

            print('Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
            print('')

        # After training (these weights don't change)
        left_input_weights = model.w_xh_linear_left_alm.weight.data.cpu().numpy()
        right_input_weights = model.w_xh_linear_right_alm.weight.data.cpu().numpy()
        # For left ALM readout (maps left ALM hidden units to output)
        left_readout_weights = model.readout_linear_left_alm.weight.data.cpu().numpy()  # shape: (n_left_neurons,)
        # For right ALM readout (maps right ALM hidden units to output)
        right_readout_weights = model.readout_linear_right_alm.weight.data.cpu().numpy()  # shape: (n_right_neurons,)

        # Save to file, append to a list, or log as needed
        np.save(os.path.join(logs_save_path, f"input_weights_left_epoch_final.npy"), left_input_weights)
        np.save(os.path.join(logs_save_path, f"input_weights_right_epoch_final.npy"), right_input_weights)
        np.save(os.path.join(logs_save_path, f"readout_weights_left_epoch_final.npy"), left_readout_weights)
        np.save(os.path.join(logs_save_path, f"readout_weights_right_epoch_final.npy"), right_readout_weights)

        np.save(os.path.join(logs_save_path, 'all_val_results_dict.npy'), all_val_results_dict)

        for epoch in range(self.configs['across_hemi_n_epochs']):
            epoch_begin_time = time.time()


            print('')
            print('Across-hemi training')

            # Optionally, you can set a different uni_pert_trials_prob or other config here if needed
            model.uni_pert_trials_prob = self.configs['uni_pert_trials_prob']

            train_losses, train_scores = self.train_helper(model, device, train_loader, optimizer_cross_hemi, epoch, loss_fct)
            val_loss, val_score = self.val_helper(model, device, val_loader, loss_fct)

            # Save best model if desired, or log as needed
            if val_score > best_val_score:
                best_val_score = val_score
                model_save_name = 'best_model_across_hemi.pth'
                torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))

            all_across_train_losses.extend(train_losses)
            all_across_train_scores.extend(train_scores)
            all_across_val_losses.append(val_loss)
            all_across_val_scores.append(val_score)

            A_across = np.array([to_float(t) for t in all_across_train_losses])
            B_across = np.array([to_float(t) for t in all_across_train_scores])
            C_across = np.array([to_float(t) for t in all_across_val_losses])
            D_across = np.array([to_float(t) for t in all_across_val_scores])

            np.save(os.path.join(logs_save_path, 'all_across_train_losses.npy'), A_across)
            np.save(os.path.join(logs_save_path, 'all_across_train_scores.npy'), B_across)
            np.save(os.path.join(logs_save_path, 'all_across_val_losses.npy'), C_across)
            np.save(os.path.join(logs_save_path, 'all_across_val_scores.npy'), D_across)

            # Track weights after each epoch
            self.track_weights(model, epoch, 'across_hemi', logs_save_path)

            epoch_end_time = time.time()
            print('Across-hemi Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
            print('')




    def track_weights(self, model, epoch, training_phase, logs_save_path):
        """
        Track weight evolution during training
        
        Args:
            model: the RNN model
            epoch: current epoch number
            training_phase: 'within_hemi' or 'across_hemi'
            logs_save_path: where to save the data
        """
        weight_data = {}
        
        # Extract weight matrices
        for name, param in model.named_parameters():
            # Only save weights, not biases, and strip prefix/suffix
            if 'w_hh_linear' in name and name.endswith('.weight'):
                # name might be 'rnn_cell.w_hh_linear_ll.weight'
                key = name.split('.')[-2]  # gets 'w_hh_linear_ll'
                weight_matrix = param.data.cpu().numpy()
                weight_data[key] = weight_matrix
        
        # Save to file
        save_path = os.path.join(logs_save_path, f'weights_epoch_{epoch}_{training_phase}.npz')
        np.savez(save_path, **weight_data)
        
        print(f'Saved weights for epoch {epoch} ({training_phase})')



    '''
    Add losses randomly after stim period.
    '''
    def train_helper(self, model, device, train_loader, optimizer, epoch, loss_fct):

        model.train()

        losses = []
        scores = []

        trial_count = 0

        begin_time = time.time()
        for batch_idx, data in enumerate(train_loader):

            inputs, labels  = data
            inputs, labels = inputs.to(device), labels.to(device)

            trial_count += len(labels)

            optimizer.zero_grad()


            '''
            hs: (n_trials, T, n_neurons)
            zs: (n_trials, T, 2) # 2 because we have a readout at each hemisphere, unless single readout
            '''
            _, zs = model(inputs)


            assert self.T == inputs.shape[1]

            dec_begin = self.delay_begin            

            # use a single readout model
            if self.configs['model_type'] == 'TwoHemiRNNTanh_single_readout':
                # For single readout, zs is (n_trials, T, 1)
                loss = loss_fct(zs[:,dec_begin:,-1].squeeze(-1), labels.float()[:,None].expand(-1,self.T-dec_begin))
            else:
                # BCEWithLogitsLoss requires that the target be float between 0 and 1.
                loss_left_alm = loss_fct(zs[:,dec_begin:,0], labels.float()[:,None].expand(-1,self.T-dec_begin))
                loss_right_alm = loss_fct(zs[:,dec_begin:,1], labels.float()[:,None].expand(-1,self.T-dec_begin))


                loss = loss_left_alm + loss_right_alm


            loss.backward()

            optimizer.step()

            # Evaluate the score.

            if self.configs['model_type'] == 'TwoHemiRNNTanh_single_readout':
                preds = (zs[:,-1,0] >= 0).long()
                score = accuracy_score(labels.cpu().data.numpy(), preds.cpu().data.numpy())

            else:
                preds_left_alm = (zs[:,-1,0] >= 0).long()
                preds_right_alm = (zs[:,-1,1] >= 0).long()

                score_left_alm = accuracy_score(labels.cpu().data.numpy(), preds_left_alm.cpu().data.numpy())
                score_right_alm = accuracy_score(labels.cpu().data.numpy(), preds_right_alm.cpu().data.numpy())

                score = (score_left_alm+score_right_alm)/2

            losses.append(loss)
            scores.append(score)

            if (batch_idx + 1) % self.configs['log_interval'] == 0:
                cur_time = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}, fraction correct: {:.1f}% ({:.3f} s)'.format(
                    epoch + 1, trial_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                    loss.item(), 100. * score, cur_time - begin_time))
                begin_time = time.time()


        return losses, scores




    def val_helper(self, model, device, val_loader, loss_fct):

        model.eval()

        total_loss = 0
        total_score = 0

        trial_count = 0

        begin_time = time.time()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):

                inputs, labels  = data
                inputs, labels = inputs.to(device), labels.to(device)

                trial_count += len(labels)

                '''
                hs: (n_trials, T, n_neurons)
                zs: (n_trials, T, 2)
                '''
                _, zs = model(inputs)
                if self.configs['model_type'] == 'TwoHemiRNNTanh_single_readout':
                    # For single readout, zs is (n_trials, T, 1)
                    loss = loss_fct(zs[:,-1,0], labels.float()).item()*len(labels) # BCEWithLogitsLoss requires that the target be float between 0 and 1.
                else:
                    loss_left_alm = loss_fct(zs[:,-1,0], labels.float()).item()*len(labels) # BCEWithLogitsLoss requires that the target be float between 0 and 1.
                    loss_right_alm = loss_fct(zs[:,-1,1], labels.float()).item()*len(labels) # BCEWithLogitsLoss requires that the target be float between 0 and 1.

                    loss = loss_left_alm + loss_right_alm


                total_loss += loss


                # Evaluate the score.

                if self.configs['model_type'] == 'TwoHemiRNNTanh_single_readout':

                    preds = (zs[:,-1,0] >= 0).long()
                    score = accuracy_score(labels.cpu().data.numpy(), preds.cpu().data.numpy())
                    total_score += score

                else:

                    preds_left_alm = (zs[:,-1,0] >= 0).long()
                    preds_right_alm = (zs[:,-1,1] >= 0).long()

                    n_correct_left_alm = accuracy_score(labels.cpu().data.numpy(), preds_left_alm.cpu().data.numpy(), normalize=False)
                    n_correct_right_alm = accuracy_score(labels.cpu().data.numpy(), preds_right_alm.cpu().data.numpy(), normalize=False)

                    total_score += (n_correct_left_alm+n_correct_right_alm)/2


        total_loss /= trial_count
        total_score /= trial_count

        cur_time = time.time()

        print('Test set ({:d} samples): loss: {:.4f}, fraction correct: {:.1f}% ({:.3f} s)'.format(trial_count, total_loss, \
            100. * total_score, cur_time - begin_time))


        return total_loss, total_score









    def plot_weights_changes(self, sub_sample=20):
        """
        Analyze and plot weight evolution over training epochs
        chooses a random subset of weights and plots their changes over training epochs

        sub_sample: number of weights selected randomly to plot
        """

        import matplotlib.pyplot as plt
        
        # Get all weight files
        weight_files = [f for f in os.listdir(self.logs_save_path) if f.startswith('weights_epoch_')]
        weight_files.sort()
        
        # Separate within-hemi and across-hemi files
        within_hemi_files = [f for f in weight_files if 'within_hemi' in f]
        across_hemi_files = [f for f in weight_files if 'across_hemi' in f]
        
        # Extract epochs
        within_epochs = [int(f.split('_')[2]) for f in within_hemi_files]
        across_epochs = [int(f.split('_')[2]) for f in across_hemi_files]
        
        # Initialize tracking arrays
        weight_norms = {
            'w_hh_linear_ll': {'within': [], 'across': []},
            'w_hh_linear_rr': {'within': [], 'across': []},
            'w_hh_linear_lr': {'within': [], 'across': []},
            'w_hh_linear_rl': {'within': [], 'across': []}
        }
        # Randomly select sub_sample indices for sampling weights across all weights 
        weight_indices = np.random.choice(self.configs['n_neurons']//2, sub_sample, replace=False)
        # Load within-hemi weights
        for epoch, filename in zip(within_epochs, within_hemi_files):
            data = np.load(os.path.join(self.logs_save_path, filename))
            for weight_name in weight_norms.keys():
                if weight_name in data:
                    weight_norms[weight_name]['within'].append(data[weight_name][weight_indices, weight_indices])
        
        # Load across-hemi weights
        for epoch, filename in zip(across_epochs, across_hemi_files):
            data = np.load(os.path.join(self.logs_save_path, filename))
            for weight_name in weight_norms.keys():
                if weight_name in data:
                    weight_norms[weight_name]['across'].append(data[weight_name][weight_indices, weight_indices])
        
        # Create plots with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
        fig.suptitle('Weight Evolution During Training', fontsize=16)
        
        # Prepare epoch arrays for plotting
        all_epochs = []
        
        # Add within-hemi epochs
        if within_epochs:
            all_epochs.extend(within_epochs)
        
        # Add across-hemi epochs
        if across_epochs:
            across_epochs_adjusted = [e + len(within_epochs) for e in across_epochs]
            all_epochs.extend(across_epochs_adjusted)
        
        # Add background colors for training phases
        if all_epochs:
            # Yellow background for within-hemi training
            if within_epochs:
                ax1.axvspan(-0.5, len(within_epochs)-0.5, alpha=0.3, color='yellow', label='Within-hemi training')
                ax2.axvspan(-0.5, len(within_epochs)-0.5, alpha=0.3, color='yellow', label='Within-hemi training')
            
            # Grey background for across-hemi training
            if across_epochs:
                ax1.axvspan(len(within_epochs)-0.5, len(all_epochs)-0.5, alpha=0.3, color='grey', label='Across-hemi training')
                ax2.axvspan(len(within_epochs)-0.5, len(all_epochs)-0.5, alpha=0.3, color='grey', label='Across-hemi training')
        
        # Plot within-hemisphere weights (top subplot) - individual lines
        if all_epochs:
            # Plot left-to-left weights
            if weight_norms['w_hh_linear_ll']['within']:
                ax1.plot(within_epochs, weight_norms['w_hh_linear_ll']['within'], 'g-', linewidth=2, label='Left-to-Left')
            if weight_norms['w_hh_linear_ll']['across']:
                ax1.plot(across_epochs_adjusted, weight_norms['w_hh_linear_ll']['across'], 'g-', linewidth=2)
            
            # Plot right-to-right weights
            if weight_norms['w_hh_linear_rr']['within']:
                ax1.plot(within_epochs, weight_norms['w_hh_linear_rr']['within'], 'g--', linewidth=2, label='Right-to-Right')
            if weight_norms['w_hh_linear_rr']['across']:
                ax1.plot(across_epochs_adjusted, weight_norms['w_hh_linear_rr']['across'], 'g--', linewidth=2)
            
            ax1.set_ylabel('Within-hemisphere weights')
            ax1.set_title('Within-Hemisphere Weight Evolution')
            # ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot inter-hemisphere weights (bottom subplot) - individual lines
        if all_epochs:
            # Plot left-to-right weights
            if weight_norms['w_hh_linear_lr']['within']:
                ax2.plot(within_epochs, weight_norms['w_hh_linear_lr']['within'], 'k-', linewidth=2, label='Left-to-Right')
            if weight_norms['w_hh_linear_lr']['across']:
                ax2.plot(across_epochs_adjusted, weight_norms['w_hh_linear_lr']['across'], 'k-', linewidth=2)
            
            # Plot right-to-left weights
            if weight_norms['w_hh_linear_rl']['within']:
                ax2.plot(within_epochs, weight_norms['w_hh_linear_rl']['within'], 'k--', linewidth=2, label='Right-to-Left')
            if weight_norms['w_hh_linear_rl']['across']:
                ax2.plot(across_epochs_adjusted, weight_norms['w_hh_linear_rl']['across'], 'k--', linewidth=2)
            
            ax2.set_xlabel('Training Epoch')
            ax2.set_ylabel('Inter-hemisphere weights')
            ax2.set_title('Inter-Hemisphere Weight Evolution')
            # ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        fig_save_path = os.path.join(self.configs['plots_dir'], self.configs['model_type'], self.sub_path)
        os.makedirs(fig_save_path, exist_ok=True)
        
        fig.savefig(os.path.join(fig_save_path, f'weight_evolution_across_hemi_n_epochs_{self.configs["across_hemi_n_epochs"]}_within_hemi_n_epochs_{self.configs["n_epochs"]}.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_save_path, f'weight_evolution_across_hemi_n_epochs_{self.configs["across_hemi_n_epochs"]}_within_hemi_n_epochs_{self.configs["n_epochs"]}.svg'), bbox_inches='tight')
        
        plt.close()
        
        print(f'Weight evolution analysis saved to {fig_save_path}')


    def plot_cd_traces(self):
        '''
        Main differences from plot_cd_traces:
        1. Add light blue time window for pert period.
        '''

        train_type_str = self.configs['train_type']
        init_cross_hemi_rel_factor = self.configs['init_cross_hemi_rel_factor']
        random_seed = self.configs['random_seed']     


        model_type = self.configs['model_type']
        
        uni_pert_trials_prob = self.configs['uni_pert_trials_prob']


        test_random_seed = self.configs['test_random_seed']

        np.random.seed(test_random_seed)
        torch.manual_seed(test_random_seed)

        self.init_sub_path(train_type_str)



        # Detect devices
        use_cuda = bool(self.configs['use_cuda'])
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update
        # device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")

        # Data loading parameters
        if use_cuda:
            params = {'batch_size': self.configs['bs'], 'shuffle': False, 'num_workers': self.configs['num_workers'], \
            'pin_memory': bool(self.configs['pin_memory'])}
        else:
            params = {'batch_size': self.configs['bs'], 'shuffle': False}

        '''
        Load the dataset and wrap it with Pytorch Dataset.
        '''

        # test

        test_save_path = os.path.join(self.configs['data_dir'], 'test')

        test_sensory_inputs = np.load(os.path.join(test_save_path, 'sensory_inputs.npy'))
        test_trial_type_labels = np.load(os.path.join(test_save_path, 'trial_type_labels.npy'))

        test_set = data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))

        test_loader = data.DataLoader(test_set, **params)


        '''
        Load the saved model.
        '''

        import sys
        model = getattr(sys.modules[__name__], model_type)(self.configs, \
            self.a, self.pert_begin, self.pert_end).to(device)

        model_save_path = os.path.join(self.configs['models_dir'], model_type, self.sub_path)
        print("model: ", model_save_path)
        state_dict = torch.load(os.path.join(model_save_path, 'best_model.pth'), weights_only=True)

        model.load_state_dict(state_dict)


        # Unless otherwise specified, we set drop_p_min and max = 1
        model.drop_p_min = 1.0
        model.drop_p_max = 1.0


        '''
        noise.
        '''
        model.sigma_input_noise = self.configs['test_sigma_input_noise']
        model.rnn_cell.sigma = self.configs['test_sigma_rec_noise']


        '''
        Compute cd proj
        '''
        # train
        train_save_path = os.path.join(self.configs['data_dir'], 'train')

        train_sensory_inputs = np.load(os.path.join(train_save_path, 'sensory_inputs.npy'))
        train_trial_type_labels = np.load(os.path.join(train_save_path, 'trial_type_labels.npy'))

        train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))

        train_loader = data.DataLoader(train_set, **params, drop_last=True)



        cds = self.get_cds(model, device, train_loader, model_type) # cds[j] = (T, n_neurons in a given hemi)

        old_cds = cds

        # Average cd over the delay period.
        cds = np.zeros((self.n_loc_names,), dtype=object)
        for j in range(self.n_loc_names):
            cds[j] = old_cds[j][self.delay_begin:].mean(0)
            cds[j] = cds[j]/np.linalg.norm(cds[j]) # (n_neurons in a given hemi)

        cd_dbs = self.get_cd_dbs(cds, model, device, train_loader, model_type) # (n_loc_names)


        # Control trials
        model.uni_pert_trials_prob = 0
        no_stim_hs, no_stim_labels = self.get_neurons_trace(model, device, test_loader, model_type, hemi_type='all')


        # left_stim trials
        model.uni_pert_trials_prob = 1
        model.left_alm_pert_prob = 1

        left_stim_hs, left_stim_labels = self.get_neurons_trace(model, device, test_loader, model_type, hemi_type='all')


        # right_stim trials
        model.uni_pert_trials_prob = 1
        model.left_alm_pert_prob = 0

        right_stim_hs, right_stim_labels = self.get_neurons_trace(model, device, test_loader, model_type, hemi_type='all')


        n_neurons = no_stim_hs.shape[2]

        no_stim_cd_projs = np.zeros((self.n_loc_names,), dtype=object)
        left_stim_cd_projs = np.zeros((self.n_loc_names,), dtype=object)
        right_stim_cd_projs = np.zeros((self.n_loc_names,), dtype=object)

        for j in range(self.n_loc_names):
            if j == 0:
                no_stim_cd_projs[j] = no_stim_hs[...,:n_neurons//2].dot(cds[j]) # (n_trials, T)
                left_stim_cd_projs[j] = left_stim_hs[...,:n_neurons//2].dot(cds[j]) # (n_trials, T)
                right_stim_cd_projs[j] = right_stim_hs[...,:n_neurons//2].dot(cds[j]) # (n_trials, T)

            elif j == 1:
                no_stim_cd_projs[j] = no_stim_hs[...,n_neurons//2:].dot(cds[j]) # (n_trials, T)
                left_stim_cd_projs[j] = left_stim_hs[...,n_neurons//2:].dot(cds[j]) # (n_trials, T)
                right_stim_cd_projs[j] = right_stim_hs[...,n_neurons//2:].dot(cds[j]) # (n_trials, T)

            # Center by db.
            no_stim_cd_projs[j] = no_stim_cd_projs[j] - cd_dbs[j]
            left_stim_cd_projs[j] = left_stim_cd_projs[j] - cd_dbs[j]
            right_stim_cd_projs[j] = right_stim_cd_projs[j] - cd_dbs[j]


        '''
        Plot
        '''


        import mimic_alpha as ma
        alpha = 0.3
        alpha_r = ma.colorAlpha_to_rgb('r', alpha=alpha)
        alpha_b = ma.colorAlpha_to_rgb('b', alpha=alpha)

        from matplotlib import colors
        skyblue_rgb = colors.to_rgb('skyblue') # Directly inputting skyblue in the below line didn't work.
        alpha_skyblue = ma.colorAlpha_to_rgb(skyblue_rgb, alpha=0.5)[0] # [0] because the output is like [array([r, g, b])].

        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(2,2, wspace=0.4, hspace=0.4)

        T = no_stim_cd_projs[0].shape[1]


        timesteps = self.trial_begin_t_in_sec + self.t_step_in_sec*np.arange(T)

        for j in range(self.n_loc_names):
            for k in range(2):
                ax = fig.add_subplot(gs[j,k])

                # lick left
                ax.plot(timesteps, no_stim_cd_projs[j][no_stim_labels==0].mean(0), color='r', ls='--', lw=5)
                if k == 0:
                    ax.plot(timesteps, left_stim_cd_projs[j][left_stim_labels==0].mean(0), color='r', ls='-', lw=5)
                    ax.fill_between(timesteps, left_stim_cd_projs[j][left_stim_labels==0].mean(0) - left_stim_cd_projs[j][left_stim_labels==0].std(0),\
                        left_stim_cd_projs[j][left_stim_labels==0].mean(0) + left_stim_cd_projs[j][left_stim_labels==0].std(0), color=alpha_r)
                else:
                    ax.plot(timesteps, right_stim_cd_projs[j][right_stim_labels==0].mean(0), color='r', ls='-', lw=5)
                    ax.fill_between(timesteps, right_stim_cd_projs[j][right_stim_labels==0].mean(0) - right_stim_cd_projs[j][right_stim_labels==0].std(0),\
                        right_stim_cd_projs[j][right_stim_labels==0].mean(0) + right_stim_cd_projs[j][right_stim_labels==0].std(0), color=alpha_r)


                # lick right
                ax.plot(timesteps, no_stim_cd_projs[j][no_stim_labels==1].mean(0), color='b', ls='--', lw=5)
                if k == 0:
                    ax.plot(timesteps, left_stim_cd_projs[j][left_stim_labels==1].mean(0), color='b', ls='-', lw=5)
                    ax.fill_between(timesteps, left_stim_cd_projs[j][left_stim_labels==1].mean(0) - left_stim_cd_projs[j][left_stim_labels==1].std(0),\
                        left_stim_cd_projs[j][left_stim_labels==1].mean(0) + left_stim_cd_projs[j][left_stim_labels==1].std(0), color=alpha_b)

                else:
                    ax.plot(timesteps, right_stim_cd_projs[j][right_stim_labels==1].mean(0), color='b', ls='-', lw=5)
                    ax.fill_between(timesteps, right_stim_cd_projs[j][right_stim_labels==1].mean(0) - right_stim_cd_projs[j][right_stim_labels==1].std(0),\
                        right_stim_cd_projs[j][right_stim_labels==1].mean(0) + right_stim_cd_projs[j][right_stim_labels==1].std(0), color=alpha_b)

                # Find y max, y min values for y_lim and yticks.
                y_agg = cat([left_stim_cd_projs[j][left_stim_labels==0].mean(0) + left_stim_cd_projs[j][left_stim_labels==0].std(0),
                    left_stim_cd_projs[j][left_stim_labels==0].mean(0) - left_stim_cd_projs[j][left_stim_labels==0].std(0),
                    right_stim_cd_projs[j][right_stim_labels==0].mean(0) + right_stim_cd_projs[j][right_stim_labels==0].std(0),
                    right_stim_cd_projs[j][right_stim_labels==0].mean(0) - right_stim_cd_projs[j][right_stim_labels==0].std(0),
                    left_stim_cd_projs[j][left_stim_labels==1].mean(0) + left_stim_cd_projs[j][left_stim_labels==1].std(0),
                    left_stim_cd_projs[j][left_stim_labels==1].mean(0) - left_stim_cd_projs[j][left_stim_labels==1].std(0),
                    right_stim_cd_projs[j][right_stim_labels==1].mean(0) + right_stim_cd_projs[j][right_stim_labels==1].std(0),
                    right_stim_cd_projs[j][right_stim_labels==1].mean(0) - right_stim_cd_projs[j][right_stim_labels==1].std(0),
                    ], 0)

                y_abs_max = np.max(np.abs(y_agg))    



                ax.axvline(self.sample_begin_t_in_sec, ls='--', color='k', lw=2)
                ax.axvline(self.delay_begin_t_in_sec, ls='--', color='k', lw=2)

                ax.axvspan(self.pert_begin_t_in_sec, self.pert_end_t_in_sec, color=alpha_skyblue, zorder=-10)

                # spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)

                # ticks
                ax.tick_params(length=4, width=2, labelsize=30)

                ax.set_xticks([-3, -2, -1, 0])
                ax.set_yticks([-np.rint(y_abs_max), 0, np.rint(y_abs_max)])

                ax.set_xlabel('Time from movement (s)', fontsize=30)
                ax.set_ylabel('CD projection', fontsize=30)



        if self.configs['sigma_input_noise'] == self.configs['test_sigma_input_noise'] and \
        self.configs['sigma_rec_noise'] == self.configs['test_sigma_rec_noise']:
            noise_str = 'test_sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(model.sigma_input_noise, model.rnn_cell.sigma)

        else:
            noise_str = \
            'train_sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}_test_sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(\
                self.configs['sigma_input_noise'], self.configs['sigma_rec_noise'], model.sigma_input_noise, model.rnn_cell.sigma)





        fig_save_path = os.path.join(self.configs['plots_dir'], 'plot_cd_traces', train_type_str, \
            'init_cross_hemi_rel_factor_{:.2f}'.format(init_cross_hemi_rel_factor),\
            noise_str, \
            'random_seed_{}'.format(random_seed), \
            'xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}'.format(self.configs['xs_left_alm_amp'], self.configs['xs_right_alm_amp']))
        os.makedirs(fig_save_path, exist_ok=True)

        fig.savefig(os.path.join(fig_save_path, 'plot_cd_traces_model_type_{}_across_hemi_n_epochs_{}_within_hemi_n_epochs_{}.png'.format(model_type, self.configs['across_hemi_n_epochs'], self.configs['n_epochs'])))        
        fig.savefig(os.path.join(fig_save_path, 'plot_cd_traces_model_type_{}_across_hemi_n_epochs_{}_within_hemi_n_epochs_{}.svg'.format(model_type, self.configs['across_hemi_n_epochs'], self.configs['n_epochs'])))        


        print('done!')

    def get_cds(self, model, device, loader, model_type, recompute=True):
        '''
        Note added: we only compute cd using correct trials.

        Return
        cds: (n_loc_names,) cds[j] is a numpy array of shape (T, n_neurons in hemi j).
        '''

        save_path = os.path.join(self.configs['results_dir'], 'misc', 'get_cds', model_type,\
            self.sub_path)


        os.makedirs(save_path, exist_ok=True)

        if not recompute and os.path.isfile(os.path.join(save_path, 'cds.npy')):
            cds = np.load(os.path.join(save_path, 'cds.npy'), allow_pickle=True)
            return cds 

        else:
            model.uni_pert_trials_prob = 0
            model.left_alm_pert_prob = 1

            control_hs, control_labels, control_pred_labels = self.get_neurons_trace(model, device, loader, model_type, hemi_type='all', return_pred_labels=True) # np array (n_trials, T, n_neurons), (n_trials)

            control_suc_labels = (control_labels==control_pred_labels).astype(int)

            lick_right_avg = control_hs[(control_labels==1)*(control_suc_labels==1)].mean(0) # (T, n_neurons)
            lick_left_avg = control_hs[(control_labels==0)*(control_suc_labels==1)].mean(0) # (T, n_neurons)

            cd_raw = lick_right_avg - lick_left_avg
            # Now we separate left and right ALM.
            n_loc_names = 2
            n_neurons = cd_raw.shape[1]
            cds = np.zeros((n_loc_names,), dtype=object)

            for j in range(n_loc_names):
                if j == 0:
                    cur_cd = cd_raw[:,:n_neurons//2]
                else:
                    cur_cd = cd_raw[:,n_neurons//2:]

                cds[j] = cur_cd/np.linalg.norm(cur_cd, axis=1, keepdims=True) # (T, n_neurons in a given hemi)

            if not recompute:
                np.save(os.path.join(save_path, 'cds.npy'), cds)

            return cds




    def get_cd_dbs(self, cds, model, device, loader, model_type, recompute=True):
        '''
        Return
        cd_dbs: (n_loc_names,) cd_dbs[j] is a number.
        '''

        save_path = os.path.join(self.configs['results_dir'], 'misc', 'get_cd_dbs', model_type,\
            self.sub_path)


        os.makedirs(save_path, exist_ok=True)

        if not recompute and os.path.isfile(os.path.join(save_path, 'cd_dbs.npy')):
            cd_dbs = np.load(os.path.join(save_path, 'cd_dbs.npy'), allow_pickle=True)
            return cd_dbs 

        else:
            model.uni_pert_trials_prob = 0
            model.left_alm_pert_prob = 1

            control_hs, control_labels, control_pred_labels = self.get_neurons_trace(model, device, loader, model_type, hemi_type='all', return_pred_labels=True) # np array (n_trials, T, n_neurons), (n_trials)

            control_suc_labels = (control_labels==control_pred_labels).astype(int)

            n_loc_names = 2
            n_neurons = control_hs.shape[2]

            # Take last time bin.
            lick_left_h = control_hs[control_labels==0][:,-1] # (n_trials of i, n_neurons)
            lick_right_h = control_hs[control_labels==1][:,-1] # (n_trials of i, n_neurons)

            cd_dbs = np.zeros((n_loc_names,), dtype=object)

            for j in range(n_loc_names):
                cur_cd = cds[j].T # (n_neurons//2) inverted CW

                if j == 0:
                    cur_lick_left_cd_proj = lick_left_h[:,:n_neurons//2].dot(cur_cd) # (n_trials of i)
                    cur_lick_right_cd_proj = lick_right_h[:,:n_neurons//2].dot(cur_cd)
                else:
                    cur_lick_left_cd_proj = lick_left_h[:,n_neurons//2:].dot(cur_cd)
                    cur_lick_right_cd_proj = lick_right_h[:,n_neurons//2:].dot(cur_cd)

                cur_lick_left_avg = cur_lick_left_cd_proj.mean()
                cur_lick_left_var = np.var(cur_lick_left_cd_proj, ddof=1)

                cur_lick_right_avg = cur_lick_right_cd_proj.mean()
                cur_lick_right_var = np.var(cur_lick_right_cd_proj, ddof=1)

                cd_dbs[j] = (cur_lick_left_avg/cur_lick_left_var + cur_lick_right_avg/cur_lick_right_var)/(1/cur_lick_left_var+1/cur_lick_right_var)


            if not recompute:
                np.save(os.path.join(save_path, 'cd_dbs.npy'), cd_dbs)

            return cd_dbs









    def get_neurons_trace(self, model, device, loader, model_type, 
                        hemi_type='left_ALM', recompute=False, return_pred_labels=False,
                        return_zs=False, hemi_agree=True, corrupt=False,
                        single_readout=False):
        '''
        Return:
        numpy arrays
        hs: (n_trials, T, 1)
        labels: (n_trials)
        '''
        random_seed = self.configs['test_random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


        total_hs = []
        total_labels = []

        model.corrupt=corrupt

        model.eval()

        trial_count = 0

        if return_pred_labels:
            total_pred_labels = []
            total_preds_left_alm = []
            total_preds_right_alm = []

        begin_time = time.time()
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):

                inputs, labels  = data
                inputs, labels = inputs.to(device), labels.to(device)

                trial_count += len(labels)

                '''
                hs: (n_trials, T, n_neurons)
                zs: (n_trials, T, 2)

                zs: (n_trials, T, 1) if single_readout
                '''

                hs, zs = model(inputs)


                n_neurons = hs.shape[2]

                if single_readout:
                    pass
                if hemi_type == 'left_ALM':
                    hs = hs[...,:n_neurons//2]
                elif hemi_type == 'right_ALM':
                    hs = hs[...,n_neurons//2:]
                elif hemi_type == 'all':
                    pass


                total_hs.append(hs.cpu().data.numpy())
                total_labels.append(labels.cpu().data.numpy())

                if return_pred_labels:
                    if single_readout:
                        preds_all = (zs[:,-1,0] >= 0).long().cpu().data.numpy()
                        total_pred_labels.append(preds_all)
                    else:
                        preds_left_alm = (zs[:,-1,0] >= 0).long().cpu().data.numpy()
                        preds_right_alm = (zs[:,-1,1] >= 0).long().cpu().data.numpy()

                        # We take only trials for which both hemispheres agree in pred_labels. The other trials have value of -1.
                        if hemi_agree and hemi_type == 'all':
                            agree_mask = (preds_left_alm == preds_right_alm)
                            pred_labels = np.zeros_like(preds_left_alm)
                            pred_labels[agree_mask] = preds_left_alm[agree_mask]
                            pred_labels[~agree_mask] = -1

                            total_pred_labels.append(pred_labels)
                        else:
                            total_preds_left_alm.append(preds_left_alm)
                            total_preds_right_alm.append(preds_right_alm)

        total_hs = cat(total_hs, 0)
        total_labels = cat(total_labels, 0)

        if return_pred_labels and (hemi_type == 'all' or single_readout):
            total_pred_labels = cat(total_pred_labels, 0)

        if return_pred_labels:
            if hemi_type == 'all' or single_readout:
                return total_hs, total_labels, total_pred_labels
            else:
                return total_hs, total_labels, cat(total_preds_left_alm, 0), cat(total_preds_right_alm, 0) 

        else:
            return total_hs, total_labels





        

    def eval_with_perturbations(self, model, device, loader, model_type, n_control=None, seed=None, control_only=False):
        """
        Evaluate model accuracy under:
        - Control (no perturbation)
        - Left ALM photoinhibition
        - Right ALM photoinhibition
        - Bilateral photoinhibition

        Args:
            model: trained model
            device: device to run on
            loader: DataLoader over (xs, labels)
            model_type: model type string
            n_control: optional cap on number of control trials to subsample (for speed)
            seed: optional RNG seed for reproducibility of any shuffling
            control_only: if True, only evaluate control condition
        
        Returns:
            dict with accuracy scores for each condition
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Get CD and decision boundaries for evaluation
        cds = self.get_cds(model, device, loader, model_type, recompute=False)

        # Get cd by averaging over delay period:
        old_cds = cds

        # Average cd over the delay period.
        cds = np.zeros((2,), dtype=object)
        for j in range(self.n_loc_names):
            cds[j] = old_cds[j][self.delay_begin:].mean(0)
            cds[j] = cds[j]/np.linalg.norm(cds[j]) # (n_neurons in a given hemi)

        cd_dbs = self.get_cd_dbs(cds, model, device, loader, model_type, recompute=False)

        results = {}

        def per_hemi_metrics(model, device, loader, model_type, cds, cd_dbs, n_control=None):
            # Get all-neuron labels for reference
            _, all_labels = self.get_neurons_trace(model, device, loader, model_type, hemi_type='all', return_pred_labels=False)
            if n_control is not None and len(all_labels) > n_control:
                indices = np.random.choice(len(all_labels), n_control, replace=False)
            else:
                indices = np.arange(len(all_labels))

            # Left ALM readout
            left_hs, _, left_pred_labels, _ = self.get_neurons_trace(model, device, loader, model_type, hemi_type='left_ALM', return_pred_labels=True, hemi_agree=False)
            left_readout_acc = np.mean(all_labels[indices] == left_pred_labels[indices])
            # Right ALM readout
            right_hs, _, _, right_pred_labels = self.get_neurons_trace(model, device, loader, model_type, hemi_type='right_ALM', return_pred_labels=True, hemi_agree=False)
            right_readout_acc = np.mean(all_labels[indices] == right_pred_labels[indices])
            # Left ALM CD
            left_cd_acc = self._calculate_cd_accuracy_single_hemi(left_hs[indices], all_labels[indices], cds[0], cd_dbs[0])
            # Right ALM CD
            right_cd_acc = self._calculate_cd_accuracy_single_hemi(right_hs[indices], all_labels[indices], cds[1], cd_dbs[1])
            return {
                'readout_accuracy_left': left_readout_acc,
                'readout_accuracy_right': right_readout_acc,
                'cd_accuracy_left': left_cd_acc,
                'cd_accuracy_right': right_cd_acc,
                'n_trials_agreed': np.sum(left_pred_labels[indices] == right_pred_labels[indices]),
                'n_trials': len(indices)
            }

        # 1. Control condition (no perturbation)
        print("Evaluating control condition...")
        model.uni_pert_trials_prob = 0
        model.left_alm_pert_prob = 0.5
        results['control'] = per_hemi_metrics(model, device, loader, model_type, cds, cd_dbs, n_control=n_control)

        if not control_only:
            # 2. Left ALM photoinhibition
            print("Evaluating left ALM photoinhibition...")
            model.uni_pert_trials_prob = 1.0
            model.left_alm_pert_prob = 1.0
            results['left_alm_pert'] = per_hemi_metrics(model, device, loader, model_type, cds, cd_dbs)

            # 3. Right ALM photoinhibition
            print("Evaluating right ALM photoinhibition...")
            model.uni_pert_trials_prob = 1.0
            model.left_alm_pert_prob = 0.0
            results['right_alm_pert'] = per_hemi_metrics(model, device, loader, model_type, cds, cd_dbs)

        # 4. Bilateral photoinhibition (both left and right ALM)
        # print("Evaluating bilateral photoinhibition...")
        # bilateral_pert_hs, bilateral_pert_labels, _ = self._apply_bilateral_perturbation(
        #     model, device, loader, model_type
        # )
        # For bilateral, use the same CD and DBs, and compute per-hemi CD accuracy
        # n_trials = len(bilateral_pert_labels)
        # left_hs = bilateral_pert_hs[:, :, :bilateral_pert_hs.shape[2]//2]
        # right_hs = bilateral_pert_hs[:, :, bilateral_pert_hs.shape[2]//2:]
        # left_cd_acc = self._calculate_cd_accuracy_single_hemi(left_hs, bilateral_pert_labels, cds[0], cd_dbs[0])
        # right_cd_acc = self._calculate_cd_accuracy_single_hemi(right_hs, bilateral_pert_labels, cds[1], cd_dbs[1])
        # results['bilateral_pert'] = {
        #     'readout_accuracy_left': np.nan,  # Not meaningful for bilateral, unless you want to add it
        #     'readout_accuracy_right': np.nan,
        #     'cd_accuracy_left': left_cd_acc,
        #     'cd_accuracy_right': right_cd_acc,
        #     'n_trials': n_trials
        # }

        # Reset model perturbation settings
        model.uni_pert_trials_prob = 0
        model.left_alm_pert_prob = 0.5

        if not control_only:
            # Print summary
            print("\n" + "="*50)
            print("PERTURBATION EVALUATION RESULTS (per hemisphere)")
            print("="*50)
            for condition, metrics in results.items():
                print(f"{condition.replace('_', ' ').title()}:")
                print(f"  Readout Accuracy Left: {metrics['readout_accuracy_left']:.3f}")
                print(f"  Readout Accuracy Right: {metrics['readout_accuracy_right']:.3f}")
                print(f"  CD Accuracy Left: {metrics['cd_accuracy_left']:.3f}")
                print(f"  CD Accuracy Right: {metrics['cd_accuracy_right']:.3f}")
                print(f"  N Trials: {metrics['n_trials']}")
                print()
        else:
            print("\n" + "="*50)
            print("CONTROL RESULTS (per hemisphere)")
            print("="*50)
            for condition, metrics in results.items():
                print(f"{condition.replace('_', ' ').title()}:")
                print(f"  Readout Accuracy Left: {metrics['readout_accuracy_left']:.3f}")
                print(f"  Readout Accuracy Right: {metrics['readout_accuracy_right']:.3f}")
                print(f"  CD Accuracy Left: {metrics['cd_accuracy_left']:.3f}")
                print(f"  CD Accuracy Right: {metrics['cd_accuracy_right']:.3f}")
                print(f"  N Trials: {metrics['n_trials']}")
                print()
        return results

    def _calculate_cd_accuracy(self, hs, labels, cds, cd_dbs):
        """
        Calculate accuracy using CD projections and decision boundaries
        
        Args:
            hs: hidden states (n_trials, T, n_neurons)
            labels: true labels (n_trials,)
            cds: choice directions for each hemisphere
            cd_dbs: decision boundaries for each hemisphere
        
        Returns:
            accuracy using CD-based classification
        """
        n_neurons = hs.shape[2]
        n_trials = hs.shape[0]
        
        # Get final time point for classification
        final_hs = hs[:, -1, :]  # (n_trials, n_neurons)
        
        # Calculate CD projections for each hemisphere
        left_cd_proj = final_hs[:, :n_neurons//2].dot(cds[0])  # (n_trials,)
        right_cd_proj = final_hs[:, n_neurons//2:].dot(cds[1])  # (n_trials,)
        
        # Make predictions using decision boundaries
        left_preds = (left_cd_proj > cd_dbs[0]).astype(int)
        right_preds = (right_cd_proj > cd_dbs[1]).astype(int)
        
        # Use agreement between hemispheres (if they disagree, mark as incorrect)
        agree_mask = (left_preds == right_preds)
        cd_preds = np.zeros_like(left_preds)
        cd_preds[agree_mask] = left_preds[agree_mask]
        cd_preds[~agree_mask] = -1  # Mark disagreement trials as incorrect
        
        # Calculate accuracy (excluding disagreement trials)
        valid_mask = (cd_preds != -1)
        if np.sum(valid_mask) == 0:
            return 0.0
        
        cd_accuracy = np.mean(labels[valid_mask] == cd_preds[valid_mask])
        return cd_accuracy

    def _calculate_cd_accuracy_single_hemi(self, hs, labels, cd, cd_db):
        # hs: (n_trials, T, n_neurons_in_hemi)
        # cd: (n_neurons_in_hemi,)
        # cd_db: scalar
        final_hs = hs[:, -1, :]  # (n_trials, n_neurons_in_hemi)
        cd_proj = final_hs.dot(cd)  # (n_trials, n_neurons_in_hemi) x (n_neurons_in_hemi,) = (n_trials,)
        preds = (cd_proj > cd_db).astype(int)
        return np.mean(labels == preds)

    def _apply_bilateral_perturbation(self, model, device, loader, model_type):
        """
        Apply bilateral perturbation by manually setting both left and right perturbation masks
        """
        # Temporarily modify the model's perturbation settings
        original_uni_pert_prob = model.uni_pert_trials_prob
        original_left_alm_pert_prob = model.left_alm_pert_prob
        
        # Set up for bilateral perturbation
        model.uni_pert_trials_prob = 1.0
        model.left_alm_pert_prob = 0.5  # This will be overridden in the forward pass
        
        total_hs = []
        total_labels = []
        total_pred_labels = []
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Manually apply bilateral perturbation
                n_trials = inputs.size(0)
                T = inputs.size(1)
                h_pre = inputs.new_zeros(n_trials, model.n_neurons)
                hs = []
                
                # Apply input processing (same as in model.forward)
                xs_noise_left_alm = math.sqrt(2/model.a) * model.sigma_input_noise * torch.randn_like(inputs)
                xs_noise_right_alm = math.sqrt(2/model.a) * model.sigma_input_noise * torch.randn_like(inputs)
                
                xs_left_alm_mask = (torch.rand(n_trials, 1, 1) >= model.xs_left_alm_drop_p).float().to(inputs.device)
                xs_right_alm_mask = (torch.rand(n_trials, 1, 1) >= model.xs_right_alm_drop_p).float().to(inputs.device)
                
                xs_injected_left_alm = model.w_xh_linear_left_alm(inputs * xs_left_alm_mask * model.xs_left_alm_amp + xs_noise_left_alm)
                xs_injected_right_alm = model.w_xh_linear_right_alm(inputs * xs_right_alm_mask * model.xs_right_alm_amp + xs_noise_right_alm)
                xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)
                
                # Apply bilateral perturbation to all trials
                for t in range(T):
                    h = model.rnn_cell(xs_injected[:, t], h_pre)
                    
                    # Apply bilateral perturbation during delay period
                    if t >= model.pert_begin and t <= model.pert_end:
                        # Silence both left and right ALM neurons
                        h[:, :model.n_neurons//2] = 0  # Left ALM
                        h[:, model.n_neurons//2:] = 0  # Right ALM
                    
                    hs.append(h)
                    h_pre = h
                
                hs = torch.stack(hs, 1)
                
                # Get readout predictions
                zs_left_alm = model.readout_linear_left_alm(hs[..., :model.n_neurons//2])
                zs_right_alm = model.readout_linear_right_alm(hs[..., model.n_neurons//2:])
                zs = torch.cat([zs_left_alm, zs_right_alm], 2)
                
                # Calculate predictions
                preds_left_alm = (zs[:, -1, 0] >= 0).long()
                preds_right_alm = (zs[:, -1, 1] >= 0).long()
                
                # Agreement-based predictions
                agree_mask = (preds_left_alm == preds_right_alm)
                pred_labels = torch.zeros_like(preds_left_alm)
                pred_labels[agree_mask] = preds_left_alm[agree_mask]
                pred_labels[~agree_mask] = -1
                
                total_hs.append(hs.cpu().data.numpy())
                total_labels.append(labels.cpu().data.numpy())
                total_pred_labels.append(pred_labels.cpu().data.numpy())
        
        # Restore original perturbation settings
        model.uni_pert_trials_prob = original_uni_pert_prob
        model.left_alm_pert_prob = original_left_alm_pert_prob
        
        total_hs = np.concatenate(total_hs, 0)
        total_labels = np.concatenate(total_labels, 0)
        total_pred_labels = np.concatenate(total_pred_labels, 0)
        
        return total_hs, total_labels, total_pred_labels
        