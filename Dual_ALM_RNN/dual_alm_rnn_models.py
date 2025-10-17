import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import itertools
from copy import deepcopy

cat = np.concatenate


class TwoHemiRNNLinear(nn.Module):
    '''
    Same as TwoHemiRNNLinear, but use TwoHemiRNNCellLinear2
    '''
    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False):
        super().__init__()

        self.configs = configs

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        self.n_neurons = configs['n_neurons']
        self.n_left_neurons = self.n_neurons//2
        self.n_right_neurons = self.n_neurons - self.n_neurons//2

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(self.n_neurons//2)
        self.right_alm_inds = np.arange(self.n_neurons//2, self.n_neurons)


        self.rnn_cell = TwoHemiRNNCellLinear(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, 
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor)
        
        self.w_xh_linear_left_alm = nn.Linear(1, self.n_neurons//2, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons-self.n_neurons//2, bias=False)

        self.readout_linear_left_alm = nn.Linear(self.n_neurons//2, 1)
        self.readout_linear_right_alm = nn.Linear(self.n_neurons-self.n_neurons//2, 1)

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']

        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']


        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']



    def get_w_hh(self):
        w_hh = torch.zeros((self.n_neurons, self.n_neurons))
        w_hh[:self.n_neurons//2,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[self.n_neurons//2:,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[self.n_neurons//2:,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:self.n_neurons//2,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh



    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)


        init.normal_(self.readout_linear_left_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons//2))
        init.constant_(self.readout_linear_left_alm.bias, 0.0)

        init.normal_(self.readout_linear_right_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons-self.n_neurons//2))
        init.constant_(self.readout_linear_right_alm.bias, 0.0)


    def apply_pert(self, h, left_pert_trial_inds, right_pert_trial_inds):
        '''
        For each trial, we sample drop_p from [drop_p_min, drop_p_max]. Then, sample drop_p fraction of neurons to silence during the stim period.
        '''
        n_trials, n_neurons = h.size()


        '''
        Construct left_pert_mask
        '''
        n_left_pert_trials = len(left_pert_trial_inds)

        left_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_left_pert_trials) # (n_left_per_trials)


        left_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_left_pert_trials):
            cur_drop_p = left_pert_drop_ps[i]
            left_pert_neuron_inds = np.random.permutation(self.n_left_neurons)[:int(self.n_left_neurons*cur_drop_p)]
            left_pert_mask[left_pert_trial_inds[i],self.left_alm_inds[left_pert_neuron_inds]] = True


        '''
        Construct right_pert_mask
        '''
        n_right_pert_trials = len(right_pert_trial_inds)

        right_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_right_pert_trials) # (n_right_per_trials)

        right_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_right_pert_trials):
            cur_drop_p = right_pert_drop_ps[i]
            right_pert_neuron_inds = np.random.permutation(self.n_right_neurons)[:int(self.n_right_neurons*cur_drop_p)]
            right_pert_mask[right_pert_trial_inds[i],self.right_alm_inds[right_pert_neuron_inds]] = True


        # left pertubation
        h[np.nonzero(left_pert_mask)] = 0

        # right pertubation
        h[np.nonzero(right_pert_mask)] = 0




    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 1)

        Output:
        hs: (n_trials, T, n_neurons)
        zs: (n_trials, T, 2)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons)
        hs = []

        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)


        # input trial mask
        n_trials = xs.size(0)
        xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
        xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)


        xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
        xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)

        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            if t >= self.pert_begin and t <= self.pert_end:
                self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs_left_alm = self.readout_linear_left_alm(hs[...,self.left_alm_inds])
        zs_right_alm = self.readout_linear_right_alm(hs[...,self.right_alm_inds])
        zs = torch.cat([zs_left_alm, zs_right_alm], 2)

        return hs, zs



class TwoHemiRNNCellLinear(nn.Module):
    '''
    Same as TwoHemiRNNCellLinear except that we separately store within-hemi and cross-hemi weights, so that
    they can easily trained separately.
    '''

    def __init__(self, n_neurons=128, a=0.2, sigma=0.05, zero_init_cross_hemi=False,
        init_cross_hemi_rel_factor=1):
        super().__init__()
        self.n_neurons = n_neurons
        self.a = a
        self.sigma = sigma
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = init_cross_hemi_rel_factor

        self.w_hh_linear_ll = nn.Linear(n_neurons//2, n_neurons//2)
        self.w_hh_linear_rr = nn.Linear(n_neurons-n_neurons//2, n_neurons-n_neurons//2)

        self.w_hh_linear_lr = nn.Linear(n_neurons//2, n_neurons-n_neurons//2, bias=False)
        self.w_hh_linear_rl = nn.Linear(n_neurons-n_neurons//2, n_neurons//2, bias=False)


        self.init_params()

    
    def init_params(self):
        init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
        init.normal_(self.w_hh_linear_rr.weight, 0.0, 1.0/math.sqrt(self.n_neurons))

        if self.zero_init_cross_hemi:
            init.constant_(self.w_hh_linear_lr.weight, 0.0)
            init.constant_(self.w_hh_linear_rl.weight, 0.0)

        else:
            init.normal_(self.w_hh_linear_lr.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))
            init.normal_(self.w_hh_linear_rl.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))


        init.constant_(self.w_hh_linear_ll.bias, 0.0)
        init.constant_(self.w_hh_linear_rr.bias, 0.0)


    def full_recurrent(self, h_pre):
        h_pre_left = h_pre[:,:self.n_neurons//2]
        h_pre_right = h_pre[:,self.n_neurons//2:]

        h1 = torch.cat([self.w_hh_linear_ll(h_pre_left), self.w_hh_linear_lr(h_pre_left)], 1) # (n_neurons)
        h2 = torch.cat([self.w_hh_linear_rl(h_pre_right), self.w_hh_linear_rr(h_pre_right)], 1) # (n_neurons)

        return h1 + h2


    def forward(self, x_injected, h_pre):
        '''
        Input:
        x_injected: (n_trials, n_neurons)
        h_pre: (n_trials, n_neurons)

        Output:
        h: (n_trials, n_neurons)
        '''
        noise = math.sqrt(2/self.a)*self.sigma*torch.randn_like(x_injected)

        h = (1-self.a)*h_pre + self.a*(self.full_recurrent(h_pre) + x_injected + noise)

        return h


















class TwoHemiRNNTanh(nn.Module):

    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False, return_input=False):
        super().__init__()

        self.one_hot = configs['one_hot']

        self.return_input = return_input

        self.configs = configs

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        # bi stim pert
        self.bi_pert_trials_prob = None

        self.n_neurons = configs['n_neurons']
        self.n_left_neurons = self.n_neurons//2
        self.n_right_neurons = self.n_neurons - self.n_neurons//2

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(self.n_neurons//2)
        self.right_alm_inds = np.arange(self.n_neurons//2, self.n_neurons)


        self.rnn_cell = TwoHemiRNNCellGeneral(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, nonlinearity=nn.Tanh(),
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor)
        
        self.w_xh_linear_left_alm = nn.Linear(1, self.n_neurons//2, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons-self.n_neurons//2, bias=False)
        if self.one_hot:
            self.w_xh_linear_left_alm = nn.Linear(2, self.n_neurons//2, bias=False)  # 2 input channels
            self.w_xh_linear_right_alm = nn.Linear(2, self.n_neurons-self.n_neurons//2, bias=False)

        self.readout_linear_left_alm = nn.Linear(self.n_neurons//2, 1)
        self.readout_linear_right_alm = nn.Linear(self.n_neurons-self.n_neurons//2, 1)

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']


        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']

        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']



        self.corrupt=False
        if 'train_type_modular_corruption' in configs['train_type']:
            self.corruption_start_epoch = configs['corruption_start_epoch']
            self.corruption_noise = configs['corruption_noise']
            self.corruption_type = configs['corruption_type']


    def get_w_hh(self):
        w_hh = torch.zeros((self.n_neurons, self.n_neurons))
        w_hh[:self.n_neurons//2,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[self.n_neurons//2:,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[self.n_neurons//2:,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:self.n_neurons//2,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh




    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)


        init.normal_(self.readout_linear_left_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons//2))
        init.constant_(self.readout_linear_left_alm.bias, 0.0)

        init.normal_(self.readout_linear_right_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons-self.n_neurons//2))
        init.constant_(self.readout_linear_right_alm.bias, 0.0)


    def apply_pert(self, h, left_pert_trial_inds, right_pert_trial_inds):
        '''
        For each trial, we sample drop_p from [drop_p_min, drop_p_max]. Then, sample drop_p fraction of neurons to silence during the stim period.
        '''
        n_trials, n_neurons = h.size()


        '''
        Construct left_pert_mask
        '''
        n_left_pert_trials = len(left_pert_trial_inds)

        left_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_left_pert_trials) # (n_left_per_trials)


        left_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_left_pert_trials):
            cur_drop_p = left_pert_drop_ps[i]
            left_pert_neuron_inds = np.random.permutation(self.n_left_neurons)[:int(self.n_left_neurons*cur_drop_p)]
            left_pert_mask[left_pert_trial_inds[i],self.left_alm_inds[left_pert_neuron_inds]] = True


        '''
        Construct right_pert_mask
        '''
        n_right_pert_trials = len(right_pert_trial_inds)

        right_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_right_pert_trials) # (n_right_per_trials)

        right_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_right_pert_trials):
            cur_drop_p = right_pert_drop_ps[i]
            right_pert_neuron_inds = np.random.permutation(self.n_right_neurons)[:int(self.n_right_neurons*cur_drop_p)]
            right_pert_mask[right_pert_trial_inds[i],self.right_alm_inds[right_pert_neuron_inds]] = True


        # left pertubation
        h[np.nonzero(left_pert_mask)] = 0

        # right pertubation
        h[np.nonzero(right_pert_mask)] = 0



    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 1) or (n_trials, T, 2)

        Output:
        hs: (n_trials, T, n_neurons)
        zs: (n_trials, T, 2)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons)
        hs = []
        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        
        if self.one_hot:
            # Input masks - now need to match 2D input
            xs_left_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_left_alm_drop_p).float().to(xs.device)
            xs_right_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_right_alm_drop_p).float().to(xs.device)
        else:
            # input trial mask
            xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
            xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)

        if self.corrupt:
            corr_level = self.corruption_noise
            if self.one_hot:
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*(torch.randn_like(xs) + 2.0) # shift the mean of the gaussian

            elif self.corruption_type == "poisson":
                # xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.poisson(torch.ones_like(xs) * corr_level)
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*torch.poisson(torch.ones_like(xs.cpu()) * corr_level).to(xs.device)

            else:
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.randn_like(xs)

            xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr)
            # Keep right side unchanged
            xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)

        else:
            # print("no corruption ", self.xs_left_alm_amp, self.xs_right_alm_amp)
            xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
            xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)

        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]

        # Bi stim pert
        if self.bi_pert_trials_prob is not None:
            n_trials = xs.size(0)
            bi_pert_trial_inds = np.random.permutation(n_trials)[:int(self.bi_pert_trials_prob*n_trials)]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            if t >= self.pert_begin and t <= self.pert_end:
                if self.bi_pert_trials_prob is None:
                    self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)
                else:
                    self.apply_bi_pert(h, bi_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs_left_alm = self.readout_linear_left_alm(hs[...,self.left_alm_inds])
        zs_right_alm = self.readout_linear_right_alm(hs[...,self.right_alm_inds])
        zs = torch.cat([zs_left_alm, zs_right_alm], 2)

        if self.return_input:
            return xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm, hs, zs
        else:
            return hs, zs



class TwoHemiRNNTanh_single_readout(nn.Module):
    # Same class as TwoHemiRNNTanh, but there is a single readout layer for both hemispheres

    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False, return_input=False, noise=True):
        super().__init__()

        self.one_hot = configs['one_hot']

        self.return_input = return_input

        self.configs = configs
        self.symmetric_weights = False

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        self.train_type = configs['train_type']

        # bi stim pert
        self.bi_pert_trials_prob = None

        self.n_neurons = configs['n_neurons']
        self.n_left_neurons = self.n_neurons//2
        self.n_right_neurons = self.n_neurons - self.n_neurons//2

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(self.n_neurons//2)
        self.right_alm_inds = np.arange(self.n_neurons//2, self.n_neurons)

        self.noise = noise

        self.rnn_cell = TwoHemiRNNCellGeneral(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, nonlinearity=nn.Tanh(),
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor, 
            symmetric_weights=self.symmetric_weights, noise=self.noise)
        
        self.w_xh_linear_left_alm = nn.Linear(1, self.n_neurons//2, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons-self.n_neurons//2, bias=False)
        if self.one_hot:
            self.w_xh_linear_left_alm = nn.Linear(2, self.n_neurons//2, bias=False)  # 2 input channels
            self.w_xh_linear_right_alm = nn.Linear(2, self.n_neurons-self.n_neurons//2, bias=False)

        self.readout_linear = nn.Linear(self.n_neurons, 1)

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']


        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']

        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']


        self.corrupt=False
        if 'train_type_modular_corruption' in configs['train_type']:
            self.corruption_start_epoch = configs['corruption_start_epoch']
            self.corruption_noise = configs['corruption_noise']
            self.corruption_type = configs['corruption_type']




    def get_w_hh(self):
        w_hh = torch.zeros((self.n_neurons, self.n_neurons))
        w_hh[:self.n_neurons//2,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[self.n_neurons//2:,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[self.n_neurons//2:,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:self.n_neurons//2,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh




    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        if self.symmetric_weights:
            print("Matching weights for left and right ALM")

            self.w_xh_linear_right_alm.weight = self.w_xh_linear_left_alm.weight
        elif 'fixed_input' in self.configs['train_type']:
            print("Fixed input weights for left and right ALM")

            if self.n_neurons < 500:
                # Only set half and half when there are two neurons per hemisphere
                channel_0 = torch.cat((torch.ones(int(self.n_left_neurons//2)), torch.zeros(int(self.n_left_neurons//2))))
                channel_1 = torch.cat((torch.zeros(int(self.n_left_neurons//2)), torch.ones(int(self.n_left_neurons//2))))

            else:
                # Set all to all when there are more than two neurons per hemisphere
                norm_factor = self.n_neurons
                norm_factor = 1
                channel_0 = torch.cat((torch.ones(int(self.n_left_neurons//2)) / norm_factor, torch.ones(int(self.n_left_neurons//2)) / norm_factor))
                channel_1 = torch.cat((torch.ones(int(self.n_left_neurons//2)) / norm_factor, torch.ones(int(self.n_left_neurons//2)) / norm_factor))

            # import pdb; pdb.set_trace()
            self.w_xh_linear_right_alm.weight.data = torch.stack((channel_0, channel_1), dim=1) #dtype=torch.float32)
            self.w_xh_linear_left_alm.weight.data = torch.stack((channel_0, channel_1), dim=1) #dtype=torch.float32)
            
        else:
            init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)

        # Set all values in readout_linear to be the same value drawn from normal distribution
        if self.symmetric_weights:
            val = torch.normal(mean=0.0, std=1.0/math.sqrt(self.n_neurons), size=(1,))
            with torch.no_grad():
                self.readout_linear.weight.fill_(val.item())
        else:
            init.normal_(self.readout_linear.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
        init.constant_(self.readout_linear.bias, 0.0)

    def apply_pert(self, h, left_pert_trial_inds, right_pert_trial_inds):
        '''
        For each trial, we sample drop_p from [drop_p_min, drop_p_max]. Then, sample drop_p fraction of neurons to silence during the stim period.
        '''
        n_trials, n_neurons = h.size()


        '''
        Construct left_pert_mask
        '''
        n_left_pert_trials = len(left_pert_trial_inds)

        left_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_left_pert_trials) # (n_left_per_trials)

        left_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_left_pert_trials):
            cur_drop_p = left_pert_drop_ps[i]
            left_pert_neuron_inds = np.random.permutation(self.n_left_neurons)[:int(self.n_left_neurons*cur_drop_p)]
            left_pert_mask[left_pert_trial_inds[i],self.left_alm_inds[left_pert_neuron_inds]] = True


        '''
        Construct right_pert_mask
        '''
        n_right_pert_trials = len(right_pert_trial_inds)

        right_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_right_pert_trials) # (n_right_per_trials)

        right_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_right_pert_trials):
            cur_drop_p = right_pert_drop_ps[i]
            right_pert_neuron_inds = np.random.permutation(self.n_right_neurons)[:int(self.n_right_neurons*cur_drop_p)]
            right_pert_mask[right_pert_trial_inds[i],self.right_alm_inds[right_pert_neuron_inds]] = True


        # left pertubation
        h[np.nonzero(left_pert_mask)] = 0

        # right pertubation
        h[np.nonzero(right_pert_mask)] = 0



    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 1) or (n_trials, T, 2)

        Output:
        hs: (n_trials, T, n_neurons)
        zs: (n_trials, T, 1)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons)
        hs = []
        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)

        if self.symmetric_weights:
            xs_noise_left_alm = xs_noise_right_alm
        
        if self.one_hot:
            # Input masks - now need to match 2D input
            xs_left_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_left_alm_drop_p).float().to(xs.device)
            xs_right_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_right_alm_drop_p).float().to(xs.device)
        else:
            # input trial mask
            xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
            xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)

        if self.corrupt:
            
            corr_level = self.corruption_noise
            if self.one_hot:
                # xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*(torch.randn_like(xs) + 2.0) # shift the mean of the gaussian to match the mean of the input
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*(torch.randn_like(xs)*corr_level + 0.0) # shift the mean of the gaussian to match the mean of the input

            elif self.corruption_type == "poisson":
                # xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.poisson(torch.ones_like(xs) * corr_level)
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*torch.poisson(torch.ones_like(xs.cpu()) * corr_level).to(xs.device)

            else:
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.randn_like(xs)

            if self.noise:

                xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr)
                # Keep right side unchanged
                xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)
            else:
                xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr)
                xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)

            # xs_injected_left_alm = self.w_xh_linear_left_alm((xs*xs_left_alm_mask + xs_noise_left_alm_corr)*self.xs_left_alm_amp) # Multiply term outside of everything
            # xs_injected_right_alm = self.w_xh_linear_right_alm((xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp)

        else:
            # print("no corruption ", self.xs_left_alm_amp, self.xs_right_alm_amp)

            if self.noise:

                if 'switch' in self.train_type:
                    # Switch between giving left and right ALM no input (just noise) or both get input
                    # Here, can also switch between giving 0.2 or 0.5 input instead of 0
                    random_bits_left = np.random.randint(0, 2, size=(n_trials, 1, 1))
                    constant_01_rows_left = torch.from_numpy(np.broadcast_to(random_bits_left, (n_trials, T, 2))).float().to(xs.device)
                    random_bits_right = np.random.randint(0, 2, size=(n_trials, 1, 1))
                    constant_01_rows_right = torch.from_numpy(np.broadcast_to(random_bits_right, (n_trials, T, 2))).float().to(xs.device)

                    # Noise variable
                    # Create a scaling factor array of shape (1000, 1, 1), value=1 when random_bits_left==0, value=sigma_input_noise when random_bits_left==1
                    scaling_factors_left = np.where(random_bits_left == 0, 1.0, self.sigma_input_noise).astype(np.float32)
                    scaling_factors_right = np.where(random_bits_right == 0, 1.0, self.sigma_input_noise).astype(np.float32)
                    # Broadcast to shape (1000, 100, 2)
                    scaling_factors_exp = np.broadcast_to(scaling_factors_left, xs.shape)
                    scaling_factors_tensor = torch.from_numpy(scaling_factors_exp).to(xs.device).type_as(xs)
                    xs_noise_left_alm = math.sqrt(2/self.a) * torch.randn_like(xs) * scaling_factors_tensor
                    scaling_factors_exp = np.broadcast_to(scaling_factors_right, xs.shape)
                    scaling_factors_tensor = torch.from_numpy(scaling_factors_exp).to(xs.device).type_as(xs)
                    xs_noise_right_alm = math.sqrt(2/self.a) * torch.randn_like(xs) * scaling_factors_tensor
                    
                    xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*constant_01_rows_left + xs_noise_left_alm)
                    xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*constant_01_rows_right + xs_noise_right_alm)
                else:
                    xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
                    xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)
            else:
                # No noise test
                print("No noise")
                xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp)
                xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)

            # xs_injected_left_alm = self.w_xh_linear_left_alm((xs*xs_left_alm_mask + xs_noise_left_alm)*self.xs_left_alm_amp) # Multiply term outside of everything
            # xs_injected_right_alm = self.w_xh_linear_right_alm((xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp)

            # xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp)
            # xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)

        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]

        # Bi stim pert
        if self.bi_pert_trials_prob is not None:
            n_trials = xs.size(0)
            bi_pert_trial_inds = np.random.permutation(n_trials)[:int(self.bi_pert_trials_prob*n_trials)]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            if t >= self.pert_begin and t <= self.pert_end:
                if self.bi_pert_trials_prob is None:
                    self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)
                else:
                    self.apply_bi_pert(h, bi_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs = self.readout_linear(hs)  # (n_trials, T, 1)

        if self.return_input:
            if self.corrupt:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr, 
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm_corr)*self.xs_left_alm_amp, 
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
            elif 'switch' in self.train_type:
                return (constant_01_rows_left, constant_01_rows_right), hs, zs
            else:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm,
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm)*self.xs_left_alm_amp,
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs
        else:
            return hs, zs


    def set_custom_weights(self, custom_weights_dict):
        """Set custom weights for specific connections"""
        with torch.no_grad():
            # Get the model's default dtype to ensure consistency
            model_dtype = next(self.parameters()).dtype
            
            # Set recurrent weights
            if 'w_hh_linear_ll' in custom_weights_dict:
                weight_tensor = custom_weights_dict['w_hh_linear_ll'].to(dtype=model_dtype)
                self.rnn_cell.w_hh_linear_ll.weight.data = weight_tensor
            if 'w_hh_linear_rr' in custom_weights_dict:
                weight_tensor = custom_weights_dict['w_hh_linear_rr'].to(dtype=model_dtype)
                self.rnn_cell.w_hh_linear_rr.weight.data = weight_tensor
            if 'w_hh_linear_lr' in custom_weights_dict:
                weight_tensor = custom_weights_dict['w_hh_linear_lr'].to(dtype=model_dtype)
                self.rnn_cell.w_hh_linear_lr.weight.data = weight_tensor
            if 'w_hh_linear_rl' in custom_weights_dict:
                weight_tensor = custom_weights_dict['w_hh_linear_rl'].to(dtype=model_dtype)
                self.rnn_cell.w_hh_linear_rl.weight.data = weight_tensor
            
            # Set input projection weights (not used)
            if 'w_xh_linear_left_alm' in custom_weights_dict:
                weight_tensor = custom_weights_dict['w_xh_linear_left_alm'].to(dtype=model_dtype)
                self.w_xh_linear_left_alm.weight.data = weight_tensor
            if 'w_xh_linear_right_alm' in custom_weights_dict:
                weight_tensor = custom_weights_dict['w_xh_linear_right_alm'].to(dtype=model_dtype)
                self.w_xh_linear_right_alm.weight.data = weight_tensor
            
            # Set readout weights
            if 'readout_linear' in custom_weights_dict:
                weight_tensor = custom_weights_dict['readout_linear'].to(dtype=model_dtype)
                self.readout_linear.weight.data = weight_tensor
            if 'readout_bias' in custom_weights_dict:
                weight_tensor = custom_weights_dict['readout_bias'].to(dtype=model_dtype)
                self.readout_linear.bias.data = weight_tensor
    
    def forward_with_custom_weights(self, xs, custom_weights_dict):
        """Forward pass with custom weights (temporarily sets weights, then restores)"""
        # # Store original weights
        # original_weights = {}
        # for name, param in self.named_parameters():
        #     original_weights[name] = param.data.clone()
        
        # Set custom weights
        self.set_custom_weights(custom_weights_dict)
        
        # Run forward pass
        with torch.no_grad():
            hs, zs = self.forward(xs)
        
        # # Restore original weights 
        # for name, param in self.named_parameters():
        #     if name in original_weights:
        #         param.data = original_weights[name]
        
        return hs, zs





class TwoHemiRNNTanh_asymmetric_single_readout(nn.Module):
    # Same class as TwoHemiRNNTanh, but there is a single readout layer for both hemispheres

    # Note: n_neurons here only refers to the right hemi (will be greater than 2), left hemi has 2 neurons

    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False, return_input=False, noise=True):
        super().__init__()

        self.one_hot = configs['one_hot']

        self.return_input = return_input

        self.configs = configs
        self.symmetric_weights = False

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        # bi stim pert
        self.bi_pert_trials_prob = None

        self.n_neurons = configs['n_neurons'] # here n neurosn only refers to the right hemi
        
        self.n_left_neurons = 2 #self.n_neurons//2
        self.n_right_neurons = self.n_neurons 

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(2)
        self.right_alm_inds = np.arange(self.n_neurons)

        self.noise = noise

        self.rnn_cell = TwoHemiRNNCellGeneral_asymmetric(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, nonlinearity=nn.Tanh(),
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor, 
            symmetric_weights=self.symmetric_weights, noise=self.noise)
        
        self.w_xh_linear_left_alm = nn.Linear(1, 2, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons, bias=False)
        if self.one_hot:
            self.w_xh_linear_left_alm = nn.Linear(2, 2, bias=False)  # 2 input channels
            self.w_xh_linear_right_alm = nn.Linear(2, self.n_neurons, bias=False)

        self.readout_linear = nn.Linear(self.n_neurons + 2, 1)

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']


        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']

        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']


        self.corrupt=False
        if 'train_type_modular_corruption' in configs['train_type']:
            self.corruption_start_epoch = configs['corruption_start_epoch']
            self.corruption_noise = configs['corruption_noise']
            self.corruption_type = configs['corruption_type']




    def get_w_hh(self):
        w_hh = torch.zeros((self.n_neurons+2, self.n_neurons+2))
        w_hh[:2,:2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[2:,2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[2:,:2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:2,2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh




    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        if self.symmetric_weights:
            print("Matching weights for left and right ALM")

            self.w_xh_linear_right_alm.weight = self.w_xh_linear_left_alm.weight
        elif 'fixed_input' in self.configs['train_type']:
            print("Fixed input weights for left and right ALM")

            
            channel_0 = torch.cat((torch.ones(int(self.n_right_neurons//2)), torch.zeros(int(self.n_right_neurons//2))))
            channel_1 = torch.cat((torch.zeros(int(self.n_right_neurons//2)), torch.ones(int(self.n_right_neurons//2))))

            self.w_xh_linear_left_alm.weight.data = torch.tensor([[1.0,0.0],[0.0,1.0]], dtype=torch.float32)
            self.w_xh_linear_right_alm.weight.data = torch.stack((channel_0, channel_1), dim=1) #dtype=torch.float32)
            
        else:
            init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)

        # Set all values in readout_linear to be the same value drawn from normal distribution
        if self.symmetric_weights:
            val = torch.normal(mean=0.0, std=1.0/math.sqrt(self.n_neurons), size=(1,))
            with torch.no_grad():
                self.readout_linear.weight.fill_(val.item())
        else:
            init.normal_(self.readout_linear.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
        # Normalize readout_linear weights so that their absolute summed value is 3
        with torch.no_grad():
            weight = self.readout_linear.weight
            abs_sum = torch.sum(torch.abs(weight))
            if abs_sum > 0:
                weight.mul_(3.0 / abs_sum)
        init.constant_(self.readout_linear.bias, 0.0)




    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 1) or (n_trials, T, 2)

        Output:
        hs: (n_trials, T, n_neurons)
        zs: (n_trials, T, 1)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons+2)
        hs = []
        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)

        if self.symmetric_weights:
            xs_noise_left_alm = xs_noise_right_alm
        
        if self.one_hot:
            # Input masks - now need to match 2D input
            xs_left_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_left_alm_drop_p).float().to(xs.device)
            xs_right_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_right_alm_drop_p).float().to(xs.device)
        else:
            # input trial mask
            xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
            xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)


        if self.noise:
            xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
            xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)
        else:
            # No noise test
            print("No noise")
            xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp)
            xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)


        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]

        # Bi stim pert
        if self.bi_pert_trials_prob is not None:
            n_trials = xs.size(0)
            bi_pert_trial_inds = np.random.permutation(n_trials)[:int(self.bi_pert_trials_prob*n_trials)]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            # if t >= self.pert_begin and t <= self.pert_end:
            #     if self.bi_pert_trials_prob is None:
            #         self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)
            #     else:
            #         self.apply_bi_pert(h, bi_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs = self.readout_linear(hs)  # (n_trials, T, 1)

        if self.return_input:
            if self.corrupt:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr, 
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm_corr)*self.xs_left_alm_amp, 
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
            else:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm,
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm)*self.xs_left_alm_amp,
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs
        else:
            return hs, zs






class TwoHemiRNNTanh_single_readout_sparse(nn.Module):
    # Same class as TwoHemiRNNTanh, but there is a single readout layer for both hemispheres
    # additionally, we separate the units that are: input vs crosshemi vs readout weights

    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False, return_input=False, noise=True):
        super().__init__()

        self.one_hot = configs['one_hot']

        self.return_input = return_input

        self.configs = configs
        self.symmetric_weights = False

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        # bi stim pert
        self.bi_pert_trials_prob = None

        self.n_neurons = configs['n_neurons']
        self.n_left_neurons = self.n_neurons//2
        self.n_right_neurons = self.n_neurons - self.n_neurons//2

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(self.n_neurons//2)
        self.right_alm_inds = np.arange(self.n_neurons//2, self.n_neurons)

        self.noise = noise

        self.rnn_cell = TwoHemiRNNCellSparse(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, nonlinearity=nn.Tanh(),
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor, symmetric_weights=self.symmetric_weights, noise=self.noise)
        

        # only give inputs to 1/3 of units on each hemisphere
        self.w_xh_linear_left_alm = nn.Linear(1, self.n_neurons//2//3, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons//2//3, bias=False)

        if self.one_hot:
            self.w_xh_linear_left_alm = nn.Linear(2, self.n_neurons//2//3, bias=False)  # 2 input channels
            self.w_xh_linear_right_alm = nn.Linear(2, self.n_neurons//2//3, bias=False)

        self.readout_linear = nn.Linear(self.n_neurons//3, 1) # only 1/3 of all units readout

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']


        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']

        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']


        self.corrupt=False
        if 'train_type_modular_corruption' in configs['train_type']:
            self.corruption_start_epoch = configs['corruption_start_epoch']
            self.corruption_noise = configs['corruption_noise']
            self.corruption_type = configs['corruption_type']




    def get_w_hh(self):

        # FIXME: need to modify this to reflect the sparse weights
        w_hh = torch.zeros((self.n_neurons, self.n_neurons))
        w_hh[:self.n_neurons//2,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[self.n_neurons//2:,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[self.n_neurons//2:,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:self.n_neurons//2,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh




    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        if self.symmetric_weights:
            print("Matching weights for left and right ALM")

            self.w_xh_linear_right_alm.weight = self.w_xh_linear_left_alm.weight
        elif 'fixed_input' in self.configs['train_type']:
            print("Fixed input weights for left and right ALM")
            self.w_xh_linear_right_alm.weight.data = torch.tensor([[1.0, 0.0],[0.0,1.0]], dtype=torch.float32)
            self.w_xh_linear_left_alm.weight.data = torch.tensor([[1.0, 0.0],[0.0,1.0]], dtype=torch.float32)
            
        else:
            init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)

        # Set all values in readout_linear to be the same value drawn from normal distribution
        if self.symmetric_weights:
            val = torch.normal(mean=0.0, std=1.0/math.sqrt(self.n_neurons), size=(1,))
            with torch.no_grad():
                self.readout_linear.weight.fill_(val.item())
        else:
            init.normal_(self.readout_linear.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
        init.constant_(self.readout_linear.bias, 0.0)

    def apply_pert(self, h, left_pert_trial_inds, right_pert_trial_inds):
        '''
        For each trial, we sample drop_p from [drop_p_min, drop_p_max]. Then, sample drop_p fraction of neurons to silence during the stim period.
        '''
        n_trials, n_neurons = h.size()


        '''
        Construct left_pert_mask
        '''
        n_left_pert_trials = len(left_pert_trial_inds)

        left_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_left_pert_trials) # (n_left_per_trials)


        left_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_left_pert_trials):
            cur_drop_p = left_pert_drop_ps[i]
            left_pert_neuron_inds = np.random.permutation(self.n_left_neurons)[:int(self.n_left_neurons*cur_drop_p)]
            left_pert_mask[left_pert_trial_inds[i],self.left_alm_inds[left_pert_neuron_inds]] = True


        '''
        Construct right_pert_mask
        '''
        n_right_pert_trials = len(right_pert_trial_inds)

        right_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_right_pert_trials) # (n_right_per_trials)

        right_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_right_pert_trials):
            cur_drop_p = right_pert_drop_ps[i]
            right_pert_neuron_inds = np.random.permutation(self.n_right_neurons)[:int(self.n_right_neurons*cur_drop_p)]
            right_pert_mask[right_pert_trial_inds[i],self.right_alm_inds[right_pert_neuron_inds]] = True


        # left pertubation
        h[np.nonzero(left_pert_mask)] = 0

        # right pertubation
        h[np.nonzero(right_pert_mask)] = 0



    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 2)

        Output:
        hs: (n_trials, T, n_neurons)
        zs: (n_trials, T, 1)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons)
        hs = []
        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)

        if self.symmetric_weights:
            xs_noise_left_alm = xs_noise_right_alm
        
        if self.one_hot:
            # Input masks - now need to match 2D input
            xs_left_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_left_alm_drop_p).float().to(xs.device)
            xs_right_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_right_alm_drop_p).float().to(xs.device)
        else:
            # input trial mask
            xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
            xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)

        if self.corrupt:
            
            corr_level = self.corruption_noise
            if self.one_hot:
                # xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*(torch.randn_like(xs) + 2.0) # shift the mean of the gaussian to match the mean of the input
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*(torch.randn_like(xs)*corr_level + 0.0) # shift the mean of the gaussian to match the mean of the input

            elif self.corruption_type == "poisson":
                # xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.poisson(torch.ones_like(xs) * corr_level)
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*torch.poisson(torch.ones_like(xs.cpu()) * corr_level).to(xs.device)

            else:
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.randn_like(xs)

            if self.noise:

                xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr)
                # Keep right side unchanged
                xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)
            else:
                xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr)
                xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)

            # xs_injected_left_alm = self.w_xh_linear_left_alm((xs*xs_left_alm_mask + xs_noise_left_alm_corr)*self.xs_left_alm_amp) # Multiply term outside of everything
            # xs_injected_right_alm = self.w_xh_linear_right_alm((xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp)

        else:
            # print("no corruption ", self.xs_left_alm_amp, self.xs_right_alm_amp)



            if self.noise:
                xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
                xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)
            else:
                # No noise test
                print("No noise")
                xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp)
                xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)

            # xs_injected_left_alm = self.w_xh_linear_left_alm((xs*xs_left_alm_mask + xs_noise_left_alm)*self.xs_left_alm_amp) # Multiply term outside of everything
            # xs_injected_right_alm = self.w_xh_linear_right_alm((xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp)

            # xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp)
            # xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)

        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]

        # Bi stim pert
        if self.bi_pert_trials_prob is not None:
            n_trials = xs.size(0)
            bi_pert_trial_inds = np.random.permutation(n_trials)[:int(self.bi_pert_trials_prob*n_trials)]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            if t >= self.pert_begin and t <= self.pert_end:
                if self.bi_pert_trials_prob is None:
                    self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)
                else:
                    self.apply_bi_pert(h, bi_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs = self.readout_linear(hs)  # (n_trials, T, 1)

        if self.return_input:
            if self.corrupt:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr, 
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm_corr)*self.xs_left_alm_amp, 
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
            else:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm,
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm)*self.xs_left_alm_amp,
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs
        else:
            return hs, zs


class TwoHemiRNNSigmoid_single_readout(nn.Module):
    # Same class as TwoHemiRNNTanh, but there is a single readout layer for both hemispheres

    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False, return_input=False):
        super().__init__()

        self.one_hot = configs['one_hot']

        self.return_input = return_input

        self.configs = configs
        self.symmetric_weights = False

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        # bi stim pert
        self.bi_pert_trials_prob = None

        self.n_neurons = configs['n_neurons']
        self.n_left_neurons = self.n_neurons//2
        self.n_right_neurons = self.n_neurons - self.n_neurons//2

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(self.n_neurons//2)
        self.right_alm_inds = np.arange(self.n_neurons//2, self.n_neurons)


        self.rnn_cell = TwoHemiRNNCellGeneral(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, nonlinearity=nn.Sigmoid(), # nonlinearity=nn.Tanh(),
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor, symmetric_weights=self.symmetric_weights)
        
        self.w_xh_linear_left_alm = nn.Linear(1, self.n_neurons//2, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons-self.n_neurons//2, bias=False)
        if self.one_hot:
            self.w_xh_linear_left_alm = nn.Linear(2, self.n_neurons//2, bias=False)  # 2 input channels
            self.w_xh_linear_right_alm = nn.Linear(2, self.n_neurons-self.n_neurons//2, bias=False)

        self.readout_linear = nn.Linear(self.n_neurons, 1)

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']


        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']

        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']



        self.corrupt=False
        if 'train_type_modular_corruption' in configs['train_type']:
            self.corruption_start_epoch = configs['corruption_start_epoch']
            self.corruption_noise = configs['corruption_noise']
            self.corruption_type = configs['corruption_type']




    def get_w_hh(self):
        w_hh = torch.zeros((self.n_neurons, self.n_neurons))
        w_hh[:self.n_neurons//2,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[self.n_neurons//2:,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[self.n_neurons//2:,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:self.n_neurons//2,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh




    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        if self.symmetric_weights:
            print("Matching weights for left and right ALM")

            self.w_xh_linear_right_alm.weight = self.w_xh_linear_left_alm.weight
        elif 'fixed_input' in self.configs['train_type']:
            print("Fixed input weights for left and right ALM")
            self.w_xh_linear_right_alm.weight.data = torch.tensor([[1.0,0.0],[0.0,1.0]], dtype=torch.float32) # / math.sqrt(self.n_neurons)
            self.w_xh_linear_left_alm.weight.data = torch.tensor([[1.0,0.0],[0.0,1.0]], dtype=torch.float32) # / math.sqrt(self.n_neurons)
            
        else:
            init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)

        # Set all values in readout_linear to be the same value drawn from normal distribution
        if self.symmetric_weights:
            val = torch.normal(mean=0.0, std=1.0/math.sqrt(self.n_neurons), size=(1,))
            with torch.no_grad():
                self.readout_linear.weight.fill_(val.item())
        else:
            init.normal_(self.readout_linear.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
        init.constant_(self.readout_linear.bias, 0.0)

    def apply_pert(self, h, left_pert_trial_inds, right_pert_trial_inds):
        '''
        For each trial, we sample drop_p from [drop_p_min, drop_p_max]. Then, sample drop_p fraction of neurons to silence during the stim period.
        '''
        n_trials, n_neurons = h.size()


        '''
        Construct left_pert_mask
        '''
        n_left_pert_trials = len(left_pert_trial_inds)

        left_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_left_pert_trials) # (n_left_per_trials)


        left_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_left_pert_trials):
            cur_drop_p = left_pert_drop_ps[i]
            left_pert_neuron_inds = np.random.permutation(self.n_left_neurons)[:int(self.n_left_neurons*cur_drop_p)]
            left_pert_mask[left_pert_trial_inds[i],self.left_alm_inds[left_pert_neuron_inds]] = True


        '''
        Construct right_pert_mask
        '''
        n_right_pert_trials = len(right_pert_trial_inds)

        right_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_right_pert_trials) # (n_right_per_trials)

        right_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_right_pert_trials):
            cur_drop_p = right_pert_drop_ps[i]
            right_pert_neuron_inds = np.random.permutation(self.n_right_neurons)[:int(self.n_right_neurons*cur_drop_p)]
            right_pert_mask[right_pert_trial_inds[i],self.right_alm_inds[right_pert_neuron_inds]] = True


        # left pertubation
        h[np.nonzero(left_pert_mask)] = 0

        # right pertubation
        h[np.nonzero(right_pert_mask)] = 0



    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 1) or (n_trials, T, 2)

        Output:
        hs: (n_trials, T, n_neurons/3)
        zs: (n_trials, T, 1)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons//3)
        hs = []
        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)

        if self.symmetric_weights:
            xs_noise_left_alm = xs_noise_right_alm
        
        if self.one_hot:
            # Input masks - now need to match 2D input
            xs_left_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_left_alm_drop_p).float().to(xs.device)
            xs_right_alm_mask = (torch.rand(n_trials,1,2) >= self.xs_right_alm_drop_p).float().to(xs.device)
        else:
            # input trial mask
            xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
            xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)

        if self.corrupt:
            
            corr_level = self.corruption_noise
            if self.one_hot:
                # xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*(torch.randn_like(xs) + 2.0) # shift the mean of the gaussian to match the mean of the input
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*(torch.randn_like(xs)*corr_level + 0.0) # shift the mean of the gaussian to match the mean of the input

            elif self.corruption_type == "poisson":
                # xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.poisson(torch.ones_like(xs) * corr_level)
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*torch.poisson(torch.ones_like(xs.cpu()) * corr_level).to(xs.device)

            else:
                xs_noise_left_alm_corr = math.sqrt(2/self.a)*corr_level*torch.randn_like(xs)

            xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr)
            # Keep right side unchanged
            xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)

            # xs_injected_left_alm = self.w_xh_linear_left_alm((xs*xs_left_alm_mask + xs_noise_left_alm_corr)*self.xs_left_alm_amp) # Multiply term outside of everything
            # xs_injected_right_alm = self.w_xh_linear_right_alm((xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp)

        else:
            # print("no corruption ", self.xs_left_alm_amp, self.xs_right_alm_amp)
            xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
            xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)

            # xs_injected_left_alm = self.w_xh_linear_left_alm((xs*xs_left_alm_mask + xs_noise_left_alm)*self.xs_left_alm_amp) # Multiply term outside of everything
            # xs_injected_right_alm = self.w_xh_linear_right_alm((xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp)

            # xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp)
            # xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp)

        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]

        # Bi stim pert
        if self.bi_pert_trials_prob is not None:
            n_trials = xs.size(0)
            bi_pert_trial_inds = np.random.permutation(n_trials)[:int(self.bi_pert_trials_prob*n_trials)]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            if t >= self.pert_begin and t <= self.pert_end:
                if self.bi_pert_trials_prob is None:
                    self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)
                else:
                    self.apply_bi_pert(h, bi_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs = self.readout_linear(hs)  # (n_trials, T, 1)

        if self.return_input:
            if self.corrupt:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm_corr, 
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm_corr)*self.xs_left_alm_amp, 
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs # xs: (n_trials, T, 1) or (n_trials, T, 2)
            else:
                return (xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm,
                xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm), hs, zs
                # return ((xs*xs_left_alm_mask + xs_noise_left_alm)*self.xs_left_alm_amp,
                # (xs*xs_right_alm_mask + xs_noise_right_alm)*self.xs_right_alm_amp), hs, zs
        else:
            return hs, zs




class TwoHemiRNNCellGeneral(nn.Module):
    '''
    Same as TwoHemiRNNCellGeneral except that we separately store within-hemi and cross-hemi weights, so that
    they can easily trained separately.
    '''

    def __init__(self, n_neurons=128, a=0.2, sigma=0.05, nonlinearity=nn.Tanh(), zero_init_cross_hemi=False,\
        init_cross_hemi_rel_factor=1, bias=True, symmetric_weights=False, noise=True):
        super().__init__()
        self.n_neurons = n_neurons
        self.a = a
        self.sigma = sigma
        self.symmetric_weights = symmetric_weights
        self.noise = noise
        self.nonlinearity = nonlinearity
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = init_cross_hemi_rel_factor

        self.bias = bias

        self.w_hh_linear_ll = nn.Linear(n_neurons//2, n_neurons//2, bias=self.bias)
        self.w_hh_linear_rr = nn.Linear(n_neurons-n_neurons//2, n_neurons-n_neurons//2, bias=self.bias)

        self.w_hh_linear_lr = nn.Linear(n_neurons//2, n_neurons-n_neurons//2, bias=False)
        self.w_hh_linear_rl = nn.Linear(n_neurons-n_neurons//2, n_neurons//2, bias=False)


        self.init_params()

    def init_params(self):

        if self.symmetric_weights:
            init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
            self.w_hh_linear_rr.weight = self.w_hh_linear_ll.weight
        else:
            init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
            init.normal_(self.w_hh_linear_rr.weight, 0.0, 1.0/math.sqrt(self.n_neurons))

        if self.zero_init_cross_hemi:
            init.constant_(self.w_hh_linear_lr.weight, 0.0)
            init.constant_(self.w_hh_linear_rl.weight, 0.0)

        else:
            init.normal_(self.w_hh_linear_lr.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))
            init.normal_(self.w_hh_linear_rl.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))


        if self.bias:
            init.constant_(self.w_hh_linear_ll.bias, 0.0)
            init.constant_(self.w_hh_linear_rr.bias, 0.0)


    def full_recurrent(self, h_pre):
        h_pre_left = h_pre[:,:self.n_neurons//2]
        h_pre_right = h_pre[:,self.n_neurons//2:]

        h1 = torch.cat([self.w_hh_linear_ll(h_pre_left), self.w_hh_linear_lr(h_pre_left)], 1) # (n_neurons)
        h2 = torch.cat([self.w_hh_linear_rl(h_pre_right), self.w_hh_linear_rr(h_pre_right)], 1) # (n_neurons)

        return h1 + h2


    def forward(self, x_injected, h_pre):
        '''
        Input:
        x_injected: (n_trials, n_neurons)
        h_pre: (n_trials, n_neurons)

        Output:
        h: (n_trials, n_neurons)
        '''

        if self.noise:
            noise = math.sqrt(2/self.a)*self.sigma*torch.randn_like(x_injected)
        else:
            noise = 0.0
        if self.nonlinearity is not None:
            h = (1-self.a)*h_pre + self.a*self.nonlinearity(self.full_recurrent(h_pre) + x_injected + noise)
        else:
            h = (1-self.a)*h_pre + self.a*(self.full_recurrent(h_pre) + x_injected + noise)

        return h



class TwoHemiRNNCellGeneral_asymmetric(nn.Module):
    '''
    Same as TwoHemiRNNCellGeneral except that we separately store within-hemi and cross-hemi weights, so that
    they can easily trained separately.
    Also there are only 2 neurons in the left hemisphere and more in the right hemisphere
    '''

    def __init__(self, n_neurons=128, a=0.2, sigma=0.05, nonlinearity=nn.Tanh(), zero_init_cross_hemi=False,\
        init_cross_hemi_rel_factor=1, bias=True, symmetric_weights=False, noise=True):
        super().__init__()
        self.n_neurons = n_neurons # here n neurosn only refers to the right hemi
        self.n_left_neurons = 2 #self.n_neurons//2
        self.n_right_neurons = self.n_neurons 
        self.a = a
        self.sigma = sigma
        self.symmetric_weights = symmetric_weights
        self.noise = noise
        self.nonlinearity = nonlinearity
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = init_cross_hemi_rel_factor

        self.bias = bias

        self.w_hh_linear_ll = nn.Linear(2, 2, bias=self.bias)
        self.w_hh_linear_rr = nn.Linear(self.n_neurons, self.n_neurons, bias=self.bias)

        self.w_hh_linear_lr = nn.Linear(2, self.n_neurons, bias=False)
        self.w_hh_linear_rl = nn.Linear(self.n_neurons, 2, bias=False)


        self.init_params()

    def init_params(self):

        if self.symmetric_weights:
            init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
            self.w_hh_linear_rr.weight = self.w_hh_linear_ll.weight
        else:
            init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_left_neurons))
            init.normal_(self.w_hh_linear_rr.weight, 0.0, 1.0/math.sqrt(self.n_right_neurons))

            # Normalize w_hh_linear_rr weights so that their absolute sum is 3
            with torch.no_grad():
                weight = self.w_hh_linear_rr.weight
                abs_sum = torch.sum(torch.abs(weight))
                if abs_sum > 0:
                    weight.mul_(3.6 / abs_sum)

        if self.zero_init_cross_hemi:
            init.constant_(self.w_hh_linear_lr.weight, 0.0)
            init.constant_(self.w_hh_linear_rl.weight, 0.0)

        else:
            init.normal_(self.w_hh_linear_lr.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_left_neurons + self.n_right_neurons))
            init.normal_(self.w_hh_linear_rl.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_left_neurons + self.n_right_neurons))


        if self.bias:
            init.constant_(self.w_hh_linear_ll.bias, 0.0)
            init.constant_(self.w_hh_linear_rr.bias, 0.0)


    def full_recurrent(self, h_pre):
        h_pre_left = h_pre[:,:self.n_left_neurons]
        h_pre_right = h_pre[:,self.n_left_neurons:]

        h1 = torch.cat([self.w_hh_linear_ll(h_pre_left), self.w_hh_linear_lr(h_pre_left)], 1) # (n_neurons)
        h2 = torch.cat([self.w_hh_linear_rl(h_pre_right), self.w_hh_linear_rr(h_pre_right)], 1) # (n_neurons) 

        return h1 + h2


    def forward(self, x_injected, h_pre):
        '''
        Input:
        x_injected: (n_trials, n_neurons)
        h_pre: (n_trials, n_neurons)

        Output:
        h: (n_trials, n_neurons)
        '''

        if self.noise:
            noise = math.sqrt(2/self.a)*self.sigma*torch.randn_like(x_injected)
        else:
            noise = 0.0
        if self.nonlinearity is not None:
            h = (1-self.a)*h_pre + self.a*self.nonlinearity(self.full_recurrent(h_pre) + x_injected + noise)
        else:
            h = (1-self.a)*h_pre + self.a*(self.full_recurrent(h_pre) + x_injected + noise)

        return h





class TwoHemiRNNCellSparse(nn.Module):
    '''
    Same as TwoHemiRNNCellGeneral except that we separately store within-hemi and cross-hemi weights, so that
    they can easily trained separately.

    Additionally, we separate the units that are: input vs crosshemi vs readout weights
    '''

    def __init__(self, n_neurons=128, a=0.2, sigma=0.05, nonlinearity=nn.Tanh(), zero_init_cross_hemi=False,\
        init_cross_hemi_rel_factor=1, bias=True, symmetric_weights=False, noise=True):
        super().__init__()
        self.n_neurons = n_neurons
        self.a = a
        self.sigma = sigma
        self.symmetric_weights = symmetric_weights
        self.noise = noise
        self.nonlinearity = nonlinearity
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = init_cross_hemi_rel_factor

        self.bias = bias

        self.w_hh_linear_ll = nn.Linear(n_neurons//2, n_neurons//2, bias=self.bias)
        self.w_hh_linear_rr = nn.Linear(n_neurons-n_neurons//2, n_neurons-n_neurons//2, bias=self.bias)

        self.w_hh_linear_lr = nn.Linear(n_neurons//6, n_neurons//6, bias=False)
        self.w_hh_linear_rl = nn.Linear(n_neurons//6, n_neurons//6, bias=False)


        self.init_params()

    def init_params(self):

        if self.symmetric_weights:
            init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
            self.w_hh_linear_rr.weight = self.w_hh_linear_ll.weight
        else:
            init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
            init.normal_(self.w_hh_linear_rr.weight, 0.0, 1.0/math.sqrt(self.n_neurons))

        if self.zero_init_cross_hemi:
            init.constant_(self.w_hh_linear_lr.weight, 0.0)
            init.constant_(self.w_hh_linear_rl.weight, 0.0)

        else:
            init.normal_(self.w_hh_linear_lr.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))
            init.normal_(self.w_hh_linear_rl.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))


        if self.bias:
            init.constant_(self.w_hh_linear_ll.bias, 0.0)
            init.constant_(self.w_hh_linear_rr.bias, 0.0)


    def full_recurrent(self, h_pre):
        h_pre_left = h_pre[:,:self.n_neurons//6] # neurons 0 to 1/6
        h_pre_right = h_pre[:,self.n_neurons//2:(self.n_neurons//2)+(self.n_neurons//6)] # neurons 1/2 to (1/2 + 1/6)

        h1 = torch.cat([self.w_hh_linear_ll(h_pre_left), self.w_hh_linear_lr(h_pre_left)], 1) # (n_neurons)
        h2 = torch.cat([self.w_hh_linear_rl(h_pre_right), self.w_hh_linear_rr(h_pre_right)], 1) # (n_neurons)

        return h1 + h2


    def forward(self, x_injected, h_pre):
        '''
        Input:
        x_injected: (n_trials, n_neurons/3)
        h_pre: (n_trials, n_neurons/3)

        Output:
        h: (n_trials, n_neurons/3)
        '''

        if self.noise:
            noise = math.sqrt(2/self.a)*self.sigma*torch.randn_like(x_injected)
        else:
            noise = 0.0
        if self.nonlinearity is not None:
            h = (1-self.a)*h_pre + self.a*self.nonlinearity(self.full_recurrent(h_pre) + x_injected + noise)
        else:
            h = (1-self.a)*h_pre + self.a*(self.full_recurrent(h_pre) + x_injected + noise)

        return h
