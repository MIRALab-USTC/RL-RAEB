import torch
import os.path as osp
import torch.nn as nn

import mbrl.torch_modules.utils as ptu
from mbrl.utils.misc_untils import to_list
import copy

class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_layers=[128,128], 
                 ensemble_size=None, 
                 nonlinearity='relu', 
                 output_nonlinearity='identity',
                 module_name='mlp',
                 **fc_kwargs
                 ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = len(hidden_layers)
        self.ensemble_size = ensemble_size

        self.output_nonlinearity = output_nonlinearity
        self.nonlinearities = to_list(nonlinearity, length=self.num_layers)
        self.nonlinearities.append(output_nonlinearity)
        self.activation_functions = [ptu.get_nonlinearity(nl) for nl in self.nonlinearities]

        self.layers = layers = [input_size] + hidden_layers + [output_size]
        self.fcs = []
        for i in range(len(layers)-1):
            fc = ptu.EnsembleLinear(layers[i], 
                                    layers[i+1], 
                                    ensemble_size, 
                                    which_nonlinearity=self.nonlinearities[i],
                                    **fc_kwargs)
            setattr(self, 'layer%d'%i, fc)
            self.fcs.append(fc)
        self.module_name = module_name
    
    def get_snapshot(self, key_must_have=''):
        new_state_dict = {}
        state_dict = self.state_dict()
        if key_must_have == '':
            new_state_dict = state_dict
        else:
            for k,v in state_dict.items():
                if key_must_have in k:
                    new_state_dict[k] = v
        return new_state_dict

    def load_snapshot(self, loaded_state_dict, key_must_have=''):
        state_dict = self.state_dict()
        if key_must_have == '':
            state_dict = loaded_state_dict
        else:
            for k,v in loaded_state_dict.items():
                if key_must_have in k:
                    state_dict[k] = v
        self.load_state_dict(state_dict)

    def save(self, save_dir, net_id=None):
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(save_dir, '%s.pt'%self.module_name)
        else:
            net_name = 'net%d'%net_id
            file_path = osp.join(save_dir, '%s_%s.pt'%(self.module_name, net_name))
        state_dict = self.get_snapshot(net_name)
        torch.save(state_dict, file_path)
    
    def load(self, load_dir, net_id=None):
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(load_dir, '%s.pt'%self.module_name)
        else:
            net_name = 'net%d'%net_id
            file_path = osp.join(load_dir, '%s_%s.pt'%(self.module_name, net_name))
            if not osp.exists(file_path):
                file_path = osp.join(load_dir, '%s.pt'%self.module_name)
        loaded_state_dict = torch.load(file_path)
        self.load_snapshot(loaded_state_dict, net_name)

    def forward(self, x):
        for fc,act_f in zip(self.fcs, self.activation_functions):
            x = fc(x)
            x = act_f(x)
        return x

    def get_weight_decay(self, weight_decays):
        weight_decays = to_list(weight_decays, len(self.fcs))
        weight_decay_tensors = []
        for weight_decay, fc in zip(weight_decays, self.fcs):
            weight_decay_tensors.append(fc.get_weight_decay(weight_decay))
        return sum(weight_decay_tensors)

class NoisyMLP(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_layers=[128,128], 
                 ensemble_size=None, 
                 nonlinearity='relu', 
                 with_noise=True,
                 output_nonlinearity='identity',
                 output_with_noise=True,
                 module_name='noisy_mlp',
                 **noisy_fc_kwargs
                 ):
        nn.Module.__init__(self)
        plain_fc_kwargs = copy.deepcopy(noisy_fc_kwargs)
        plain_fc_kwargs.pop('noise_type',{})
        plain_fc_kwargs.pop('global_noise',{})
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = len(hidden_layers)
        self.ensemble_size = ensemble_size

        self.output_nonlinearity = output_nonlinearity
        self.nonlinearities = to_list(nonlinearity, length=self.num_layers)
        self.nonlinearities.append(output_nonlinearity)
        self.activation_functions = [ptu.get_nonlinearity(nl) for nl in self.nonlinearities]

        self.output_noise = output_with_noise
        self.layer_noises = to_list(with_noise, length=self.num_layers)
        self.layer_noises.append(output_with_noise)

        self.layers = layers = [input_size] + hidden_layers + [output_size]
        self.fcs = []
        for i in range(len(layers)-1):
            if self.layer_noises[i]:
                fc = ptu.NoisyEnsembleLinear(layers[i], 
                                             layers[i+1], 
                                             ensemble_size, 
                                             which_nonlinearity=self.nonlinearities[i],
                                             **noisy_fc_kwargs)
            else:
                fc = ptu.EnsembleLinear(layers[i], 
                                        layers[i+1], 
                                        ensemble_size, 
                                        which_nonlinearity=self.nonlinearities[i],
                                        **plain_fc_kwargs)
            setattr(self, 'layer%d'%i, fc)
            self.fcs.append(fc)
        self.module_name = module_name
    
    def forward(self, 
                x, 
                deterministic=False,
                reparameterize=True):
        for fc,act_f,noisy in zip(self.fcs, 
                                  self.activation_functions,
                                  self.layer_noises):
            if noisy:
                x = fc(x, deterministic, reparameterize)
            else:
                x = fc(x)
            x = act_f(x)
        return x
