import torch
import os.path as osp
import torch.nn as nn

if __name__ == "__main__":
    import sys
    import os
    mbrl_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(mbrl_dir)

import mbrl.torch_modules.utils as ptu
from mbrl.utils.misc_untils import to_list

class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_layers=[128,128], 
                 ensemble_size=None, 
                 nonlinearity='relu', 
                 output_nonlinearity='identity',
                 module_name='MLP',
                 **fc_kwargs
                 ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.num_layers = len(hidden_layers)

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

if __name__ == "__main__":
    from mbrl.torch_modules.torch_normalizer import TorchNormalizer
    import numpy as np
    
    class TestMLP(MLP):
        def __init__(self, N):
            super(TestMLP, self).__init__(1,3,[3,3],2,'tanh')
            self.N = N
        def forward(self,x):
            return super(TestMLP, self).forward(self.N(x))

    normalizer=TorchNormalizer((1,))
    x=ptu.FloatTensor([[3]])
    mlp=TestMLP(normalizer)
    print(mlp.state_dict())
    print(mlp(x))
    print('\n\n')
    mlp.save('/home/qizhou',1)
    mlp.save('/home/qizhou')

    mlp2=TestMLP(normalizer)
    print(mlp2.state_dict())
    print(mlp2(x))
    print('\n\n')
    mlp2.load('/home/qizhou',1)
    print(mlp2.state_dict())
    print(mlp2(x))
    print('\n\n')
    mlp2.load('/home/qizhou')
    print(mlp2.state_dict())
    print(mlp2(x))
    print('\n\n')

    normalizer.set_mean_std_np(np.array([2]), np.array([1.5]))
    print(mlp.state_dict())
    print(mlp(x))
    print('\n\n')
    print(mlp2.state_dict())
    print(mlp2(x))
    print('\n\n')
    mlp2.load('/home/qizhou')
    print(mlp2.state_dict())
    print(mlp(x))
    x=ptu.FloatTensor([[[3]],[[6]]])
    print(mlp(x))
    x=ptu.FloatTensor([[6]])
    print(mlp(x))

