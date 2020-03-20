import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math


"""
GPU wrappers
"""
_use_gpu = False
device = None
_gpu_id = 0

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")
    if _use_gpu:
        set_device(gpu_id)

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)

def LongTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.LongTensor(*args, **kwargs, device=torch_device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)

def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)

def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)

def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)

def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)

def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs).to(device)

############################ our utils ############################

class Swish(nn.Module):
    def __init__(self):        
        super(Swish, self).__init__()     
    def forward(self, x):        
        x = x * F.sigmoid(x)        
        return x

class Identity(nn.Module):
    def __init__(self):        
        super(Identity, self).__init__()     
    def forward(self, x):            
        return x

swish = Swish()
identity = Identity()

def get_nonlinearity(act_name='relu'):
    nonlinearity_dict = {
        'relu': F.relu,
        'swish': swish,
        'tanh': torch.tanh,
        'identity': identity,
    }
    return nonlinearity_dict[act_name]

class EnsembleLinear(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 ensemble_size=None,
                 which_nonlinearity='relu',
                 with_bias=True,
                 init_weight_mode='uniform',
                 init_bias_constant=None,
                 ):
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.with_bias = with_bias
        self.init_weight_mode = init_weight_mode
        self.init_bias_constant = init_bias_constant
        if which_nonlinearity == 'identity':
            self.which_nonlinearity = 'linear'
        else:
            self.which_nonlinearity = which_nonlinearity
        
        if ensemble_size is None:
            self.weight, self.bias = self._creat_weight_and_bias()
        else:
            self.weights, self.biases = [], []
            for i in range(ensemble_size):
                weight, bias = self._creat_weight_and_bias()
                weight_name, bias_name = 'weight_net%d'%i, 'bias_net%d'%i
                self.weights.append(weight)
                self.biases.append(bias)
                setattr(self, weight_name, weight)
                setattr(self, bias_name, bias)
        self.reset_parameters()
    
    def _creat_weight_and_bias(self):
        weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        if self.with_bias:
            bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            bias = None
        return weight, bias

    def _reset_weight_and_bias(self, weight, bias):
        fanin_init(weight, nonlinearity=self.which_nonlinearity, mode=self.init_weight_mode)
        if bias is not None:
            if self.init_bias_constant is None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(bias, -bound, bound)
            else:
                init.constant_(bias, self.init_bias_constant)

    def reset_parameters(self):
        if self.ensemble_size is None:
            self._reset_weight_and_bias(self.weight, self.bias)
        else:
            for w,s in zip(self.weights, self.biases):
                self._reset_weight_and_bias(w, s)

    def extra_repr(self):
        return 'in_features={}, out_features={}, ensemble_size={}, bias={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.with_bias)

    def forward(self, x):
        if self.ensemble_size is None:
            w,b = self.weight, self.bias
        else:
            w = torch.stack(self.weights, 0)
            if self.biases[0] is None:
                b = None
            else:
                b = torch.stack(self.biases, 0)
        if self.with_bias:
            return x.matmul(w) + b
        else:
            return x.matmul(w)

    def get_weight_decay(self, weight_decay=5e-5):
        if self.ensemble_size is not None:
            return (self.weight ** 2).sum() * weight_decay * 0.5
        else:
            decays = []
            for w in self.weights:
                decays.append((w ** 2).sum() * weight_decay * 0.5)
            return sum(decays)

def fanin_init(tensor, nonlinearity='relu', mode='uniform'):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    else:
        raise Exception("Shape must be have dimension 2.")
    gain = init.calculate_gain(nonlinearity)
    if mode == 'uniform':
        bound = gain * math.sqrt(3.0) / np.sqrt(fan_in)
        return tensor.data.uniform_(-bound, bound)
    elif mode == 'normal':
        std = gain / np.sqrt(fan_in)
        return tensor.data.normal_(-std, std)

def fanin_init_weights_like(tensor):
    new_tensor = FloatTensor(tensor.size())
    fanin_init(new_tensor)
    return new_tensor

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()

def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v

def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }

def _elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return get_numpy(elem_or_tuple)

def torch_to_np_info(torch_info):
    return {
        k: _elem_or_tuple_to_numpy(x)
        for k, x in torch_info.items()
    }
