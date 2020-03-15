import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
import numpy as np

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
 
swish = Swish()

def get_activation(act_name='relu'):
    activation_dict = {
        'relu': F.relu,
        'swish': swish,
        'tanh': torch.tanh,
    }
    return activation_dict[act_name]

def get_affine_params(input_size, output_size):
    scale = np.sqrt(2/input_size)
    #w_data = truncnorm.rvs(-2*scale, 2*scale, scale=scale, size=(input_size, output_size))
    w_data = np.random.randn(input_size, output_size) * scale
    w = nn.Parameter(torch.FloatTensor(w_data))
    b = nn.Parameter(torch.zeros(1, output_size))
    return w, b

class FC(nn.Module):
    def __init__(self, input_size, output_size, ensemble_size=None):
        super(FC, self).__init__()
        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        if ensemble_size is None:
            self.w, self.b = get_affine_params(input_size, output_size)
        else:
            self.ws, self.bs = [], []
            for i in range(ensemble_size):
                w_name, b_name = 'w_net%d'%i, 'b_net%d'%i
                w, b = get_affine_params(input_size, output_size)
                setattr(self, w_name, w)
                setattr(self, b_name, b)
                self.ws.append(w)
                self.bs.append(b)

    def forward(self, x):
        if self.ensemble_size is None:
            w,b = self.w,self.b
        else:
            w = torch.stack(self.ws, 0)
            b = torch.stack(self.bs, 0)
        return x.matmul(w) + b

    def __repr__(self):
        ensemble_size = '[ensemble size: '+str(self.ensemble_size)+']' if self.ensemble_size else ''
        return "FC(%d, %d)%s"%(self.input_size, self.output_size, ensemble_size)

    def get_weight_decay(self, weight_decay=5e-5):
        return (self.w ** 2).sum() * weight_decay * 0.5

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


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
