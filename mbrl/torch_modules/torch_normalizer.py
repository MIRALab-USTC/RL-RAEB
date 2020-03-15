import torch
import torch.nn as nn
import mbrl.torch_modules.utils as ptu

class TorchNormalizer(nn.Module):
    def __init__(self, shape, epsilon=1e-6):
        super(TorchNormalizer, self).__init__()
        self.epsilon = epsilon
        self.mean = nn.Parameter(ptu.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(ptu.ones(shape), requires_grad=False)
    
    def forward(self, x):
        return self.normalize(x)

    def normalize(self, x):    
        return (x-self.mean) / (self.std + self.epsilon)

    def denormalize(self, x):
        return x * (self.std + self.epsilon) + self.mean

    def set_mean_std_np(self, mean, std):
        self.mean.data = ptu.from_numpy(mean)
        self.std.data = ptu.from_numpy(std)

    def mean_std_np(self):
        return ptu.get_numpy(self.mean.data), ptu.get_numpy(self.std.data)
