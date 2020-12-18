import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# import sys
# sys.path.insert(0, "/home/rl_shared/zhihaiwang/research/mbrl_sparse_reward")
from mbrl.models.base_model_without_reward import Model

ACTIVATION_DICT = {
    'relu': F.relu,
}

INIT_DICT = {
    'kaiming_uniform_': nn.init.kaiming_uniform_,
    'kaiming_normal_': nn.init.kaiming_normal_
}

class MyLinear(nn.Linear):
    def __init__(self, i_size, o_size, init_mode):
        self.init_mode = init_mode
        nn.Linear.__init__(self, i_size, o_size)

    def reset_parameters(self):
        init_func = INIT_DICT[self.init_mode]
        init_func(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class RandomNet(nn.Module, Model):
    def __init__(self, env, input_mode, init_mode, hidden_layers, activation='relu', output_shape=None):
        super(RandomNet, self).__init__()
        if input_mode == "state":
            input_shape = env.observation_space.shape[0]
        elif input_mode == "state_action":
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0]
        if output_shape is None:
            output_shape = env.observation_space.shape[0]
        self.input_mode = input_mode
        self.activation = activation
        
        layers = [input_shape] + hidden_layers + [output_shape]
        self.fcs = nn.ModuleList([MyLinear(layers[i], layers[i+1], init_mode) for i in range(len(layers)-1)])

    def forward(self, x):
        # input: batch*input_size
        # output: ensemble_size*batch*output_size
        activation = ACTIVATION_DICT[self.activation]

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i != len(self.fcs) - 1:
                x = activation(x)
        return x

if __name__ == "__main__":
    # for test
    import gym
    '''
    print("******start*********")
    env = gym.make('Ant-v2')
    random_net = RandomNet(env, "state", "kaiming_uniform_", [256,256])
    random_target = RandomNet(env, "state", "kaiming_normal_", [256,256])
    x = env.reset()
    x = torch.FloatTensor(x)
    y = random_net(x)
    y_tar = random_target(x)
    print((y-y_tar)**2)
    print(torch.sum((y-y_tar)**2))
    '''
    for c in Model.__subclasses__():
        print(c.__name__)

    # print(y)