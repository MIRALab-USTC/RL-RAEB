from mbrl.processors.base_processor import Processor
from mbrl.torch_modules.torch_normalizer import TorchNormalizer

class Normalizer(Processor):
    def __init__(self, shape, epsilon=1e-6):
        self.input_shape = self.output_shape = shape
        self.epsilon = epsilon
        self.module = TorchNormalizer(shape, epsilon)

    def set_mean_std_np(self, mean, std):
        self.module.set_mean_std_np(mean, std)

    def mean_std_np(self):
        return self.module.mean_std_np()

    def process(self, x):
        return self.module.normalize(x)

    def recover(self, x):
        return self.module.denormalize(x)