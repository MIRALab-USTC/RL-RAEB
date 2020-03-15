import abc
import mbrl.torch_modules.utils as ptu

class Processor(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, shape):
        pass

    @abc.abstractmethod
    def process(self, x):
        pass

    def process_np(self, x, **kwarg):
        x = ptu.from_numpy(x)
        return ptu.get_numpy(self.process(x, **kwarg)) 

    @abc.abstractmethod
    def recover(self, x):
        pass

    def recover_np(self, x, **kwarg):
        x = ptu.from_numpy(x)
        return ptu.get_numpy(self.recover(x, **kwarg)) 

class Identity(Processor, metaclass=abc.ABCMeta):
    def __init__(self, shape):
        self.input_shape = self.output_shape = shape

    def process(self, x):
        return x

    def process_np(self, x):
        return x

    def recover(self, x):
        return x
    
    def recover_np(self, x):
        return x