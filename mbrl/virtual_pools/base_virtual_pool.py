import abc

class VirtualPool(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, env):
        pass

    @abc.abstractmethod
    def add_samples(self, samples):
        pass

    @abc.abstractmethod
    def update_hash_table(self, states):
        pass


    @abc.abstractmethod
    def clear(self):
        # 清空pool
        pass

    def get_diagnostics(self):
        return {}

    
