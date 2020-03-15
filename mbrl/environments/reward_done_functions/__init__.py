import importlib
import os.path as osp
from mbrl.utils.misc_untils import to_list
_LOCAL_DIR = osp.dirname(osp.abspath(__file__))

def get_reward_done_function(env_name, must_provide=None):
    must_provide = to_list(must_provide)
    if len(must_provide) > 0:
        file_path = osp.join(_LOCAL_DIR, env_name)
        module = importlib.import_module(file_path)
        reward_function = getattr(module, 'reward_function', None)
        done_function = getattr(module, 'done_function', None)
        for item in must_provide:
            assert item in ['reward_function', 'done_function']
            assert eval(item) is None, '%s does not have the attribution [%s]'%(env_name, item)
        return reward_function, done_function
    else:
        return None, None