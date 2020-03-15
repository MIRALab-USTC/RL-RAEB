from collections import OrderedDict
import numpy as np

def to_list(x, length=None):
    x = x if x is not None else []
    x = x if type(x) is list else [x]
    if length is not None:
        while len(x) < length:
            x.append(x[-1])
        x = x[:length]
    return x

def get_valid_part(item, valid):
    if type(item) is list:
        new_item = []
        for i,v in zip(item, valid):
            if bool(v):
                new_item.append(i)
    elif type(item) is np.ndarray:
        new_item = item[valid.astype(np.bool)]
    elif type(item) is dict or type(item) is OrderedDict:
        new_item = OrderedDict()
        for key in item:
            new_item[key] = get_valid_part(item[key], valid)
    else:
        raise NotImplementedError
    return new_item

def split_item(item, valid_num):
    if type(item) is list or type(item) is np.ndarray:
        new_item = item[:valid_num]
        res_item = item[valid_num:]
    elif type(item) is dict or type(item) is OrderedDict:
        new_item = OrderedDict()
        res_item = OrderedDict()
        for key in item:
            new_item[key], res_item[key] = split_item(item[key], valid_num)
    else:
        raise NotImplementedError
    return new_item, res_item

def split_items(data, valid_num):
    data = [split_item(item, valid_num) for item in data]
    new_data, res_data = list(zip(*data))
    return new_data, res_data

def combine_item(item, cur_item):
    if type(item) is list:
        assert type(cur_item) is list
        new_item = item + cur_item
    elif type(item) is np.ndarray:
        assert type(cur_item) is np.ndarray
        new_item = np.concatenate([item, cur_item])
    elif type(item) is dict or type(item) is OrderedDict:
        new_item = OrderedDict()
        for key in item:
            assert key in cur_item
            new_item[key] = combine_item(item[key], cur_item[key])
    else:
        raise NotImplementedError
    return new_item

def combine_items(data, cur_data):
    new_data = [combine_item(item, cur_item) for item,cur_item in zip(data,cur_data)]
    return tuple(new_data)

def format_for_process(params):
    new_params = []
    for k, v in params.items():
        k = k.rjust(18)
        v = '{:12}'.format(v)
        new_params.append([k, v])
    return new_params

    