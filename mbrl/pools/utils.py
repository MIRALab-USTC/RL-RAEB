import numpy as np
from collections import OrderedDict

def get_unnested_data(data, pre_key=''):
    inner_dict = {}
    keys = list(data.keys())
    for key in keys:
        value = data.pop(key)
        if type(value) in [OrderedDict, dict]:
            inner_dict.update(get_unnested_data(value, pre_key=pre_key+key+'.'))
        else:
            data[pre_key+key] = value
    data.update(inner_dict)
    return data

def get_valid_dataset_size(dataset, keys=None):
    if keys is None:
        keys = list(dataset.keys())
    min_size = np.inf
    for key in keys:
        value = dataset[key]
        cur_size = len(value)
        if cur_size < min_size:
            min_size = cur_size
    return min_size

def get_batch(dataset, batch_index, keys=None):
    if keys is None:
        keys = list(dataset.keys())
    batch = {}
    for key in keys:
        value = dataset[key]
        batch[key] = value[batch_index]
    return batch

def _get_ensemble_batch(dataset, batch_indexes, ensemble_size, keys=None):
    if keys is None:
        keys = list(dataset.keys())
    batch = {}
    for key in keys:
        value = dataset[key]
        batch[key] = [value[batch_indexes[i]] for i in range(ensemble_size)]
        batch[key] = np.stack(batch[key])
    return batch

    
def _random_batch_independently(dataset, batch_size, valid_size, keys=None):
    batch_index = np.random.randint(0, valid_size, batch_size)
    return get_batch(dataset, batch_index, keys=keys)

def random_batch_ensemble(dataset, batch_size, valid_size, ensemble_size, keys=None):
    indices = [np.random.randint(0, valid_size, batch_size) for _ in range(ensemble_size)]
    return _get_ensemble_batch(dataset, indices, ensemble_size, keys)

def random_batch_independently(dataset, batch_size, keys=None):
    valid_size = get_valid_dataset_size(dataset, keys=keys)
    return _random_batch_independently(dataset, batch_size, valid_size, keys)

def _shuffer_and_random_batch(dataset, batch_size, valid_size, keys=None):
    _batch_index = np.random.permutation(np.arange(valid_size))
    ts = 0 
    while ts < valid_size:
        te = ts + batch_size
        if te + batch_size > valid_size:
            te += batch_size
        yield get_batch(dataset, _batch_index[ts:te], keys=keys)
        ts = te

def shuffer_and_random_batch(dataset, batch_size, keys=None):
    valid_size = get_valid_dataset_size(dataset, keys=keys)
    for batch in _shuffer_and_random_batch(dataset, batch_size, valid_size, keys):
        yield batch

def _shuffer_and_random_batch_model(dataset, batch_size, valid_size, ensemble_size, keys=None):
    _batch_indexes = np.array([np.random.permutation(np.arange(valid_size)) for _ in range(ensemble_size)])
    ts = 0 
    while ts < (valid_size-1):
        te = ts + batch_size
        if te > valid_size:
            te = valid_size-1
        yield _get_ensemble_batch(dataset, _batch_indexes[:, ts:te], ensemble_size, keys=keys)
        ts = te