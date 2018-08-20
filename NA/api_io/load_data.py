import json
from os import path as os_path
from os.path import join as os_join, dirname, abspath
from pickle import loads as pkl_loads
from zlib import decompress
from pandas import read_pickle

from NA.modeling.segmentation_utils import convert_dict_keys

try:
    _ = os_path.abspath(__file__)
except NameError:
    DATA_PATH = '/Users/jbrubaker/projects/copra_project/copra_hw/models'
else:
    DATA_PATH = os_join(dirname(dirname(abspath(__file__))), 'models')


def load_segmentation(reg):
    dat_path = os_join(DATA_PATH, reg)

    fname = os_join(dat_path, 'segmentation.json')

    with open(fname, 'r') as f:
        seg_dict = json.load(f)

    # Convert keys from strings to tuples
    convert_dict_keys(seg_dict, to_string=False)

    return seg_dict


def load_pickle(fname, compressed):
    if compressed:
        with open(fname, 'rb') as fp:
            data = decompress(fp.read())
            obj = pkl_loads(data)
    else:
        obj = read_pickle(fname)

    return obj


def load_model_dict(fname):
        # new model definition
        with open(fname, 'r') as f:
            model_def = json.load(f)

        # convert keys back
        if 'mixed_effects' in list(model_def.keys()):  # new model format
            convert_dict_keys(model_def['mixed_effects'], to_string=False)
        else:  # old model format
            convert_dict_keys(model_def, to_string=False)

        return model_def


def load_model(reg, switch='comp', compressed=False):
    dat_path = os_join(DATA_PATH, reg)

    if switch == 'comp':
        fname = 'comp_model.json'
        fpath = os_join(dat_path, fname)
        model = load_model_dict(fpath)
    elif switch == 'q_quant':
        fname = 'quote_quant.pkl'
        if compressed and '.gz' not in fname:
            fname += '.gz'
        fpath = os_join(dat_path, fname)
        model = load_pickle(fpath, compressed)
    elif switch == 'c_quant':
        fname = 'comp_quant.pkl'
        if compressed and '.gz' not in fname:
            fname += '.gz'
        fpath = os_join(dat_path, fname)
        model = load_pickle(fpath, compressed)
    else:
        print(('Switch must be either comp or quote. Received {}'.format(switch)))
        model = None

    return model
