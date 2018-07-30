from copy import deepcopy
from json import dump
from pickle import dump as pkl_dump

from modeling.segmentation_utils import convert_dict_keys


def write_dict_to_file(d, fname, convert_keys):
    print(('writing dict to file {}...'.format(fname)))

    # deepcopy so original variable isn't mutated
    d_ = deepcopy(d)

    if convert_keys:  # Convert keys from tuples to strings
        if 'mixed_effects' in list(d_.keys()):
            for k in list(d_['mixed_effects'].keys()):
                convert_dict_keys(d_['mixed_effects'][k], to_string=True)
        else:
            convert_dict_keys(d_, to_string=True)

    with open(fname, 'w') as f:
        dump(d_, f)

    print('done')


def write_model_dict(d, fname, convert_keys):
    write_dict_to_file(d, fname, convert_keys)


def write_pickle(fname, obj):
    # write arbitrary object to pickle (expected to be nested dict with quantile regression models)
    print(('Writing pickle to file {}...'.format(fname)))

    with open(fname, 'w+') as f:
        pkl_dump(obj, f)

    print('Done')
