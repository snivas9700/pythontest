from copy import deepcopy
from json import dump
from pickle import dumps as cpkl_dumps, HIGHEST_PROTOCOL
from zlib import compress
from pandas import to_pickle

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


def write_pickle(fname, obj, compressed=False):

    if compressed:
        # write arbitrary object to pickle (expected to be nested dict with quantile regression models)
        print(('Writing pickle to (compressed) file {}...'.format(fname)))

        # because of model filesize, have to resort to compressed files
        with open(fname, 'wb') as fp:
            fp.write(compress(cpkl_dumps(obj, HIGHEST_PROTOCOL), 9))

    else:
        print(('Writing pickle to file {}...'.format(fname)))
        to_pickle(obj, fname)

    print('Done')
