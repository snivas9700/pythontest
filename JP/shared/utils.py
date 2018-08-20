from time import time
from pandas import DataFrame


# Wrapper to time any function
def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()

        print(('{n} took {t} sec'.format(n=method.__name__, t=round(te - ts, 2))))
        return result

    return timed


def parse_regmethod(regmethod):
    #: See comments in method_defs.py for definitions of each piece of regmethod

    reg_key = list(regmethod.keys())[0]  # eg. 'linear', 'logit', etc
    x_var = list(regmethod[reg_key]['feats'].keys())
    y_var = regmethod[reg_key]['targets']  # a list of response variable

    x_param = DataFrame().from_dict(regmethod[reg_key]['feats'])

    return reg_key, x_var, y_var, x_param


def round_numeric(x):
    if isinstance(x, str):
        return x
    else:
        return round(x, 2)
