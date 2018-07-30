from time import time
from pandas import DataFrame, read_csv, merge, concat, Series
from datetime import datetime as dt
from numpy import log, exp
from os import listdir
from os.path import join as os_join


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


def build_quarter_map(data):
    # builds dataframe of quote_id, quarter_designation
    dat = data[['quoteid', 'quote_date']].drop_duplicates(subset='quote_date')

    # training data SUBMIT_DATE has format YYYYMMDD
    dat['date'] = dat['quote_date'].apply(lambda x: dt.strptime(str(x), '%m/%d/%y') if isinstance(x, str) else dt.strptime(str(x), '%Y%m%d'))

    dat['quarter'] = dat['date'].apply(lambda x: str((x.month-1)//3+1) + 'Q' + str(x.year)[2:] )

    return dat[['quoteid', 'quarter']]


# TODO - optimize this
def apply_bounds(data_optprice):
    print('Applying bounds...')

    mask_h = (data_optprice['price_opt'] > 1.2 * data_optprice['Value'])
    mask_l = (data_optprice['price_opt'] < 0.8 * data_optprice['Value'])

    print(('{}% hit upper bound by VS'.format(mask_h.sum() / len(data_optprice.index))))
    print(('{}% hit lower bound by VS'.format(mask_l.sum() / len(data_optprice.index))))

    data_optprice.loc[mask_h, 'price_opt'] = 1.2 * data_optprice.loc[mask_h, 'Value']
    data_optprice.loc[mask_l, 'price_opt'] = 0.8 * data_optprice.loc[mask_l, 'Value']

    mask_h = (data_optprice['price_opt'] > data_optprice['ENTITLED_SW'])
    mask_l = (data_optprice['price_opt'] < 0.15 * data_optprice['ENTITLED_SW'])

    print(('{}% hit upper bound by EP'.format(mask_h.sum() / len(data_optprice.index))))
    print(('{}% hit lower bound by EP'.format(mask_l.sum() / len(data_optprice.index))))

    data_optprice.loc[mask_h, 'price_opt'] = data_optprice.loc[mask_h, 'ENTITLED_SW']
    data_optprice.loc[mask_l, 'price_opt'] = 0.15 * data_optprice.loc[mask_l, 'ENTITLED_SW']

    data_optprice['Discount_opt'] = 1 - data_optprice['price_opt'] / data_optprice['ENTITLED_SW']

    print('Done')
    return data_optprice


# use train_vs (component-level value score calc dataframe) to determine quote type (SSW/SaaS/mixed)
# and apply that type designation to the optimal price output
def add_quote_type(op, vs):
    type_dict = vs.groupby('WEB_QUOTE_NUM').apply(
        lambda df: 'mixed' if len(df['SSW_Saas'].unique()) > 1 else df['SSW_Saas'].unique()[0]).to_dict()

    op['type'] = op['WEB_QUOTE_NUM'].map(type_dict)
    return op


def post_process_op(optprice, vs, quarters):
    optprice['unbounded_price_opt'] = optprice['price_opt']

    optprice['deviance'] = -log(1 - optprice['wp_act'])
    optprice.loc[optprice['WIN'], 'deviance'] = -log(optprice.loc[optprice['WIN'], 'wp_act'])
    optprice['mean_deviance'] = 1 - exp(-optprice['deviance'].mean())

    optprice = merge(optprice, quarters, on='WEB_QUOTE_NUM', how='left')

    op = apply_bounds(optprice.copy())
    op = add_quote_type(op=op, vs=vs)  # to help the test suite

    return op


def load_data(src_path, label='training'):
    csvs = [x for x in listdir(src_path) if label in x]

    data = concat([read_csv(os_join(src_path, csv)) for csv in csvs])
    data = data.drop('Unnamed: 0', axis=1) if 'Unnamed: 0' in data.columns else data
    data.columns = [x.lower() for x in data.columns]

    return data


def round_numeric(x):
    if isinstance(x, str):
        return x
    else:
        return round(x, 2)
