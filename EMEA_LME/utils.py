from time import time
from pandas import DataFrame, read_csv, merge, concat, Series, read_excel
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

        print('{n} took {t} sec'.format(n=method.__name__, t=round(te - ts, 2)))
        return result

    return timed


def parse_regmethod(regmethod):
    #: See comments in method_defs.py for definitions of each piece of regmethod

    reg_key = regmethod.keys()[0]  # eg. 'linear', 'logit', etc
    x_var = regmethod[reg_key]['feats'].keys()
    y_var = regmethod[reg_key]['targets']  # a list of response variable

    x_param = DataFrame().from_dict(regmethod[reg_key]['feats'])

    return reg_key, x_var, y_var, x_param


def build_quarter_map(data):
    # builds dataframe of quote_id, quarter_designation
    dat = data[['quoteid', 'quote_date']].drop_duplicates(subset='quote_date')

    # training data SUBMIT_DATE has format YYYYMMDD
    dat['date'] = dat['quote_date'].apply(lambda x: dt.strptime(str(x), '%m/%d/%y') if isinstance(x, basestring) else dt.strptime(str(x), '%Y%m%d'))

    dat['quarter'] = dat['date'].apply(lambda x: str((x.month-1)//3+1) + 'Q' + str(x.year)[2:] )

    return dat[['quoteid', 'quarter']]


# TODO - optimize this

def apply_bounds_hw(data_optprice):
    print('Applying bounds for hw...')

    mask_h1 = (data_optprice['op_GP_pct_list_hw'] > 6.0 * data_optprice['y_pred'])
    mask_l1 = (data_optprice['op_GP_pct_list_hw'] < 0.5 * data_optprice['y_pred'])

    mask_h2 = (data_optprice['op_REV_pct_list_hw'] > 6.0 * data_optprice['y_pred'])
    mask_l2 = (data_optprice['op_REV_pct_list_hw'] < 0.5 * data_optprice['y_pred'])

    print('GP Optimal Price:')
    print('{}% hit upper bound by VS'.format(mask_h1.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by VS'.format(mask_l1.sum() / len(data_optprice.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by VS'.format(mask_h2.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by VS'.format(mask_l2.sum() / len(data_optprice.index)))

    data_optprice.loc[mask_h1, 'op_GP_pct_list_hw'] = 6.0 * data_optprice.loc[mask_h1, 'y_pred']
    data_optprice.loc[mask_l1, 'op_GP_pct_list_hw'] = 0.5 * data_optprice.loc[mask_l1, 'y_pred']

    data_optprice.loc[mask_h2, 'op_REV_pct_list_hw'] = 6.0 * data_optprice.loc[mask_h2, 'y_pred']
    data_optprice.loc[mask_l2, 'op_REV_pct_list_hw'] = 0.5 * data_optprice.loc[mask_l2, 'y_pred']

    mask_h1 = (data_optprice['op_GP_pct_list_hw'] > 1.0)
    mask_l1 = (data_optprice['op_GP_pct_list_hw'] < 0.2)
    mask_h2 = (data_optprice['op_REV_pct_list_hw'] > 1.0)
    mask_l2 = (data_optprice['op_REV_pct_list_hw'] < 0.2)

    print('GP Optimal Price:')
    print('{}% hit upper bound by LP'.format(mask_h1.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by LP'.format(mask_l1.sum() / len(data_optprice.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by LP'.format(mask_h2.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by LP'.format(mask_l2.sum() / len(data_optprice.index)))

    data_optprice.loc[mask_h1, 'op_GP_pct_list_hw'] = 1.0
    data_optprice.loc[mask_l1, 'op_GP_pct_list_hw'] = 0.2
    data_optprice.loc[mask_h2, 'op_REV_pct_list_hw'] = 1.0
    data_optprice.loc[mask_l2, 'op_REV_pct_list_hw'] = 0.2

    data_optprice['discount_opt_GP_hw'] = 1 - data_optprice['op_GP_pct_list_hw']
    data_optprice['discount_opt_REV_hw'] = 1 - data_optprice['op_REV_pct_list_hw']

    data_optprice['optimal_price_GP_hw'] = data_optprice['op_GP_pct_list_hw'] * data_optprice['p_list_hw']
    data_optprice['optimal_price_REV_hw'] = data_optprice['op_REV_pct_list_hw'] * data_optprice['p_list_hw']

    print('Done with Applying Bounds.')
    return data_optprice


def apply_bounds_tss(data_optprice):
    print('Applying bounds...')

    # First set of boundaries
    mask_h1 = (data_optprice['op_GP_pct_list_hwma'] > 6.0 * data_optprice['y_pred'])
    mask_l1 = (data_optprice['op_GP_pct_list_hwma'] < 0.5 * data_optprice['y_pred'])

    mask_h2 = (data_optprice['op_PTI_pct_list_hwma'] > 6.0 * data_optprice['y_pred'])
    mask_l2 = (data_optprice['op_PTI_pct_list_hwma'] < 0.5 * data_optprice['y_pred'])

    mask_h3 = (data_optprice['op_REV_pct_list_hwma'] > 6.0 * data_optprice['y_pred'])
    mask_l3 = (data_optprice['op_REV_pct_list_hwma'] < 0.5 * data_optprice['y_pred'])

    print('GP Optimal Price:')
    print('{}% hit upper bound by VS'.format(mask_h1.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by VS'.format(mask_l1.sum() / len(data_optprice.index)))

    print('PTI Optimal Price:')
    print('{}% hit upper bound by VS'.format(mask_h2.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by VS'.format(mask_l2.sum() / len(data_optprice.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by VS'.format(mask_h3.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by VS'.format(mask_l3.sum() / len(data_optprice.index)))

    data_optprice.loc[mask_h1, 'op_GP_pct_list_hwma'] = 6.0 * data_optprice.loc[mask_h1, 'y_pred']
    data_optprice.loc[mask_l1, 'op_GP_pct_list_hwma'] = 0.5 * data_optprice.loc[mask_l1, 'y_pred']

    data_optprice.loc[mask_h2, 'op_PTI_pct_list_hwma'] = 6.0 * data_optprice.loc[mask_h2, 'y_pred']
    data_optprice.loc[mask_l2, 'op_PTI_pct_list_hwma'] = 0.5 * data_optprice.loc[mask_l2, 'y_pred']

    data_optprice.loc[mask_h3, 'op_REV_pct_list_hwma'] = 6.0 * data_optprice.loc[mask_h3, 'y_pred']
    data_optprice.loc[mask_l3, 'op_REV_pct_list_hwma'] = 0.5 * data_optprice.loc[mask_l3, 'y_pred']

    # Second set of boundaries
    mask_h1 = (data_optprice['op_GP_pct_list_hwma'] > 1.0)
    mask_l1 = (data_optprice['op_GP_pct_list_hwma'] < 0.2)
    mask_h2 = (data_optprice['op_PTI_pct_list_hwma'] > 1.0)
    mask_l2 = (data_optprice['op_PTI_pct_list_hwma'] < 0.2)
    mask_h3 = (data_optprice['op_REV_pct_list_hwma'] > 1.0)
    mask_l3 = (data_optprice['op_REV_pct_list_hwma'] < 0.2)

    print('GP Optimal Price:')
    print('{}% hit upper bound by LP'.format(mask_h1.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by LP'.format(mask_l1.sum() / len(data_optprice.index)))

    print('PTI Optimal Price:')
    print('{}% hit upper bound by LP'.format(mask_h2.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by LP'.format(mask_l2.sum() / len(data_optprice.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by LP'.format(mask_h3.sum() / len(data_optprice.index)))
    print('{}% hit lower bound by LP'.format(mask_l3.sum() / len(data_optprice.index)))

    data_optprice.loc[mask_h1, 'op_GP_pct_list_hwma'] = 1.0
    data_optprice.loc[mask_l1, 'op_GP_pct_list_hwma'] = 0.2
    data_optprice.loc[mask_h2, 'op_PTI_pct_list_hwma'] = 1.0
    data_optprice.loc[mask_l2, 'op_PTI_pct_list_hwma'] = 0.2
    data_optprice.loc[mask_h3, 'op_REV_pct_list_hwma'] = 1.0
    data_optprice.loc[mask_l3, 'op_REV_pct_list_hwma'] = 0.2

    data_optprice['discount_opt_GP_tss'] = 1 - data_optprice['op_GP_pct_list_hwma']
    data_optprice['discount_opt_PTI_tss'] = 1 - data_optprice['op_PTI_pct_list_hwma']
    data_optprice['discount_opt_REV_tss'] = 1 - data_optprice['op_REV_pct_list_hwma']

    data_optprice['optimal_price_GP_tss'] = data_optprice['op_GP_pct_list_hwma'] * data_optprice['p_list_hwma']
    data_optprice['optimal_price_PTI_tss'] = data_optprice['op_PTI_pct_list_hwma'] * data_optprice['p_list_hwma']
    data_optprice['optimal_price_REV_tss'] = data_optprice['op_REV_pct_list_hwma'] * data_optprice['p_list_hwma']

    print('Done with Applying Bounds.')
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


def load_data(src_path, label='training', usecols_=[]):
    """
    Load (multiple) csv or Excel files, removes legacy index columns,
    concatenate into one DataFrame, and reformat field names to lowercase

    Args:
        usecols_: To load data more quickly, allows user to specify a
        subset of columns to load
        src_path (str): data store location
        label (str): portion of file name, to restrict files read in one
        folder, instead of reading all files in a folder

    Returns:
        DataFrame
    """
    csvs = [x for x in listdir(src_path) if label in x and 'csv' in x]
    excels = [x for x in listdir(src_path) if label in x and 'xls' in x]

    if len(csvs) > 0:
        # If no columns are specified, assign usecols_ all fields;
        # nrows=1 ensures rapid determination of field names without loading
        # the  whole data set; only works for csv's;
        # read_excel requires column indices (A, A:C or 1, 1:3, etc.), as it
        # cannot infer field names from a header row
        if len(usecols_) == 0:
            usecols_ = read_csv(os_join(src_path, csvs[0]), nrows=1,
                                encoding='latin-1').columns
        data = concat([read_csv(os_join(src_path, csv), encoding='latin-1',
                                usecols=usecols_, low_memory=False)
                       for csv in csvs])
    elif len(excels) > 0:
        data = concat([read_excel(os_join(src_path, excel))
                       for excel in excels])
    else:
        print('--------------------------------------------------')
        print('Data not found; check that the file is in src_path')
        print('--------------------------------------------------')

    # Remove spurious columns
    # In which lazy people ; ) do not suppress the export of the default
    # DataFrame index when writing csv's
    data = data.drop('Unnamed: 0', axis=1) if 'Unnamed: 0' in data.columns else data

    # In which a column consists entirely of NaNs
    for field in data.columns:
        if data[field].isnull().sum() == len(data):
            data.drop(field, axis=1, inplace=True)

    data.columns = [x.lower() for x in data.columns]

    return data


def round_numeric(x):
    if isinstance(x, basestring):
        return x
    else:
        return round(x, 2)

