from pandas import merge, DataFrame, ExcelWriter, read_csv
from numpy import exp, log, round, arange, around
from collections import OrderedDict
from copy import copy
from os.path import join as os_join
import json
import os
import re
import hashlib
from datetime import datetime as dt


def apply_bounds(data_optprice):
    print('Applying bounds...')

    mask_h = (data_optprice['price_opt'] > 1.2 * data_optprice['value_score'])
    mask_l = (data_optprice['price_opt'] < 0.8 * data_optprice['value_score'])

    print(('{}% hit upper bound by VS'.format(mask_h.sum() / len(data_optprice.index))))
    print(('{}% hit lower bound by VS'.format(mask_l.sum() / len(data_optprice.index))))

    data_optprice.loc[mask_h, 'price_opt'] = 1.2 * data_optprice.loc[mask_h, 'value_score']
    data_optprice.loc[mask_l, 'price_opt'] = 0.8 * data_optprice.loc[mask_l, 'value_score']
    data_optprice.loc[:, 'vs_bnd'] = mask_l | mask_h

    mask_h = (data_optprice['price_opt'] > data_optprice['list_price'])
    mask_l = (data_optprice['price_opt'] < 0.15 * data_optprice['list_price'])

    print(('{}% hit upper bound by EP'.format(mask_h.sum() / len(data_optprice.index))))
    print(('{}% hit lower bound by EP'.format(mask_l.sum() / len(data_optprice.index))))

    data_optprice.loc[mask_h, 'price_opt'] = data_optprice.loc[mask_h, 'list_price']
    data_optprice.loc[mask_l, 'price_opt'] = 0.15 * data_optprice.loc[mask_l, 'list_price']
    data_optprice.loc[:, 'ep_bnd'] = mask_l | mask_h

    data_optprice['discount_opt'] = 1 - data_optprice['price_opt'] / data_optprice['list_price']

    print('Done')
    return data_optprice


def post_process_op(optprice, quarters):
    optprice['unbounded_price_opt'] = optprice['price_opt']

    optprice['deviance'] = -log(1 - optprice['wp_act'])
    optprice.loc[optprice['winloss'], 'deviance'] = -log(optprice.loc[optprice['winloss'], 'wp_act'])
    optprice['mean_deviance'] = 1 - exp(-optprice['deviance'].mean())

    optprice = merge(optprice, quarters, on='quoteid', how='left')

    op = apply_bounds(optprice.copy())

    return op


def refresh_analysis(df):
    df['vs_discount'] = 1 - df['value_score'] / df['list_price']
    df_ = slice_data(df.copy(), ent_col='list_price')

    n = len(df_)
    mape = round(df_.APE.mean() * 100, 1)

    # to ensure column order
    base_cols = ['leading_brand', 'Deal_Size_Min', 'Deal_Size_Max']
    rename_dict = OrderedDict([('winloss_mean', 'Win rate'),
                              ('winloss_count', 'Number of Observations'),
                              ('discount_act_mean', 'Actual Discount'),
                              ('vs_discount_mean', 'Value Score Discount'),
                              ('discount_opt_mean', 'Optimal Discount'),
                              ('APE_mean', 'APE')])

    col_order = base_cols + list(rename_dict.values()) + ['Revenue % Delta', 'switch']
    agg = DataFrame()
    for switch in [True, False]:
        b_cols = copy(base_cols)
        if not switch:
            [b_cols.remove(x) for x in ['Deal_Size_Min', 'Deal_Size_Max']]

        tmp = group_agg(df_, b_cols, rename_dict)
        tmp['switch'] = switch
        agg = agg.append(tmp)

    # enforce order
    agg = agg[col_order]

    return agg, n, mape, col_order


def group_agg(df, base_cols, rename_dict):
    col_list = base_cols + ['winloss', 'discount_act', 'vs_discount', 'discount_opt', 'APE', 'price_opt', 'price_act']

    agg_dict = {'winloss': ['mean', 'count'],
                'discount_act': 'mean',
                'vs_discount': 'mean',
                'discount_opt': 'mean',
                'APE': 'mean',
                'price_opt': 'sum',
                'price_act': 'sum'}

    group_table = df[col_list].groupby(base_cols).agg(agg_dict).reset_index()

    group_table.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in group_table.columns.values]
    group_table = group_table.rename(columns=rename_dict)

    group_table['Revenue % Delta'] = group_table['price_opt_sum'] / group_table['price_act_sum']
    group_table.drop(['price_opt_sum', 'price_act_sum'], axis=1, inplace=True)

    group_table['APE'] = round(group_table['APE'].ix[0] * 100, 1)

    return group_table


def slice_data(df, ent_col='list_price'):
    cond_1 = df[ent_col] <= 50000
    cond_2 = (df[ent_col] > 50000) & (df[ent_col] <= 500000)
    cond_3 = (df[ent_col] > 500000) & (df[ent_col] <= 5000000)
    cond_4 = df[ent_col] > 5000000

    df['Deal_Size_Min'] = 0
    df['Deal_Size_Max'] = 0

    df.loc[cond_1, 'Deal_Size_Min'] = 0
    df.loc[cond_1, 'Deal_Size_Max'] = 50000

    df.loc[cond_2, 'Deal_Size_Min'] = 50001
    df.loc[cond_2, 'Deal_Size_Max'] = 500000

    df.loc[cond_3, 'Deal_Size_Min'] = 500001
    df.loc[cond_3, 'Deal_Size_Max'] = 5000000

    df.loc[cond_4, 'Deal_Size_Min'] = 5000001
    df.loc[cond_4, 'Deal_Size_Max'] = 'infinity'

    return df


def parse_agg_df(agg_df, meta, folder='output'):
    fname = os_join(folder, 'report.xls')
    print(('Building report in file "{}"...'.format(fname)))
    xl = ExcelWriter(fname)

    sheet_name = 'n_{n}|mape_{m}'.format(n=meta['n'], m=meta['mape'])
    print(('\twriting sheet "{}"'.format(sheet_name)))
    dat = agg_df.copy()

    df = DataFrame()
    for lab in dat['switch'].unique():
        df = df.append(dat.loc[dat['switch'].eq(lab)].drop('switch', axis=1))  # split up data
        # insert empty rows for readability
        df = df.append(DataFrame(data={df.columns[0]: [None, None]}, index=[0, 1]), ignore_index=True)

    df[meta['col_order']].to_excel(xl, sheet_name=sheet_name)

    xl.save()


def run_model_stats(vs, op, agg_by='tier_3', folder='output'):
    fname = os_join(folder, 'model_stats.csv')
    print('Calculating model summary stats...')
    stats = {}
    stats.update({'tot': calc_model_stats(vs, op)})

    for grp, op_grp in op.groupby(agg_by):
        qids = op_grp['quoteid'].unique()
        vs_grp = vs.loc[vs['quoteid'].isin(qids)].reset_index(drop=True)
        stats.update({str(grp): calc_model_stats(vs_grp, op_grp)})

    # hard coded for now
    col_order = ['count', 'lme_mape', 'op_mape', 'deviance', 'vs_bound', 'ep_bound', 'roc', 'quadrant_1',
                 'quadrant_2', 'quadrant_3', 'quadrant_4']
    df = DataFrame().from_dict(stats, orient='index').applymap(lambda x: round(x, 4)).sort_index()
    df.index.name = 'tier_3'
    print('Writing model summary to csv...')
    df.to_csv(fname, columns=col_order, index=True)
    print('Done')


def calc_model_stats(vs, op):

    lme_mape = (abs(vs['value_score'] - vs['quoted_price'])/vs['quoted_price']).mean()
    op_mape = (abs(op['price_opt'] - op['quoted_price'])/op['quoted_price']).mean()
    dev = op['deviance'].mean()

    # vs_bnd = op['vs_bnd'].sum() / float(op.shape[0])
    # ep_bnd = op['ep_bnd'].sum() / float(op.shape[0])
    roc_metric = calc_roc(op)

    quad_dat = run_quads(op.copy())
    quad_cnt = quad_dat['quadrant'].value_counts()
    quad_dict = {quad_cnt.name + '_' + str(idx): val/float(op.shape[0]) for idx, val in quad_cnt.items()}

    stats = {'lme_mape': lme_mape,
             'op_mape': op_mape,
             'deviance': dev,
             # 'vs_bound': vs_bnd,
             # 'ep_bound': ep_bnd,
             'roc': roc_metric,
             'count': op.shape[0]
             }
    stats.update(quad_dict)

    return stats


def run_quads(dat):
    dat['discount_opt'] = 1 - dat['unbounded_price_opt'] / dat['list_price']
    dat['OP'] = dat['unbounded_price_opt'].copy()

    dat['quadrant'] = 0
    dat.loc[dat['winloss'].eq(1) & (dat['quoted_price'] < dat['OP']), 'quadrant'] = 1
    dat.loc[dat['winloss'].eq(1) & (dat['quoted_price'] >= dat['OP']), 'quadrant'] = 2
    dat.loc[dat['winloss'].eq(0) & (dat['quoted_price'] <= dat['OP']), 'quadrant'] = 3
    dat.loc[dat['winloss'].eq(0) & (dat['quoted_price'] > dat['OP']), 'quadrant'] = 4

    dat['quadrant_wp_op'] = 0
    dat.loc[dat['quadrant'].eq(1), 'quadrant_wp_op'] = dat['wp_opt']/dat['wp_act']
    dat.loc[dat['quadrant'].eq(2), 'quadrant_wp_op'] = 1.
    dat.loc[dat['quadrant'].eq(3), 'quadrant_wp_op'] = 0.
    dat.loc[dat['quadrant'].eq(4), 'quadrant_wp_op'] = (1. - (1. - dat['wp_opt']) / (1. - dat['wp_act']))

    mask = dat['wp_opt'].eq(0) & dat['quadrant'].eq(1)
    dat.loc[mask, 'quadrant_wp_op'] = 1.

    dat['quadrant_price'] = dat['quadrant_wp_op'] * dat['OP']
    dat['quadrant_discount'] = 1 - dat['quadrant_price'] / dat['list_price']
    dat['md1'] = 'opt'
    mask = (dat['quoted_price'] - dat['OP']) / dat['OP'] > 0.1
    dat.loc[mask, 'md1'] = 'pos'
    mask = (dat['quoted_price'] - dat['OP']) / dat['OP'] < -0.1
    dat.loc[mask, 'md1'] = 'neg'

    return dat


def calc_roc(data):

    roc_fpr = []
    roc_tpr = []
    acc = []
    dec_boundary = []

    for shift in arange(-.5, 0.51, 0.01):
        dec_boundary.append(0.5 + shift)
        roc_fpr.append(FPR(data['winloss'], data['wp_act'], shift))
        roc_tpr.append(TPR(data['winloss'], data['wp_act'], shift))
        acc.append(accuracy(data['winloss'], data['wp_act'], shift))

    roc = DataFrame(data={'decision_boundary': dec_boundary,
                             'FPR': roc_fpr,
                             'TPR': roc_tpr,
                             'acc': acc
                             })

    roc_metric = -1 * roc['FPR'].diff().fillna(roc['FPR'].diff().mean()).dot(roc['TPR'])
    return roc_metric


def FPR(y_true, y_pred, shift=0):
    '''Function to calculate the False Positive Rate ('Fallout', 'Probability of False Alarm') of a binary prediction.
       y_true is the true (binary) labels
       y_pred is the predicted label. y_pred can be binary or in [0,1].
       In the latter case shift is used to binarize the values with decision boundary = 0.5-shift
    '''
    y_bin = around(y_pred.values - shift)
    fp = (y_bin.T.dot(1 - y_true)).sum()
    tn = ((1 - y_bin).T.dot((1 - y_true))).sum()
    return fp / (fp + tn)


def TPR(y_true, y_pred, shift=0):
    '''Function to calculate the True Positive Rate ('Recall', 'Probability of detection') of a binary prediction.
       y_true is the true (binary) labels
       y_pred is the predicted label. y_pred can be binary or in [0,1].
       In the latter case shift is used to binarize the values with decision boundary = 0.5-shift
    '''
    y_bin = around(y_pred.values - shift)
    tp = (y_bin.T.dot(y_true)).sum()
    fn = ((1 - y_bin).T.dot(y_true)).sum()
    return tp / (tp + fn)


def accuracy(y_true, y_pred, shift=0):
    '''Function to calculate the False Positive Rate ('Fallout', 'Probability of False Alarm') of a binary prediction.
       y_true is the true (binary) labels
       y_pred is the predicted label. y_pred can be binary or in [0,1].
       In the latter case shift is used to binarize the values with decision boundary = 0.5-shift
    '''
    y_bin = around(y_pred.values - shift)

    return (y_bin == y_true).mean()


def update_tracking_file(config, folder_hash, base_folder, d):

    d.update({'LME formula': {'re': config['model']['re'], 'fe': config['model']['fe']}})
    d.update({'WR features': config['wr_feats']})
    d.update({'misc': config['misc']})

    fname = os_join(base_folder, 'summary.txt')
    try:
        print(('Updating tracking file at {}...'.format(fname)))
        _ = open(fname, 'rw')
    except IOError:
        print(('File does not exist yet. Will create it'.format(fname)))
        open(fname, 'w+')

    summary = '--- folder: {h} ({d}) --- \n\n{r}\n\n\n'.format(
        h=folder_hash, d=dt.strftime(dt.now(), '%Y-%m-%d %H:%M:%S'), r=json.dumps(d, indent=1))

    # clear out newlines from text within square brackets
    summary = re.sub(r'\[.*?\]', lambda m: m.group().replace('\n', '').replace(' ', ''),
                     summary, flags=re.DOTALL).replace('"', "'")

    with open(fname, 'a') as f:
        f.write(summary)

    print('Done')


def post_hoc_wrapper(train_vs, train_op, config, reg_folder, d, base_folder):
    # store = DataFrame()
    # meta = {}
    #
    # op_dat = train_op.copy()

    # agg_df, n, mape, cols = refresh_analysis(op_dat)
    # store = store.append(agg_df)
    # meta.update({'n': n, 'mape': mape})

    # cols.remove('switch')  # switch column isn't wanted in final report
    # meta.update({'col_order': cols})  # cols won't change across types (SSW/Saas/mixed)

    config_str = json.dumps(config)
    folder_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:10]  # unique for each distinct configuration
    hash_folder = os_join(base_folder, folder_hash)
    out_folder = os_join(hash_folder, reg_folder)

    try:
        os.listdir(out_folder)
    except OSError:
        try:
            os.listdir(hash_folder)
        except OSError:
            print(('Folder for hash {} does not exist. Creating'.format(folder_hash)))
            os.mkdir(hash_folder)
        print(('Folder for hash {h} output region {r} does not exist. Creating'.format(h=folder_hash, r=reg_folder)))
        os.mkdir(out_folder)

    # parse_agg_df(store, meta, folder=out_folder)
    #
    # run_model_stats(train_vs, train_op, agg_by='countrycode', folder=out_folder)
    #
    # # build summary readme here
    # if reg_folder == 'EMEA':
    #     update_tracking_file(config, folder_hash, base_folder, d)

    return hash_folder


def build_summary(base_folder):
    tots = DataFrame()
    hash_folders = [x for x in os.listdir(base_folder) if '.' not in x]
    for hash in hash_folders:
        regions = [x for x in os.listdir(os_join(base_folder, hash)) if '.' not in x]
        for region in regions:
            dat = read_csv(os_join(base_folder, hash, region, 'model_stats.csv'))
            dat['hash'] = hash
            dat['region'] = region
            entry = dat.set_index(['hash', 'region', 'tier_3']).xs('tot', level='tier_3')
            tots = tots.append(entry)

    print('Writing summary stats to csv...')
    tots.to_csv(os_join(base_folder, 'tot_stats.csv'))
