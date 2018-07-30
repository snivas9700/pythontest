from numpy import isnan, float64
from pandas import merge, DataFrame, concat

from optimization import optprice_mtier
from variable_selection import main_regression

from utils import timeit


def compute_metrics(data, reg_vals, coef_lev, layer, idx_fold=True):
    # data must be the original one, NOT the train set for CV. It can have the column of 'fold' but NOT 'fold_cv'
    # coef_lev has columns such as [x_var]+['level'] with a suffix corresponding to its layer defined by the index names
    # coef_lev has index such as (1) [tier_1,...,tier_l] or (2) [tier_1,...,tier_l, fold]
    # layer_in is the max layer indexed in coef_lev, which is obtained from the index names of coef, no more than L
    # layer_out is the max layer in outdf, which is an input parameter
    # idx_fold=True if fold ID will be included in output, and False if it is excluded
    tier_list = [x for x in data.columns if 'tier_' in x]
    L = len(tier_list)  # find the total layers defined in data, including tier_0. If tier_list=[], return 0
    coef = coef_lev.ix[:, :-1]  # remove the last column 'level' from coef_lev
    coef_layer_list = [int(l.split('_')[1]) for l in coef_lev.index.names if 'tier_' in l]

    layer_in = len(coef_layer_list) - 1  # layer_in = 0 for initial coef_lev
    layer_out = max(min(layer, L - 1), 0)  # layer_out = max(min(layer, L-1), layer_in)

    reg_key = reg_vals[0]  # logit, linear, etc

    # data_coef: apply coef to data, and calculate columns to be used in func_dict
    # If all columns are available in data, then data_coef = data, otherwise we need to calculate them by coef
    #     'logit' needs OptPrice / OptDiscount as new columns
    #     'linear' needs Pred / APE as new columns
    # func_dict: calculate metrics from the columns of data_coef, such as 'EntitledDiscount','Win'
    #               and 'OptDiscount' in 'logit'
    # metrics_list: specify the matrics as a one-to-one mapping from func_dict
    if reg_key == 'logit':
        data_coef = optprice_mtier(data, coef)  # 'OptDiscount_'+str(layer_in)
        func_dict = dict()
        func_dict['discount_act'] = [['mean', 'std', 'max', 'min'],
                                     ['discount_avg', 'discount_std', 'discount_max', 'discount_min']]
        func_dict['discount_opt'] = [['mean'], ['discount_opt']]
        func_dict['winloss'] = [['count', 'sum'], ['count_all', 'count_win']]

    # elif reg_key == 'linear':
    #     data_coef = predict_mtier(data, coef=False)
    #     func_dict = dict()
    #     func_dict['Discount_act'] = [['mean', 'std', 'max', 'min'],
    #                                  ['Discount_avg', 'Discount_std', 'Discount_max', 'Discount_min']]
    #     func_dict['Discount_pred'] = [['mean'], ['Discount_pred']]
    #     func_dict['APE'] = [['count', 'mean'], ['Count_all', 'MAPE']]

    metrics_dict = dict((key + '_' + func_dict[key][0][i], func_dict[key][1][i]) for key in func_dict.keys() for i in
                        range(len(func_dict[key][0])))

    metrics_func = dict((key, func_dict[key][0]) for key in func_dict.keys())
    index_list = tier_list[:layer_out + 1] + ['fold'] if 'fold' in coef_lev.index.names and idx_fold else tier_list[:layer_out + 1]

    metrics = data_coef.groupby(index_list).agg(metrics_func)  # groupby tier 0,...,tier_layer, (fold)
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]  # flatten columns
    metrics.rename(columns=metrics_dict, inplace=True)

    suff = '_' + str(layer_in)
    metrics = metrics.add_suffix(suff)

    return metrics


@timeit
def compute_regout(data, reg_vals, coef_lev, layer):
    # data must be the original one, NOT the unfolded one. It can have the column of 'fold' but NOT 'fold_cv'
    # coef_lev has columns such as [x_var]+['level'] with a suffix corresponding the layer defined by the index names
    # coef_lev has index such as (1) [tier_0,...,tier_l] or (2) [tier_0,...,tier_l, fold]
    # layer_in is the max layer of index of coef_lev, which is obtained from the index names of coef, no more than L-1
    # layer_out is the max layer of index of outdf, which is an input parameter
    tier_list = [x for x in data.columns if 'tier_' in x]
    L = len(tier_list)  # total number of tiers defined in data, equal to 0, if tier_list=[].

    # coef_layer_list=[0] for an initial coef_lev
    coef_layer_list = [int(l.split('_')[1]) for l in coef_lev.index.names if 'tier_' in l]

    layer_in = len(coef_layer_list) - 1  # layer_in = 0 for initial coef_lev
    layer_out = max(min(layer, L - 1), layer_in)
    # calculate metrics indexed up to layer_out
    metrics_df = compute_metrics(data, reg_vals, coef_lev, layer_out, idx_fold=True)
    metrics_df.reset_index(inplace=True)  # place index into columns

    outdf = merge(metrics_df, coef_lev, left_on=coef_lev.index.names, right_index=True)  # merge coef_lev

    index_list = tier_list[:layer_out + 1] + ['fold'] if 'fold' in coef_lev.index.names else tier_list[:layer_out + 1]
    outdf = outdf.set_index(index_list)  # outdf sets up the same index as metrics_df

    return outdf


def tier_selection(outdf_0, outdf_1, reg_vals):

    reg_key, x_var, y_var, x_param = reg_vals[0], reg_vals[1], reg_vals[2], reg_vals[3]

    lbd = x_param.xs('lower_bound')
    ubd = x_param.xs('upper_bound')

    outdf_all = merge(outdf_0, outdf_1, left_index=True, right_index=True)  # allign the index of outdf_0 and outdf_1

    layer_0 = int(list(set([l.split('_')[1] for l in outdf_0.columns if 'level_' in l]))[0])
    layer_1 = int(list(set([l.split('_')[1] for l in outdf_1.columns if 'level_' in l]))[0])

    c0_cols = [y for y in outdf_0.columns if any([c in y for c in x_var])]
    c1_cols = [y for y in outdf_1.columns if any([c in y for c in x_var])]
    coef_0 = outdf_all[c0_cols]
    coef_1 = outdf_all[c1_cols]

    # check for bounds violations
    c0cols = ['_'.join(x.split('_')[:-1]) for x in coef_0.columns]
    check_coef_0 = (coef_0.fillna(0) >= lbd.to_frame().T[c0cols].values).all(1) & (coef_0.fillna(0) <= ubd.to_frame().T[c0cols].values).all(1)
    check_coef_0 = check_coef_0 & (1 - isnan(coef_0).all(1))  # False if NA for all coefficients

    c1cols = ['_'.join(x.split('_')[:-1]) for x in coef_1.columns]
    check_coef_1 = (coef_1.fillna(0) >= lbd.to_frame().T[c1cols].values).all(1) & (coef_1.fillna(0) <= ubd.to_frame().T[c1cols].values).all(1)
    check_coef_1 = check_coef_1 & (1 - isnan(coef_1).all(1))  # False if NA for all coefficients

    pick_coef_0 = check_coef_1 < check_coef_0  # select outdf_0 if check_coef_0 & ~check_coef_1
    pick_coef_1 = check_coef_1 > check_coef_0  # select outdf_1 if ~check_coef_0 & check_coef_1

    suff = '_' + str(layer_1)
    outdf_names = [x.split(suff)[0] for x in outdf_1.columns]
    outdf_all = concat([outdf_all, DataFrame(columns=outdf_names, dtype=float64)])

    # pick_seg is specified by Reg_key, which is series of bools to indicate if outdf_1 is selected (True)
    if reg_key == 'linear':
        pick_seg = outdf_all['MAPE_%d' % (layer_0)] > outdf_all['MAPE_%d' % (layer_1)]
    if reg_key == 'logit':
        beta = 0.25
        discount_gap_0 = outdf_all['discount_opt_%d' % layer_0].fillna(0) - outdf_all['discount_avg_%d' % layer_0].fillna(0)
        discount_gap_1 = outdf_all['discount_opt_%d' % layer_1].fillna(0) - outdf_all['discount_avg_%d' % layer_1].fillna(0)
        disc_std = 'discount_std_{}'.format(layer_0)
        pick_seg = (abs(discount_gap_1) <= beta * outdf_all[disc_std].fillna(0)) | (abs(discount_gap_1) <= abs(discount_gap_0))

    # stitch together picks from outdf_1 and outdf_0
    cols_0 = outdf_0.columns
    cols_1 = outdf_1.columns

    mask1 = pick_coef_1  # prioritize most recent coefficient values
    mask2 = pick_coef_0 & ~pick_coef_1  # fallback to previous values
    mask3 = ~pick_coef_1 & ~pick_coef_0 & pick_seg   # fallback to lower MAPE
    mask4 = ~pick_coef_1 & ~pick_coef_0 & ~pick_seg  # default to outdf_0 values

    # order of mask_list dictates order of mask filtering - preserved order from old code
    mask_list = [(mask1, cols_1), (mask2, cols_0), (mask3, cols_1), (mask4, cols_0)]

    # less compact but should be faster than row-by-row
    for (mask, cols) in mask_list:
        if any(mask):
            outdf_all.loc[mask, outdf_names] = outdf_all.loc[mask, cols].values

    outdf = outdf_all[outdf_names].add_suffix('_%d' % layer_1)
    return outdf


def regress_addtier(data, reg_vals, outdf, data_vol, min_score):
    # Run regression by adding one more tier
    # outdf (DataFrame obj) is obtained from the parent nodes, indexed from 0 to layer-1, for layer>=1
    # layer is the current tier ID after adding one more tier, layer = 0,...,L-1
    # Columns = metrics_list + x_var + ['level']
    # suffix '0' means the coefficients from parent tiers
    # suffix '1' means the coefficients at current tier
    # tier_list, and layer are obtained from the columns of data, named as [tier_0,...,tier_L]

    x_var = reg_vals[1]

    tier_list = [x for x in data.columns if 'tier_' in x]
    n_tiers = len(tier_list)  # find the total layers defined in data, if tier_list=[], return L=0

    # current and upper layer IDs are obtained from the index names of 'outdf'
    # new layer (tier ID) after adding one more tier on outdf
    layer = max([int(x.split('_')[-1]) for x in outdf.index.names if 'tier_' in x]) + 1

    # layer is tier ID, from 0 to L-1, since outdf has L tiers. If tier_L-1 has been reached,
    #   no more layer can be added.
    if layer >= n_tiers:
        print('Error: tier_{} exceeds the maximum tiers defined in data.'.format(layer))
    else:
        print('\n###################################################')
        print('tier_{} is the current tier of segmentation tree.'.format(layer))
        print('####################################################')
        # Re-use parent's coef_lev_0 to calculate the metrics after adding one more layer
        # Re-format: outdf.columns are in the format of [metrics_2]+[x_var_2]+[level_2], if layer=3

        # find columns in outdf that correspond to features (usually have a suffix)
        # NOTE: Preserve the "level_*" column(s)
        feat_cols = [x for x in outdf.columns if any([y in x for y in x_var + ['level']])]
        coef_prev = outdf[feat_cols]

        # coef & level indexed up to layer, suffix at layer-1
        outdf_prev = compute_regout(data, reg_vals, coef_prev, layer)

        # Run regression up to tier_layer
        print('Running main_regression on groupby object...')
        coef_curr = data.groupby(tier_list[:layer + 1]).apply(main_regression, reg_vals, data_vol, min_score)
        coef_curr['level'] = layer  # add 'level' as the last column
        coef_curr = coef_curr.add_suffix('_%d' % layer)  # add '_layer' as suffix = '_%d'

        outdf_curr = compute_regout(data, reg_vals, coef_curr, layer)
        outdf = tier_selection(outdf_prev, outdf_curr, reg_vals)
    return outdf


@timeit
def regress_mtier(data, reg_vals, data_vol, min_score, tier_0_reg=False):

    data_vol = max(0.8, min(1., data_vol))  # correct wrong data_vol (eg.100) and allow at most 20% deletion
    min_score = max(0, min(10, min_score))  # score is in the range of 0 to 10

    # Top-Down application of regression on all data by add one more tier in each step of iteration
    # tier_0_reg=False, if regression won't be applied at tier_0 for all data. Otherwise, True
    # find the number of tiers defined in data from the columns: tier_0,...,tier_L-1
    tier_list = [x for x in data.columns if 'tier_' in x]
    L = len(tier_list)  # find the total number of tiers (tier_0 always included), if tier_list=[], return 0
    if L == 0:
        print('Error: No tier is found in data.')

    x_var = reg_vals[1]
    # Initialization of coef_lev: index is the same as tier_0(+fold), columns include x_var and level, suffix by 0
    if not tier_0_reg:  # Initialize coef_lev as an empty DataFrame with a suffix 0
        coef = DataFrame({}, columns=x_var, index=['all'], dtype=float64)
        coef.index.names = ['tier_0']
    else:  # Initialize coef_lev after running regression for all data in tier_0
        coef = data.groupby(tier_list[:1]).apply(main_regression, reg_vals, data_vol, min_score)

    coef['level'] = 0  # initialize column of level
    coef_lev = coef.add_suffix('_0')  # add suffix

    layer = 0  # layer is the tier ID
    outdf = compute_regout(data, reg_vals, coef_lev, layer)
    outdf_list = [outdf]
    for layer in range(0, L - 1):  # from 0 to L-2
        outdf = regress_addtier(data, reg_vals, outdf_list[layer], data_vol, min_score)
        outdf_list.append(outdf)

    return outdf
