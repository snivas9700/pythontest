from __future__ import division
from numpy import absolute, mean, log10, arange, int32, inf, tile, argmax, array, ones, linspace, exp, asarray, floor
from pandas import Series, DataFrame, concat, MultiIndex, merge
from scipy.optimize import least_squares
from time import time
from copy import deepcopy

from EMEA_LME.modeling_hw import quant_reg_pred, quant_reg_train


def run_quants_offline(data, idx_cols, in_feats, out_feat, grp_cols, percentile, labels, label_str):
    dat = data.copy()

    # store trained model objects
    mod_out = DataFrame()

    skipped = []
    if len(grp_cols) > 0:
        # raw quantile calculations
        qs = concat([dat.groupby(grp_cols)[out_feat].apply(lambda s: s.quantile(q=q)).to_frame(str(q))
                        for q in percentile], axis=1)
        qs = qs.rename(columns=labels)

        q_reg = DataFrame()
        for t3, grp in data.groupby(grp_cols):
            # train = grp.loc[grp['winloss'].eq(1)].reset_index(drop=True)
            train = grp.reset_index(drop=True)
            if train.shape[0] > 0:
                # establish filter to find testing data relevant to this groupby index
                # allow for arbitrary number of groupby columns
                mask = [True] * data.shape[0]
                masks = [data[c].eq(t3[i]) if len(grp_cols) > 1 else data[c].eq(t3) for i, c in enumerate(grp_cols)]
                for m in masks:
                    mask = mask & m

                # extract testing data + set all values to win = 1; i.e. predict price that would WIN the quote
                test = data.loc[mask].copy().reset_index(drop=True)
                test.loc[:, 'win_ind'] = 1
                q_mods = quant_reg_train(train.set_index(idx_cols), percentile, in_feats, out_feat, label_str)
                q_res = quant_reg_pred(test.set_index(idx_cols), in_feats, q_mods, qs.columns)

                q_reg = q_reg.append(q_res)

                q_mods.index = MultiIndex.from_tuples(list(zip(tile(t3, len(q_mods.index)), q_mods.index)))
                q_mods.index.names = grp_cols + ['quantile']
                mod_out = mod_out.append(q_mods)  # store multi-indexed quantile model parameters
            else:
                print(('Not enough samples in {g} level {v} ({n} samples)'.format(g=grp_cols, v=t3, n=train.shape[0])))
                # remove observed quantiles for skipped entries
                qs = qs.drop(t3)
                skipped.append(t3)

    # calculate population-level to fill in for segment_id values that were skipped
    qs_ = Series(data={str(q): dat[out_feat].quantile(q=q) for q in percentile})
    qs_ = qs_.rename(index=labels)
    qs_.name = 'ALL'

    qs = qs.append(qs_)

    train = dat
    test = dat.copy()
    test.loc[:, 'win_ind'] = 1
    q_mods = quant_reg_train(train.set_index(idx_cols), percentile, in_feats, out_feat, label_str)

    q_mods.index = MultiIndex.from_tuples(list(zip(tile('ALL', len(q_mods.index)), q_mods.index)))
    q_mods.index.names = grp_cols + ['quantile']
    mod_out = mod_out.append(q_mods)

    if len(skipped) > 0:
        # ID the entries that were skipped above
        mask = [True] * test.shape[0]
        masks = [test[c].isin(skipped[i]) if len(grp_cols) > 1 else test[c].isin(skipped) for i, c in enumerate(grp_cols)]
        for m in masks:
            mask = mask & m
        # run predictions on subset of data that was skipped
        test_ = test.loc[mask, :].set_index(idx_cols)
        # NOTE: Because q_mods is multiindexed, and quant_reg_pred() can't handle that, have to
        #       extract out the relevant q_mods entry before passing in
        q_res = quant_reg_pred(test_, in_feats, q_mods.xs('ALL', level=grp_cols[0]), qs_.index)
        # update outputs predictions
        q_reg = q_reg.append(q_res)

    pred_cols = [c for c in q_reg.columns if 'pred_' in c]  # get prediction labels

    return q_reg, qs, pred_cols, mod_out


def quantile_train(data, config, uplift=1.0):
    ind = config['ind'][0]
    if ind == 'hw':
        target = 'p_pct_list_hw'
    else:
        target = 'p_pct_list_hwma'

    data['metric'] = (data[target] / data['y_pred']).replace([-inf, inf], 0.0)  # inf -> 0.0

    data=data.reset_index(drop=True)
    alpha = 0.0
    idx_cols = config['index']
    grp_cols = config['grp_cols']
    percentile = config['percentile']
    labels = config['labels']
    label_str = list(config['labels'].values())
    # run quantile approaches; quantile regression + raw calculated quantiles
    quants, qs, pred_cols, models = run_quants_offline(
        data.copy(), idx_cols, config['quant_feats']['in_feats'], out_feat='metric', grp_cols=grp_cols,
        percentile=percentile, labels=labels, label_str=label_str)

    return quants, models, qs

#
# def run_mf_calcs(train_op, pred_cols, adj_cols, alpha):
#
#     train_op['norm_op'] = train_op.apply(
#         lambda row: calc_quant_op(*row[adj_cols].values, cf=0., cv=0., fcw=0., fcl=0.), axis=1)
#
#     train_op['price_opt'] = train_op['norm_op'] * train_op['value_score']  # calc_quant_op predicts QP/VS
#
#     # train_op = bound_op(train_op, alpha)
#
#     train_op['wp_opt'] = train_op.apply(
#         lambda row: calc_win_prob(row['norm_op'], *row[adj_cols].values), axis=1)
#
#     train_op['wp_act'] = train_op.apply(
#         lambda row: calc_win_prob(row['metric'], *row[adj_cols].values), axis=1)
#
#     # train_op['price_act'] = train_op['quoted_price'].copy()
#
#     # train_op[['ci_low', 'ci_high']] = train_op.apply(lambda row: Series(
#     #     opt_price_ci(row['norm_op'], *row[adj_cols].values, Cf=0., Cv=0, FCw=0, FCl=0, Pct=.95)),
#     #     axis=1)
#
#     op_cols = []
#     for col in pred_cols:
#         c = col.split('pred_')[-1]
#         c = 'price_' + c
#         train_op[c] = train_op[col] * train_op['value_score']
#         op_cols.append(c)  # for easy access later
#
#     return train_op


def pti(dat, df_cf):
    df_cf.rename(
        columns={'EUROPE & MEA IOTs\nCOST FACTORS\n(not for use in the\n"Cross-Brand Pricing Tool")': 'CostFactor'},
        inplace=True)
    df_cf = df_cf.loc[:, ['Type', 'Model', 'TypeModel', 'Brand', 'CostFactor']]
    # Clean up data
    df_cf.Type = df_cf.Type.astype('str').str.strip()
    df_cf.Model = df_cf.Model.astype('str').str.strip()
    df_cf.TypeModel = df_cf.TypeModel.astype('str').str.strip()
    df_cf.Brand = df_cf.Brand.astype('str').str.strip()

    df_cf.Type = df_cf.Type.str.lower()
    df_cf.Model = df_cf.Model.str.lower()
    df_cf.TypeModel = df_cf.TypeModel.str.lower()

    dat = dat.reset_index()

    dat["machine_type"] = dat["mtm"].str.slice(start=0, stop=4).str.strip()
    dat["machine_model"] = dat["mtm"].str.slice(start=4).str.strip()
    dat["machine_type"] = dat.machine_type.str.lower()
    dat["machine_model"] = dat.machine_model.str.lower()

    dat_mtm = dat[['index', 'machine_type', 'machine_model']].copy()
    df_specific = merge(dat_mtm, df_cf, left_on=['machine_model', 'machine_type'], right_on=['Model', 'Type'],
                        how='left')
    df_cf_broad = df_cf[df_cf['Model'] == "all"]
    df_broad = merge(df_specific, df_cf_broad, left_on=['machine_type'], right_on=['Type'], how='left')

    mask = (df_broad['CostFactor_x'].notnull())
    df_broad.loc[mask, 'costfactor'] = df_broad.loc[mask, 'CostFactor_x']
    mask = (df_broad['CostFactor_y'].notnull())
    df_broad.loc[mask, 'costfactor'] = df_broad.loc[mask, 'CostFactor_y']

    df_broad = df_broad[['index', 'costfactor']]
    dat = merge(dat, df_broad, on=['index'], how='left')

    dat['RA_percentage'] = .3
    dat['p_list_hwma'] = 10 ** asarray(dat['p_list_hwma_log'])
    dat['p_bid'] = dat['p_list_hwma'] * dat['p_pct_list_hwma']
    dat['cost'] = dat['p_list_hwma'] * dat['costfactor']
    dat['PTI_0price'] = dat['p_list_hwma'] * (dat['costfactor'] / (1 - dat['RA_percentage']))
    dat['PTI_5price'] = dat['p_list_hwma'] * (dat['costfactor'] / (1 - dat['RA_percentage'] - 0.05))
    # GP = QP - LP * CF - QP * RA
    # dat['PTI_actual'] = 1 - ((dat['p_list_hwma'] * dat['costfactor']) / dat['p_bid']) - dat['RA_percentage']

    return dat

def pti_tsso(dat, df_cf):
    df_cf.rename(
        columns={'EUROPE & MEA IOTs\nCOST FACTORS\n(not for use in the\n"Cross-Brand Pricing Tool")': 'CostFactor'},
        inplace=True)
    df_cf = df_cf.loc[:, ['Type', 'Model', 'TypeModel', 'Brand', 'CostFactor']]
    # Clean up data
    df_cf.Type = df_cf.Type.astype('str').str.strip()
    df_cf.Model = df_cf.Model.astype('str').str.strip()
    df_cf.TypeModel = df_cf.TypeModel.astype('str').str.strip()
    df_cf.Brand = df_cf.Brand.astype('str').str.strip()

    df_cf.Type = df_cf.Type.str.lower()
    df_cf.Model = df_cf.Model.str.lower()
    df_cf.TypeModel = df_cf.TypeModel.str.lower()

    dat = dat.reset_index()

    dat["machine_type"] = dat["mtm"].str.slice(start=0, stop=4).str.strip()
    dat["machine_model"] = dat["mtm"].str.slice(start=4).str.strip()
    dat["machine_type"] = dat.machine_type.str.lower()
    dat["machine_model"] = dat.machine_model.str.lower()

    dat_mtm = dat[['index', 'machine_type', 'machine_model']].copy()
    df_specific = merge(dat_mtm, df_cf, left_on=['machine_model', 'machine_type'], right_on=['Model', 'Type'],
                        how='left')
    df_cf_broad = df_cf[df_cf['Model'] == "all"]
    df_broad = merge(df_specific, df_cf_broad, left_on=['machine_type'], right_on=['Type'], how='left')

    mask = (df_broad['CostFactor_x'].notnull())
    df_broad.loc[mask, 'costfactor'] = df_broad.loc[mask, 'CostFactor_x']
    mask = (df_broad['CostFactor_y'].notnull())
    df_broad.loc[mask, 'costfactor'] = df_broad.loc[mask, 'CostFactor_y']

    df_broad = df_broad[['index', 'costfactor']]
    dat = merge(dat, df_broad, on=['index'], how='left')

    dat['RA_percentage'] = .3
    # dat['p_list_hwma'] = 10 ** asarray(dat['p_list_hwma_log'])
    dat['p_bid'] = dat['p_list_hwma'] * dat['p_pct_list_hwma']
    dat['cost'] = dat['p_list_hwma'] * dat['costfactor']
    dat['PTI_0price'] = dat['p_list_hwma'] * (dat['costfactor'] / (1 - dat['RA_percentage']))
    dat['PTI_5price'] = dat['p_list_hwma'] * (dat['costfactor'] / (1 - dat['RA_percentage'] - 0.05))
    # GP = QP - LP * CF - QP * RA
    # dat['PTI_actual'] = 1 - ((dat['p_list_hwma'] * dat['costfactor']) / dat['p_bid']) - dat['RA_percentage']

    return dat


def func(p, X, y):
    return 1.0 - (1.0 + p[0] * exp(-p[1] * X)) ** p[2] - y  # Generalized Logistic functions


def wr_train_hw(data, config, uplift=1.0):

    data['normalized_c'] = data['cost_hw'] / (data['y_pred'] * data['p_list_hw'])

    percentile = config['percentile']
    # Initializations
    er = [0] * data.shape[0]
    op = [0] * data.shape[0] # Cost
    op2 = [0] * data.shape[0] # PTI
    P = [[0, 0, 0]] * data.shape[0]
    winrate_o = [0] * data.shape[0] # win probability at optimal price op
    winrate_o2 = [0] * data.shape[0]  # win probability at optimal price op2
    winrate_a = [0] * data.shape[0] # win probability at actual price
    ac_price_normalized = [0] * data.shape[0]

    y = array(percentile[::-1])
    X2 = linspace(0, 6, 301)

    # Error Check
    cols = [data.columns.get_loc(c) for c in [c for c in data.columns if 'pred_' in c]]
    if len(cols) != len(percentile):
        print("Error in length of the number of quantiles (len(Q))!")
        return data

    # Regression Iterations
    print("Calculating win-rates for hw...")

    hundred_points = floor(linspace(0, data.shape[0], 100))

    for i in range(data.shape[0]):
        if i in hundred_points:
            print("Processing win rates calculations: {i} number of data completed".format(i=i))

        X = array(data.iloc[i][cols].tolist()) * uplift  # uplift
        res_lsq = least_squares(func, ones(3), loss="soft_l1", f_scale=1, args=(X, y))

        # save res_lsq.x in P and stack them
        P[i] = res_lsq.x

        y2 = func(res_lsq.x, X2, 0)

        m = (X2 - data.iloc[i]["normalized_c"]) * y2
        m2 = X2 * y2
        #
        er[i] = res_lsq.cost
        op[i] = X2[argmax(m)]
        op2[i] = X2[argmax(m2)]
        winrate_o[i] = func(P[i], op[i], 0)
        winrate_o2[i] = func(P[i], op2[i], 0)
        ac_price_normalized[i] = data.iloc[i]["p_bid_hw"] / (data.iloc[i]['y_pred'] * data.iloc[i]["p_list_hw"])
        winrate_a[i] = func(P[i], ac_price_normalized[i], 0)

    PP = DataFrame(P, columns=["P1", "P2", "P3"])
    data.reset_index(drop=True, inplace=True)
    PP.reset_index(drop=True, inplace=True)
    data = concat([data, PP], axis=1, join_axes=[data.index])

    data['op_GP_pct_list_hw'] = op * data['y_pred']
    data['op_REV_pct_list_hw'] = op2 * data['y_pred']
    data['optimal_price_GP_unbounded_hw'] = data['op_GP_pct_list_hw'] * data["p_list_hw"]
    data['optimal_price_REV_unbounded_hw'] = data['op_REV_pct_list_hw'] * data["p_list_hw"]
    data["Win_Rate_at_OP_GP_hw"] = winrate_o
    data["Win_Rate_at_OP_REV_hw"] = winrate_o2

    data["Win_Rate_at_AP_hw"] = winrate_a
    return data


def wr_train_tss(data, config, uplift=1.0):

    data['normalized_c'] = data['cost'] / (data['y_pred'] * data['p_list_hwma'])
    data['normalized_c2'] = data['PTI_0price'] / (data['y_pred'] * data['p_list_hwma'])

    percentile = config['percentile']
    # Initializations
    er = [0] * data.shape[0]
    op = [0] * data.shape[0] # Cost
    op2 = [0] * data.shape[0] # PTI
    op3 = [0] * data.shape[0] # Revenue
    P = [[0, 0, 0]] * data.shape[0]
    winrate_o = [0] * data.shape[0] # win probability at optimal price op
    winrate_o2 = [0] * data.shape[0]  # win probability at optimal price op2
    winrate_o3 = [0] * data.shape[0]  # win probability at optimal price op3
    winrate_a = [0] * data.shape[0] # win probability at actual price
    ac_price_normalized = [0] * data.shape[0]

    y = array(percentile[::-1])

    # y = y+(0.5-y)/3 #scaling quantiles

    X2 = linspace(0, 6, 301)

    # Error Check
    cols = [data.columns.get_loc(c) for c in [c for c in data.columns if 'pred_' in c]]
    if len(cols) != len(percentile):
        print("Error in length of the number of quantiles (len(Q))!")
        return data

    # Regression Iterations
    print("Calculating win-rates for tss...")

    hundred_points = floor(linspace(0, data.shape[0], 100))
    for i in range(data.shape[0]):
        if i in hundred_points:
            print("Processing win rates calculations: {i} number of data completed".format(i=i))

        X = array(data.iloc[i][cols].tolist()) * uplift  # uplift

        if 'taxon_hw_level_3' in data.columns:  # component-level data
            if data.iloc[i]['taxon_hw_level_3'] == 'pwr':
                y_new = 0.5625 * y + 0.21875
            else:
                y_new = y

        else:  # Quote-Level Data
            if data.iloc[i]['leading_brand'] == 'powerhw':
                y_new = 0.5625 * y + 0.21875
            else:
                y_new = y

        res_lsq = least_squares(func, ones(3), loss="soft_l1", f_scale=1, args=(X, y_new))

        # save res_lsq.x in P and stack them
        P[i] = res_lsq.x

        y2 = func(res_lsq.x, X2, 0)

        m = (X2 - data.iloc[i]["normalized_c"]) * y2
        # m = (1 - data.iloc[i]['normalized_c']/(X2+.00001))*y2
        m2 = (X2 - data.iloc[i]["normalized_c2"]) * y2
        m3 = X2 * y2
        #
        er[i] = res_lsq.cost
        op[i] = X2[argmax(m)]
        op2[i] = X2[argmax(m2)]
        op3[i] = X2[argmax(m3)]
        winrate_o[i] = func(P[i], op[i], 0)
        winrate_o2[i] = func(P[i], op2[i], 0)
        winrate_o3[i] = func(P[i], op3[i], 0)
        ac_price_normalized[i] = data.iloc[i]["p_bid_hwma"] / (data.iloc[i]['y_pred'] * data.iloc[i]["p_list_hwma"])
        winrate_a[i] = func(P[i], ac_price_normalized[i], 0)

    PP = DataFrame(P, columns=["P1", "P2", "P3"])
    data.reset_index(drop=True, inplace=True)
    PP.reset_index(drop=True, inplace=True)
    data = concat([data, PP], axis=1, join_axes=[data.index])

    data['op_GP_pct_list_hwma'] = op * data['y_pred']
    data['op_PTI_pct_list_hwma'] = op2 * data['y_pred']
    data['op_REV_pct_list_hwma'] = op3 * data['y_pred']
    data['optimal_price_GP_unbounded_tss'] = data['op_GP_pct_list_hwma'] * data["p_list_hwma"]
    data['optimal_price_PTI_unbounded_tss'] = data['op_PTI_pct_list_hwma'] * data["p_list_hwma"]
    data['optimal_price_REV_unbounded_tss'] = data['op_REV_pct_list_hwma'] * data["p_list_hwma"]
    data["Win_Rate_at_OP_GP_tss"] = winrate_o
    data["Win_Rate_at_OP_PTI_tss"] = winrate_o2
    data["Win_Rate_at_OP_REV_tss"] = winrate_o3

    data["Win_Rate_at_AP_tss"] = winrate_a

    return data
