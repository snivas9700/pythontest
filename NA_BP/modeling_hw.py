

from numpy import absolute, mean, inf
from pandas import Series, DataFrame, concat
from time import time
from copy import deepcopy

from statsmodels.regression.quantile_regression import QuantReg

from lme import lmeMod
from modeling_utils import price_adj, calc_quant_op, calc_win_prob, opt_price_ci, bound_quant, bound_op


def build_model(train_data, formula_fe, formula_re):
    mod = lmeMod(train_data, formula_fe, formula_re)
    return mod


def train_model(mod):
    lme_start_time = time()
    mod = mod.fit()

    print("Fitting lme model takes", round(time()-lme_start_time, 2), "seconds.")
    return mod


def run_model(mod, test_data, label='TESTING'):
    y_pred = mod.predict(test_data).fillna(0.)
    test_data['discount_pred'] = y_pred.values

    # enforce hard bounds on the discount prediction
    test_data.loc[test_data['discount_pred'] > 1.0, 'discount_pred'] = 1.0
    test_data.loc[test_data['discount_pred'] < 0.0, 'discount_pred'] = 0.0

    rel_dist = absolute(test_data['discount_pred'] - test_data['discount']) / test_data['discount']
    # clean up distances
    rel_dist = rel_dist.fillna(0.).replace([-inf, inf], 0.0)  # any discount = 0.0 generates distance of +/-inf
    mape = mean(rel_dist)
    print(('MAPE on {l} predictions: {m}'.format(l=label, m=mape)))

    return test_data


def quant_reg_train(train_data, qs, in_feats, out_feat):

    # NOTE: endogenous = response variable, exogenous = explanatory variable(s)
    mod = QuantReg(endog=train_data[out_feat].values, exog=train_data[in_feats])

    m_dict = {}

    labs = ['L', 'M', 'H']  # L = 0.05, M = 0.5, H = 0.95
    for i, q in enumerate(qs):
        res = mod.fit(q=q)  # fit quantile q
        m_dict.update({labs[i]: deepcopy(res)})  # store trained model

    return m_dict


def quant_reg_pred(test_data, in_feats, m_dict, labs):

    for lab in labs:
        res = m_dict[lab]  # L/M/H model

        pred = res.predict(exog=test_data[in_feats].values)  # make predictions

        col = 'pred_' + lab  # prediction label

        if isinstance(test_data, Series):
            test_data[col] = pred[0]
        else:
            test_data[col] = pred  # insert predictions back into data object

    return test_data


def run_quants_offline(data, idx_cols, in_feats, out_feat, grp_cols):
    dat = data.copy()

    # store trained model objects
    mod_out = {}

    if len(grp_cols) > 0:
        # raw quantile calculations
        qs = concat([dat.groupby(grp_cols)[out_feat].apply(lambda s: s.quantile(q=q)).to_frame(str(q))
                        for q in [0.05, 0.5, 0.95]], axis=1)
        qs = qs.rename(columns={'0.05': 'L', '0.5': 'M', '0.95': 'H'})

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
                test.loc[:, 'winloss'] = 1
                q_mods = quant_reg_train(train.set_index(idx_cols), [0.05, 0.5, 0.95], in_feats, out_feat)
                q_res = quant_reg_pred(test.set_index(idx_cols), in_feats, q_mods, qs.columns)

                q_reg = q_reg.append(q_res)
                mod_out.update({t3: q_mods})
            else:
                print(('No win data in {g}={v}'.format(g=grp_cols, v=t3)))

    else:
        qs = Series(data={str(q): dat[out_feat].quantile(q=q) for q in [0.05, 0.5, 0.95]})
        qs = qs.rename(index={'0.05': 'L', '0.5': 'M', '0.95': 'H'})

        train = dat
        test = dat.copy()
        test.loc[:, 'winloss'] = 1
        q_mods = quant_reg_train(train.set_index(idx_cols), [0.05, 0.5, 0.95], in_feats, out_feat)
        q_reg = quant_reg_pred(test.set_index(idx_cols), in_feats, q_mods, qs.index)
        mod_out.update({'ALL': q_mods})

    pred_cols = [c for c in q_reg.columns if 'pred_' in c]  # get prediction labels

    return q_reg, qs, pred_cols, mod_out


# separates out calculation of OP-related values from prediction of L/M/H price points
# applies model factory functions to the data
# assumes data_obj is a single entry (row) of the data
def calc_mf_opt(data_obj, alpha):
    data_obj['cf'] = data_obj['tmc']/data_obj['value_score']

    data_obj['norm_op'] = calc_quant_op(
        data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'], data_obj['cf'], 0., 0., 0.)

    data_obj['price_opt'] = data_obj['norm_op'] * data_obj['value_score']

    data_obj = bound_op(data_obj, alpha)

    # win probability calculated on normalized OP
    data_obj['wp_opt'] = calc_win_prob(data_obj['norm_op'], data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'])

    data_obj['ci_low'], data_obj['ci_high'] = opt_price_ci(
        data_obj['norm_op'], data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'], data_obj['cf'])

    return data_obj
