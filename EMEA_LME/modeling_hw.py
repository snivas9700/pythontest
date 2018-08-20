from __future__ import division

from numpy import absolute, mean, inf, sum as np_sum
from pandas import Series, DataFrame, concat
from time import time
from copy import deepcopy

from statsmodels.regression.quantile_regression import QuantReg

from EMEA_LME.modeling_utils import price_adj, bound_quant



def quant_reg_train(train_data, qs, in_feats, out_feat, label_str):

    # NOTE: endogenous = response variable, exogenous = explanatory variable(s)
    mod = QuantReg(endog=train_data[out_feat].values, exog=train_data[in_feats])

    # m_dict = {}
    m_track = DataFrame()

    labs = label_str  # L = 0.05, M = 0.5, H = 0.95
    for i, q in enumerate(qs):
        res = mod.fit(q=q,max_iter=2000)  # fit quantile q
        p = res.params.to_frame().T
        p.index = [labs[i]]  # assign model parameters to L/M/H
        m_track = m_track.append(p)  # store parameters

    return m_track


def quant_reg_pred(test_data, in_feats, mods, labs):

    for lab in labs:
        w = mods.loc[lab].values  # L/M/H model parameters (weights)
        X = test_data[in_feats].values

        if len(X.shape) > 1:
            pred = np_sum(w*X, axis=1)  # make predictions
        else:
            pred = np_sum(w*X)

        col = 'pred_' + lab  # prediction label

        test_data[col] = pred  # insert predictions back into data object

    return test_data


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

def run_quants_online(data_obj, models, seg_field, uplift=1., alpha=0.0):
    qs = models['raw_qs']
    q_mods = models['pred_qs']['models']
    in_feats = models['pred_qs']['in_feats']

    s_id = data_obj[seg_field]  # extract segment_id field value for this quote/component
    if s_id is not None and s_id in q_mods.index.levels[0]:  # s_id exists and is in the known list of models
        mods = q_mods.xs(s_id, level=seg_field)
    else:  # segmentation can't match. default to ALL
        print(('Unable to find lvl1 value {}'.format(s_id)))
        mods = q_mods.xs('ALL', level=seg_field)

    q_list = qs.columns
    q_res = quant_reg_pred(data_obj, in_feats, mods, q_list)

    # adjust the quantile regression results with the results of the rank-ordered quantiles
    # quants = bound_quant(q_res, qs, grp_cols=[seg_field])

    # adjust quantile estimates here
    # data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'] = [
    #     x*uplift for x in price_adj(*quants[['pred_L', 'pred_M', 'pred_H']].values)]

    # for q in q_list:
    #     pred_col = 'adj_pred_' + q
    #     price_col = 'adj_price_' + q
    #     data_obj[price_col] = data_obj[pred_col] * data_obj['value_score']

    # data_obj = calc_mf_opt(data_obj, alpha)

    return q_res