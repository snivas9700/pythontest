from __future__ import division

from pandas import Series

from NA_BP.modeling_utils import price_adj, calc_win_prob, opt_price_ci, bound_quant, calc_quant_op, bound_op
import pandas as pd


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


def run_quants_online(data_obj, models, seg_field='segment_id', uplift=1., alpha=0.0):

    qs = models['raw_qs']
    q_mods = models['pred_qs']['models']
    in_feats = models['pred_qs']['in_feats']

    s_id = data_obj[seg_field]  # extract segment_id field value for this quote/component
    if s_id is not None and s_id in q_mods.keys():  # s_id exists and is in the known list of models
        mod = q_mods[s_id]
        q_list = qs.columns
    else:  # segmentation can't match. default to ALL
        #print('Unable to find match for lvl1 value {l}. Defaulting to models for "ALL"'.format(l=s_id))
        mod = q_mods['ALL']
        q_list = qs.columns

    q_res = quant_reg_pred(data_obj, in_feats, mod, q_list)

    # adjust the quantile regression results with the results of the rank-ordered quantiles
    quants = bound_quant(q_res, qs, grp_cols=[seg_field])

    # adjust quantile estimates here
    # NOTE: due to CIO requirements, must preserve adjusted predictions as separate fields
    data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'] = [
        x*uplift for x in price_adj(*quants[['pred_L', 'pred_M', 'pred_H']].values)]
     
    data_obj = pd.DataFrame(data_obj)
    data_obj = data_obj.T
    data_obj.index = range(len(data_obj))
    
    for i in range(len(data_obj)):
        data_obj.loc[i, 'adj_price_L'] = (data_obj.loc[i, 'list_price'] * data_obj.loc[i, 'adj_pred_L'])
        data_obj.loc[i, 'adj_price_M'] = (data_obj.loc[i, 'list_price'] * data_obj.loc[i, 'adj_pred_M'])
        data_obj.loc[i, 'adj_price_H'] = (data_obj.loc[i, 'list_price'] * data_obj.loc[i, 'adj_pred_H'])
           
    data_obj = data_obj.T.squeeze()
    data_obj = calc_mf_opt(data_obj, alpha)

    return data_obj


# separates out calculation of OP-related values from prediction of L/M/H price points
# applies model factory functions to the data
# assumes data_obj is a single entry (row) of the data
def calc_mf_opt(data_obj, uplift):
    data_obj['cf'] = data_obj['tmc']/data_obj['value_score']

    data_obj['norm_op'] = calc_quant_op(
        data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'], data_obj['cf'], 0., 0., 0.)

    data_obj['price_opt'] = data_obj['norm_op'] * data_obj['value_score']

    data_obj = bound_op(data_obj, uplift)

    # win probability calculated on normalized OP
    data_obj['wp_opt'] = calc_win_prob(
        data_obj['norm_op'], data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'])

    data_obj['ci_low'], data_obj['ci_high'] = opt_price_ci(
        data_obj['norm_op'], data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'], data_obj['cf'])

    return data_obj

