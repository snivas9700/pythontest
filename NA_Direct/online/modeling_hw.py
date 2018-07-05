from shared.modeling_hw import quant_reg_pred, calc_mf_opt
from shared.modeling_utils import price_adj, bound_quant


def run_quants_online(data_obj, models, seg_field='segment_id', uplift=1., alpha=0.0):
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
    quants = bound_quant(q_res, qs, grp_cols=[seg_field])

    # adjust quantile estimates here
    data_obj['adj_pred_L'], data_obj['adj_pred_M'], data_obj['adj_pred_H'] = [
        x*uplift for x in price_adj(*quants[['pred_L', 'pred_M', 'pred_H']].values)]

    for q in q_list:
        pred_col = 'adj_pred_' + q
        price_col = 'adj_price_' + q
        data_obj[price_col] = data_obj[pred_col] * data_obj['value_score']

    data_obj = calc_mf_opt(data_obj, alpha)

    return data_obj
