from numpy import absolute, mean, inf, sum as np_sum
from pandas import Series, DataFrame
from statsmodels.regression.quantile_regression import QuantReg

from NA.shared.modeling_utils import calc_quant_op, calc_win_prob, opt_price_ci, bound_op
from NA.shared.utils import timeit
from NA.modeling.modeling_main import apply_lme

@timeit
def run_lme_model(mod, test_data, label='TESTING'):
    test_ = test_data.loc[:, mod.field_types.keys()]
    test_.loc[:,'winloss'] = 1  # assume value scores are all win bids
    y_pred = mod.predict(test_).fillna(0.)
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

    # m_dict = {}
    m_track = DataFrame()

    labs = ['L', 'M', 'H']  # L = 0.05, M = 0.5, H = 0.95
    for i, q in enumerate(qs):
        res = mod.fit(q=q)  # fit quantile q
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


def bound_lme_discount(obj,model,default=0):
    """
    set the default lme discount output if it is greater than 1, i.e. negative value score 
    """
    pred_discount = apply_lme(obj,model)
    if pred_discount > 1 :
        print('negative value score occured! pred_discount = ',pred_discount)
        print(obj)
        pred_discount = 0 # discountious jump -- but make sure the value score isn't negative or zero
    return pred_discount
