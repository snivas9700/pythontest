"""
Created on Thu May  3 15:13:38 2018

@author: yxchen
"""

#import pickle
#from numpy import inf, intersect1d, mean, log10
#from pandas import DataFrame, Series, set_option, merge
#from offline.conversions import component_parse_wrapper, process_params
#from offline.method_defs import segmethod, regmethod
#from offline.modeling_hw import build_lme_model, train_lme_model, run_quants_offline
#from offline.optimization import optprice_mtier
#from offline.post_hoc import post_hoc_wrapper, post_process_op
#from offline.segment_selection import regress_mtier
#from offline.segmentation import segment, seg_to_dict
#from shared.io_utils import write_models, load_data, load_models
#from shared.modeling_hw import run_lme_model, quant_reg_pred, calc_mf_opt
#from shared.modeling_utils import price_adj, calc_quant_op, calc_win_prob, opt_price_ci, bound_quant, bound_op
#from shared.utils import parse_regmethod
#from shared.utils_hw import spread_comp_quants, build_output, build_quarter_map

from datetime import datetime as dt
from os import path as os_path
from numpy import log10, intersect1d, mean
from pandas import DataFrame, Series, set_option
from modeling.modeling_main import apply_lme
from modeling.segmentation_utils import find_seg_id
from shared.io_utils import load_data, load_models
from shared.data_prep import prep_comp, prep_quote
from shared.modeling_hw import quant_reg_pred, calc_mf_opt, bound_lme_discount
from shared.modeling_utils import price_adj, calc_quant_op, calc_win_prob, opt_price_ci, bound_quant, bound_op
from shared.utils_hw import spread_comp_quants, build_output


#from offline_main import run_quant_testing, run_pp_redist

set_option('display.width', 180)

try:
    wd = os_path.dirname(__file__)
except NameError:
    print('Working locally')
    wd = '/Users/yxchen/PycharmProjects/copra_hw_port36'

print('working directory',wd)
DATA_PATH = os_path.join(wd, 'data')

# All fields that will be used in the LME model need to have established data types, and must be
#   listed in the FIELD_TYPES dict so that they can be converted to R vectors of the same types
FIELD_TYPES = {
    'comcostpofl': 'float'
    , 'comloglistprice': 'float'
    , 'com_contrib': 'float'
    , 'upgmes': 'int'
    , 'ln_ds': 'float'
    , 'indirect': 'int'
    , 'comdelgpricel4pofl': 'float'
    , 'endofqtr': 'int'
    , 'crmindustryname': 'str'
    , 'crmsectorname': 'str'
    , 'winloss': 'int'
    , 'lvl4': 'str'
    , 'lvl3': 'str'
    , 'lvl2': 'str'
    , 'lvl1': 'str'
    , 'n_sbc': 'int'
    , 'ufc_incl': 'int'
    , 'sbc_incl': 'bool'
    , 'discount': 'float'
    }

FIELD_MAP = {
    'featureid': 'ufc'
    , 'comspclbidcode1': 'sbc1'
    , 'comspclbidcode2': 'sbc2'
    , 'comspclbidcode3': 'sbc3'
    , 'comspclbidcode4': 'sbc4'
    , 'comspclbidcode5': 'sbc5'
    , 'comspclbidcode6': 'sbc6'
    , 'level_0': 'lvl0'
    , 'level_1': 'lvl1'
    , 'level_2': 'lvl2'
    , 'level_3': 'lvl3'
    , 'level_4': 'lvl4'
    , 'indirect(1/0)': 'indirect'
    , 'clientseg=e': 'client_e'
    , 'comlistprice': 'list_price'
    , 'comquoteprice': 'quoted_price'
    , 'comtmc': 'tmc'
}

COMP_META_COLS = ['quoteid', 'quoted_price', 'winloss', 'list_price', 'indirect']
# for parsing the VS model to dict - list out fixed effects categoricals and binaries
KNOWN_CATEGORICALS = ['lvl4', 'lvl3', 'lvl2', 'lvl1']
KNOWN_BINARIES = ['upgmes', 'indirect', 'endofqtr']

# "uplift" term to scale L/M/H price points AFTER adjustment (i.e. post price_adj())
UPLIFT = 1.06
# scale on lower bound, where lower bound = ALPHA*VS
ALPHA = 0.0
print('')
print('**********************')
print('Starting testing flow...')
print('**********************')
print('')
# define functions 

def run_quant_testing(data, models, idx_cols, grp_cols, uplift=1., alpha=0.0):
#    raw_qs = models['raw_qs']
    qs = models['raw_qs']
    pred_mods = models['pred_qs']['models']
    in_feats = models['pred_qs']['in_feats']

    # calculate bounded quantiles for component- or quote-level data
    # if level == 'quote':
    q_track = DataFrame()
    # known a priori
#    q_list = raw_qs.columns
    q_list = qs.columns
    pred_cols = ['pred_' + x for x in q_list]
    adj_cols = ['adj_' + x for x in pred_cols]
    # make joining in later easy
    data = data.set_index(idx_cols)

    for s_id, grp in data.groupby(grp_cols):
        if s_id is not None and s_id in pred_mods.index.levels[0]:  # try to match identifying field
            mods = pred_mods.xs([s_id], level=grp_cols)
#            qs = raw_qs.loc[s_id]
        else:  # can't match, defualt to "ALL"
            mods = pred_mods.xs(['ALL'], level=grp_cols)
#            qs = raw_qs.loc['ALL']

        q_res = quant_reg_pred(grp, in_feats, mods, q_list)
        # mix raw quantiles + quantile regressions
        quants = q_res[pred_cols + grp_cols].apply(lambda row: bound_quant(row, qs, grp_cols), axis=1)

        q_track = q_track.append(quants)

    # run model factory L/M/H price adjustments
    q_track[adj_cols] = q_track.apply(lambda row: Series(price_adj(*row[pred_cols].values)), axis=1) * uplift

    data = data.join(q_track, how='left').reset_index()

    data.to_csv('./debug/quant_pred.csv')

    data = data.apply(lambda row: calc_mf_opt(row, alpha), axis=1)

    return data


def run_pp_redist(comp_quant, opt_quant, source, alpha):
    t3 = dt.now()
    # redistribute L/M/H price points
    # Filter quotes/components so nothing breaks
    shared_qids = intersect1d(comp_quant['quoteid'].unique(), opt_quant['quoteid'].unique())

    failed = comp_quant.loc[(comp_quant['price_opt'] <= 0.) | (comp_quant['value_score'] <= 0), :].reset_index(drop=True)
    failed_ids = failed['quoteid'].unique()

    c_mask = comp_quant['quoteid'].isin(shared_qids) & ~comp_quant['quoteid'].isin(failed_ids)
    q_mask = opt_quant['quoteid'].isin(shared_qids) & ~opt_quant['quoteid'].isin(failed_ids)

    # select out shared quotes
    compq = comp_quant.loc[c_mask].copy().reset_index(drop=True)
    quoteq = opt_quant.loc[q_mask].copy().reset_index(drop=True)

    # redistribute L/M/H price points
    cq, _ = spread_comp_quants(compq, quoteq, 'offline', alpha)

    # calculate outputs per online flow + spread optimal price
    q_out, tot_stats = build_output(cq, 'offline')

    if source == 'train':
        mape = mean(abs(q_out['bot_spread_OP'] - q_out['price_act'])/q_out['price_act'])
        print(('MAPE of post-spread comp-level OP: {}'.format(mape)))

    print(('Redistribution of price points took {}'.format(dt.now() - t3)))

    return q_out, tot_stats



# define region
#region = 'NA_newmap_4lvl_DirectOnly_non_SAPHANA'
#region = 'NA_direct_nonSAPHANA_apprved'
region = 'NA_direct_nonSAPHANA_apprved_quant_intercept'
src_path = os_path.join(DATA_PATH, region)
sbc_map = load_data(src_path, 'SBC_table')
model_path = os_path.join(wd,'models',region)

t_test = dt.now()
test_data = load_data(src_path, 'testing').rename(columns=FIELD_MAP)
# test_quarters = build_quarter_map(test_data)
test_pn = prep_comp(test_data, sbc_map)
# set testing winloss = 1 before applying LME 
test_pn['winloss_orig'] = test_pn['winloss']
test_pn.loc[:,'winloss'] = 1
# load in applicable models
loaded_models = {region: load_models(region=region, compressed=False)}

lme = loaded_models[region][0]
seg_defs = loaded_models[region][1]
quant_models = loaded_models[region][2]


print('Applying LME...')
t_lme = dt.now()
#test_pn['discount_pred'] = test_pn.apply(lambda row: apply_lme(row, lme), axis=1)
test_pn['discount_pred'] = test_pn.apply(lambda row: bound_lme_discount(row, lme), axis=1)
print(('Application of lme on {n} samples took {t}'.format(n=test_pn.shape[0], t=dt.now()-t_lme)))

test_pn['value_score'] = (1. - test_pn['discount_pred']) * test_pn['list_price']
test_pn['ln_vs'] = test_pn['value_score'].apply(lambda x: log10(x + 1))

test_pn.to_csv(os_path.join(src_path, 'test_vs.csv'), index=False)

t_qt = dt.now()

test_qt = prep_quote(test_pn)
test_qt['segment_id'] = test_qt.apply(lambda row: find_seg_id(row, seg_defs), axis=1)
# set testing winloss = 1 
test_qt['winloss_orig'] = test_qt['winloss']
test_qt['winloss'] = 1

print('Applying quantile regression models...')
#print('Adding the quantile regression intercept column to the design matrix for quote-level data')
#test_qt['quant_intercept'] = 1 # add quantile regression intercept column

#print('Adding the quantile regression intercept column to the design matrix for comp-level data')
#test_pn['quant_intercept'] = 1 # add quantile regression intercept column

test_quote = run_quant_testing(test_qt, quant_models['qt'], ['quoteid'], ['segment_id'], UPLIFT)
test_comp = run_quant_testing(test_pn, quant_models['pn'], ['quoteid', 'componentid'], ['lvl1'], UPLIFT)

test_q_out, test_tot_stats = run_pp_redist(test_comp, test_quote, source='test', alpha=ALPHA)

# add actual winprob and bot_spread winprob to component-level output
test_q_out['wp_act'] = test_q_out.apply(lambda row: calc_win_prob(row['quoted_price'],
          row['adj_price_L'],row['adj_price_M'],row['adj_price_H']),axis = 1)
test_q_out['wp_bot_OP'] = test_q_out.apply(lambda row: calc_win_prob(row['bot_spread_OP'],
          row['adj_price_L'],row['adj_price_M'],row['adj_price_H']),axis = 1)
# add actual winprob to quote-level output
q_p = DataFrame(test_q_out.groupby('quoteid')['quoted_price'].sum()).reset_index()
q_p.rename(columns={'quoted_price':'price_act'},inplace=True)

test_tot_stats = test_tot_stats.merge(q_p,on='quoteid',how='left')
test_tot_stats['act_wp'] = test_tot_stats.apply(lambda row: calc_win_prob(row['price_act'],
          row['adj_price_L'],row['adj_price_M'],row['adj_price_H']),axis = 1)

print(('Quote-level testing calcs took {}'.format(dt.now() - t_qt)))

print(('Entire testing flow took {}'.format(dt.now() - t_test)))

test_q_out.to_csv(os_path.join(model_path, 'quote_df_test.csv'))
test_tot_stats.to_csv(os_path.join(model_path, 'total_deal_stats_test.csv'))
