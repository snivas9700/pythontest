import pickle
from datetime import datetime as dt
from os import path as os_path

from numpy import inf, intersect1d, mean, log10
from pandas import DataFrame, Series, set_option, merge
#%%
from modeling.modeling_main import apply_lme
from modeling.segmentation_utils import find_seg_id
from offline.conversions import component_parse_wrapper, process_params
from offline.method_defs import segmethod, regmethod
from offline.modeling_hw import build_lme_model, train_lme_model, run_quants_offline
from offline.optimization import optprice_mtier
from offline.post_hoc import post_hoc_wrapper, post_process_op
from offline.segment_selection import regress_mtier
from offline.segmentation import segment, seg_to_dict
from shared.data_prep import prep_comp, prep_quote
from shared.io_utils import write_models, load_data
from shared.modeling_hw import run_lme_model, quant_reg_pred, calc_mf_opt
from shared.modeling_utils import price_adj, calc_quant_op, calc_win_prob, opt_price_ci, bound_quant, bound_op
from shared.utils import parse_regmethod
from shared.utils_hw import spread_comp_quants, build_output, build_quarter_map

set_option('display.width', 180)

try:
    wd = os_path.dirname(__file__)
except NameError:
    print('Working locally')
    wd = '/Users/bonniebao/Box Sync/copra_hw_jp_py3/copra_hw-py3_port'


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
    #, 'n_sbc': 'int'
    #, 'ufc_incl': 'int'
    #, 'sbc_incl': 'bool'
    , 'discount': 'float'
    }

FIELD_MAP = {
    #'featureid': 'ufc'
    #, 'comspclbidcode1': 'sbc1'
    #, 'comspclbidcode2': 'sbc2'
    #, 'comspclbidcode3': 'sbc3'
    #, 'comspclbidcode4': 'sbc4'
    #, 'comspclbidcode5': 'sbc5'
    #, 'comspclbidcode6': 'sbc6'
    #, 'level_0': 'lvl0'
     'level_1': 'lvl1'
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
UPLIFT = 1.1
# scale on lower bound, where lower bound = ALPHA*VS
ALPHA = 0.0

# define the config of each run
config = {'model':
              {'fe': 'discount ~ 1 + comcostpofl + comloglistprice + com_contrib + upgmes + '
                     'ln_ds + indirect + comdelgpricel4pofl + endofqtr + winloss' #ufc_incl + winloss + n_sbc' #edited by Bonnie
                , 're': '(comloglistprice + comcostpofl || lvl4/lvl3/lvl2/lvl1) + (comloglistprice + comcostpofl || crmindustryname/crmsectorname)'
               }
            , 'wr_feats': parse_regmethod(regmethod)[1]
            , 'quant_feats': {'in_feats': ['quant_intercept','ln_ds', 'ln_vs'], 'target': 'metric'} #added 'quant_intercept' by BB on May 30th, 2018
            , 'uplift': UPLIFT
            , 'alpha': ALPHA
            , 'misc': 'includes component segmentation'
          }

# dictate region here
summary_dict = {}

# flag for triggering testing flow
RUN_TEST = False

output_folder = os_path.join(wd, 'output')


def run_wr_training(train_qt):
    # Function to pack together the relevant pieces of the regmethod dict
    reg_vals = parse_regmethod(regmethod)

    # Run quote-level regression
    data_vol = 0.9  # max volume of data to drop when dropping outliers
    min_score = 0  # min data_score() to calculate after dropping values

    # NOTE: This is the main win-rate training step
    outdf_qt = regress_mtier(train_qt, reg_vals, data_vol, min_score)

    # ## Calculating the optimal price depending on the final output dataframe.
    tier_list = [x for x in outdf_qt.index.names if 'tier_' in x]

    x_var = reg_vals[1]

    suffix = len(tier_list) - 1
    x_var_suff = [var + '_' + str(suffix) for var in x_var]
    coef_qt = outdf_qt[x_var_suff]

    train_op = optprice_mtier(train_qt, coef_qt)

    train_op['ape'] = abs(train_op['quoted_price'] - train_op['price_opt']) / train_op['quoted_price']

    return train_op


def run_mf_calcs(train_op, pred_cols, adj_cols, alpha):
    train_op['cf'] = train_op['tmc']/train_op['value_score']

    train_op['norm_op'] = train_op.apply(
        lambda row: calc_quant_op(*row[adj_cols].values, cf=row['cf'], cv=0., fcw=0., fcl=0.), axis=1)

    train_op['price_opt'] = train_op['norm_op'] * train_op['value_score']  # calc_quant_op predicts QP/VS

    train_op = bound_op(train_op, alpha)

    train_op['wp_opt'] = train_op.apply(
        lambda row: calc_win_prob(row['norm_op'], *row[adj_cols].values), axis=1)

    train_op['wp_act'] = train_op.apply(
        lambda row: calc_win_prob(row['metric'], *row[adj_cols].values), axis=1)

    train_op['price_act'] = train_op['quoted_price'].copy()

    train_op[['ci_low', 'ci_high']] = train_op.apply(lambda row: Series(
        opt_price_ci(row['norm_op'], *row[adj_cols].values, Cf=row['cf'], Cv=0, FCw=0, FCl=0, Pct=.95)),
        axis=1)

    op_cols = []
    for col in pred_cols + adj_cols:
        c = col.split('pred_')[-1]
        c = 'price_' + c
        train_op[c] = train_op[col] * train_op['value_score']
        op_cols.append(c)  # for easy access later

    return train_op

#%%
def run_quant_training(data, config, idx_cols=['quoteid'], grp_cols=[], uplift=1., alpha=0.0):
    # prep dependent variable
    data['metric'] = (data['quoted_price'] / data['value_score']).replace([-inf, inf], 0.0)  # inf -> 0.0

    # run quantile approaches; quantile regression + raw calculated quantiles
    quants, qs, pred_cols, models = run_quants_offline(
        data.copy(), idx_cols, in_feats=config['quant_feats']['in_feats'], out_feat=config['quant_feats']['target'],
        grp_cols=grp_cols)

    # save as variable for easy access
    adj_cols = ['adj_' + x for x in pred_cols]

    # mix raw quantiles + quantile regressions
    quants = quants.apply(lambda row: bound_quant(row, qs, grp_cols), axis=1)[pred_cols]

    # run model factory L/M/H price adjustments
    quants[adj_cols] = quants.apply(lambda row: Series(price_adj(*row[pred_cols].values)), axis=1) * uplift

    # merge quantile calcs into training data
    train_op = merge(data, quants.reset_index(), on=idx_cols, how='left')

    # run OP, WP, CI calcs
    train_op = run_mf_calcs(train_op, pred_cols, adj_cols, alpha)
    
    return train_op, models, qs

#%%
def run_quant_testing(data, models, idx_cols, grp_cols, uplift=1., alpha=0.0):
    raw_qs = models['raw_qs']

    pred_mods = models['pred_qs']['models']
    in_feats = models['pred_qs']['in_feats']

    # calculate bounded quantiles for component- or quote-level data
    # if level == 'quote':
    q_track = DataFrame()
    # known a priori
    q_list = raw_qs.columns
    pred_cols = ['pred_' + x for x in q_list]
    adj_cols = ['adj_' + x for x in pred_cols]
    # make joining in later easy
    data = data.set_index(idx_cols)

    for s_id, grp in data.groupby(grp_cols):
        if s_id in list(pred_mods.keys()):  # try to match identifying field
            mods = pred_mods.xs(s_id, level=grp_cols)
            qs = raw_qs.loc[s_id]
        else:  # can't match, defualt to "ALL"
            mods = pred_mods.xs('ALL', level=grp_cols)
            qs = raw_qs.loc['ALL']

        q_res = quant_reg_pred(grp, in_feats, mods, q_list)
        # mix raw quantiles + quantile regressions
        quants = q_res[pred_cols].apply(lambda row: bound_quant(row, qs, grp_cols), axis=1)

        q_track = q_track.append(quants)

    # run model factory L/M/H price adjustments
    q_track[adj_cols] = q_track.apply(lambda row: Series(price_adj(*row[pred_cols].values)), axis=1) * uplift

    data = data.join(q_track, how='left').reset_index()

    data = data.apply(lambda row: calc_mf_opt(row, alpha), axis=1)

    return data

#%%
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

#%%
for region in ['JP_direct_only_WL_sw_cost_0_060118']:
    print('Starting run on region {}...'.format(region))

    src_path = os_path.join(DATA_PATH, region)

    train_data = load_data(src_path, 'training').rename(columns=FIELD_MAP)
   # train_data = train_data0.sample(n=300).copy()
    train_quarters = build_quarter_map(train_data)
    #sbc_map = load_data(src_path, 'SBC_table') #commented out by Bonnie

    t1 = dt.now()

    train_pn = prep_comp(train_data)#modified by Bonnie
    t = dt.now()

    mod = build_lme_model(train_pn[list(FIELD_TYPES.keys())], formula_fe=config['model']['fe'], formula_re=config['model']['re'], field_types=FIELD_TYPES)
    vs_mod = train_lme_model(mod)

    with open('vs_mod.pkl', 'wb') as f:
        pickle.dump(vs_mod, f)

    train_vs = run_lme_model(vs_mod, train_pn, 'TRAINING')
    train_vs['value_score'] = (1. - train_vs['discount_pred']) * train_vs['list_price'].astype(float)
    train_vs['ln_vs'] = train_vs['value_score'].apply(lambda x: log10(x + 1))

    print('Training LME model took {}'.format(dt.now() - t))

    params = process_params(vs_mod, train_pn, {}, KNOWN_CATEGORICALS, KNOWN_BINARIES)

    t2 = dt.now()

    train_qt = prep_quote(train_vs)

    grp = train_qt.groupby('leading_brand')['quoteid'].count()
    filt_brand = grp.loc[grp.eq(1)].index
    if len(filt_brand) > 0:
        train_qt = train_qt.loc[~train_qt['leading_brand'].isin(filt_brand)].reset_index(drop=True)

    # Run quote-level segmentation
    train_qt, meta_qt, tier_map_qt = segment(train_qt, segmethod)
    train_qt = train_qt.reset_index(drop=True)

    # Convert quote-level segmentation to tree format
    seg_dict_qt = seg_to_dict(train_qt, meta_qt)
    train_qt['segment_id'] = train_qt.apply(lambda row: find_seg_id(row, seg_dict_qt), axis=1)
#%%    
    print('Starting quote-level quantile regression training...')
    #train_qt['const'] = 1
    opt_quant, pred_qs_qt, raw_qs_qt = run_quant_training(train_qt, config, ['quoteid'], grp_cols=['segment_id'], uplift=UPLIFT, alpha=ALPHA)
    quant_mape = mean(abs(opt_quant['price_opt'] - opt_quant['quoted_price'])/opt_quant['quoted_price'])
    print('MAPE of quote-level quant OP: {}'.format(quant_mape))

    # build quantile modeling output object
    q_quant = {'raw_qs': raw_qs_qt,
               'pred_qs': {
                   'models': pred_qs_qt,
                   'in_feats': config['quant_feats']['in_feats'],
                   'target': config['quant_feats']['target']
                   }
               }

    print('Starting comp-level quantile regression training...')
    #train_vs['const'] = 1
    train_vs.to_csv('./debug/train_vs_test_quant.csv')#added in by BB on May 30th, 2018

    
    comp_quant, pred_qs_pn, raw_qs_pn = run_quant_training(train_vs, config, ['quoteid', 'componentid'], grp_cols=['lvl1'], uplift=UPLIFT, alpha=ALPHA)
    comp_mape = mean(abs(comp_quant['price_opt'] - comp_quant['price_act'])/comp_quant['price_act'])
    print('MAPE of pre-spread comp-level quant OP: {}'.format(comp_mape))

    print('Quantile training took {}'.format(dt.now() - t2))

    c_quant = {'raw_qs': raw_qs_pn,
               'pred_qs': {
                   'models': pred_qs_pn,
                   'in_feats': config['quant_feats']['in_feats'],
                   'target': config['quant_feats']['target']
                    }
               }

    q_out, tot_stats = run_pp_redist(comp_quant, opt_quant, source='train', alpha=ALPHA)

    quant_tot = post_process_op(opt_quant.copy(), train_quarters)

    # TODO - confirm proper output then update post_hoc_wrapper
    hash_folder = post_hoc_wrapper(train_vs, quant_tot, config, region, summary_dict, output_folder)
    #
    # quant_tot.to_csv(os_path.join(hash_folder, region, 'quant_dat_full.csv'), index=False)

    # outdf_qt.to_csv(os_path.join(hash_folder, region, 'outdf_qt.csv'), index=False)
    # train_op.to_csv(os_path.join(hash_folder, region, 'train_op.csv'), index=False)

    # prep model outputs
    model_dict_pn = component_parse_wrapper(params)

    # store outputs
    fpath = os_path.join(hash_folder, region)
    train_vs.to_csv(os_path.join(fpath, 'insample_VS.csv'), index=False)
    train_qt.to_csv(os_path.join(fpath, 'train_qt.csv'), index=False)
    q_out.to_csv(os_path.join(fpath, 'quote_df.csv'))
    tot_stats.to_csv(os_path.join(fpath, 'total_deal_stats.csv'))

    write_models(model_dict_pn, seg_dict_qt, c_quant, q_quant, hash_folder, region, compressed=False)

    print(('Finished training on region {r}. Took {t}'.format(r=region, t=dt.now() - t)))

    if RUN_TEST:
        print('Starting testing flow...')
        t_test = dt.now()
        test_data = load_data(src_path, 'testing').rename(columns=FIELD_MAP) #modified by BB on May 30th, 2018
        # test_quarters = build_quarter_map(test_data)
        test_pn = prep_comp(test_data)

        # TODO - build logic to allow run_model() to work with out of sample predictions
        # test_vs = run_model(vs_mod, test_pn, 'TESTING')

        print('Applying LME...')
        t_lme = dt.now()
        test_pn['discount_pred'] = test_pn.apply(lambda row: apply_lme(row, model_dict_pn), axis=1)
        print(('Application of lme on {n} samples took {t}'.format(n=test_pn.shape[0], t=dt.now()-t_lme)))

        test_pn['value_score'] = (1. - test_pn['discount_pred']) * test_pn['list_price']
        test_pn['ln_vs'] = test_pn['value_score'].apply(lambda x: log10(x + 1))

        test_pn.to_csv(os_path.join(src_path, 'test_vs.csv'), index=False)

        t_qt = dt.now()

        test_qt = prep_quote(test_pn)
        test_qt['segment_id'] = test_qt.apply(lambda row: find_seg_id(row, seg_dict_qt), axis=1)

        print('Applying quantile regression models...')
        test_quote = run_quant_testing(test_qt, q_quant, ['quoteid'], ['segment_id'], UPLIFT)
        test_comp = run_quant_testing(test_pn, c_quant, ['quoteid', 'componentid'], ['lvl1'], UPLIFT)

        test_q_out, test_tot_stats = run_pp_redist(test_comp, test_quote, source='test', alpha=ALPHA)

        print(('Quote-level testing calcs took {}'.format(dt.now() - t_qt)))

        print(('Entire testing flow took {}'.format(dt.now() - t_test)))

        test_q_out.to_csv(os_path.join(fpath, 'quote_df_test.csv'))
        test_tot_stats.to_csv(os_path.join(fpath, 'total_deal_stats_test.csv'))

# build_summary(output_folder)
