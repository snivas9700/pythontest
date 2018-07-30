from pandas import DataFrame, Series, set_option, merge
from numpy import inf, intersect1d, mean, log10
from os import path as os_path
from datetime import datetime as dt

from data_prep import prep_comp, prep_quote
from modeling_hw import build_model, train_model, run_model, run_quants_offline
from io_utils import write_models

from segmentation import segment, seg_to_dict
from segment_selection import regress_mtier
from optimization import optprice_mtier

from utils import parse_regmethod, load_data, build_quarter_map
from method_defs import segmethod, regmethod
from post_hoc import post_hoc_wrapper, post_process_op, build_summary
from modeling_utils import price_adj, calc_quant_op, calc_win_prob, opt_price_ci, bound_quant
from conversions import component_parse_wrapper, process_params
from utils_hw import spread_comp_quants, build_output

from modeling_hw import quant_reg_pred, calc_mf_opt

from modeling.segmentation_utils import find_seg_id
from modeling.modeling_main import apply_lme

set_option('display.width', 180)

try:
    wd = os_path.dirname(__file__)
except NameError:
    print('Working locally')
    wd = '/Users/jbrubaker/projects/copra_project/copra_hw'


DATA_PATH = os_path.join(wd, 'data')

# TODO - what was I going to do with these??
# NOTE: combrand, comgroup, commt were all removed in latest dataset
#       combrand -> lvl3, comgroup -> lvl2, commt -> lvl1
MODEL_COLS = ['comcostpofl', 'comloglistprice', 'com_contrib', 'upgmes', 'ln_ds', 'indirect', 'comdelgpricel4pofl',
              'endofqtr', 'crmindustryname', 'crmsectorname',
              'lvl4', 'lvl3', 'lvl2', 'lvl1',
              'ufc_incl', 'sbc_incl',  # 'sbc_nci', 'sbc_cwm', 'sbc_msp', 'sbc_mmp', 'sbc_p4v', 'sbc_sap_hana', 'sbc_maint',
              'discount']
COMP_META_COLS = ['quoteid', 'quoted_price', 'winloss', 'list_price', 'indirect']
# for parsing the VS model to dict - list out fixed effects categoricals and binaries
KNOWN_CATEGORICALS = ['lvl4', 'lvl3', 'lvl2', 'lvl1']
KNOWN_BINARIES = ['upgmes', 'indirect', 'endofqtr']

# define the config of each run
config = {'model':
              {'fe': 'discount ~ 1 + comcostpofl + comloglistprice + com_contrib + upgmes + '
                     'ln_ds + indirect + comdelgpricel4pofl + endofqtr + ufc_incl + sbc_incl'
               , 're': '(comcostpofl || lvl4/lvl3/lvl2/lvl1) + (comcostpofl || crmindustryname/crmsectorname)'
               }
            , 'wr_feats': parse_regmethod(regmethod)[1]
            , 'quant_feats': {'in_feats': ['ln_ds', 'ln_vs'], 'target': 'metric'}
            , 'misc': 'n/a'
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


def run_mf_calcs(train_op, pred_cols, adj_cols):
    train_op['cf'] = train_op['tmc']/train_op['value_score']

    train_op['norm_op'] = train_op.apply(
        lambda row: calc_quant_op(*row[adj_cols].values, cf=row['cf'], cv=0., fcw=0., fcl=0.), axis=1)

    train_op['wp_opt'] = train_op.apply(
        lambda row: calc_win_prob(row['norm_op'], *row[adj_cols].values), axis=1)

    train_op['wp_act'] = train_op.apply(
        lambda row: calc_win_prob(row['metric'], *row[adj_cols].values), axis=1)

    train_op['price_opt'] = train_op['norm_op'] * train_op['value_score']  # calc_quant_op predicts QP/VS
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


def run_quant_training(data, config, idx_cols=['quoteid'], grp_cols=[]):
    # prep dependent variable
    data['metric'] = (data['quoted_price'] / data['value_score']).replace([-inf, inf], 0.0)  # inf -> 0.0

    # run quantile approaches; quantile regression + raw calculated quantiles
    quants, qs, pred_cols, models = run_quants_offline(
        data.copy(), idx_cols, config['quant_feats']['in_feats'], out_feat='metric', grp_cols=grp_cols)

    # save as variable for easy access
    adj_cols = ['adj_' + x for x in pred_cols]

    # mix raw quantiles + quantile regressions
    quants = quants.apply(lambda row: bound_quant(row, qs), axis=1)[pred_cols]

    # run model factory L/M/H price adjustments
    quants[adj_cols] = quants.apply(lambda row: Series(price_adj(*row[pred_cols].values)), axis=1)

    # merge quantile calcs into training data
    train_op = merge(data, quants.reset_index(), on=idx_cols, how='left')

    # run OP, WP, CI calcs
    train_op = run_mf_calcs(train_op, pred_cols, adj_cols)
    
    return train_op, models, qs


def run_quant_testing(data, models, level='quote'):
    raw_qs = models['raw_qs']

    pred_mods = models['pred_qs']['models']
    in_feats = models['pred_qs']['in_feats']

    # calculate bounded quantiles for component- or quote-level data
    if level == 'quote':
        q_track = DataFrame()
        # known a priori
        q_list = raw_qs.columns
        pred_cols = ['pred_' + x for x in q_list]
        adj_cols = ['adj_' + x for x in pred_cols]
        # make joining in later easy
        data = data.set_index('quoteid')

        for s_id, grp in data.groupby('segment_id'):
            mod = pred_mods[s_id]
            qs = raw_qs.ix[s_id]

            q_res = quant_reg_pred(grp, in_feats, mod, q_list)
            # mix raw quantiles + quantile regressions
            quants = q_res[pred_cols].apply(lambda row: bound_quant(row, qs), axis=1)

            q_track = q_track.append(quants)

        # run model factory L/M/H price adjustments
        q_track[adj_cols] = q_track.apply(lambda row: Series(price_adj(*row[pred_cols].values)), axis=1)

        data = data.join(q_track, how='left').reset_index()
    else:
        mod = pred_mods['ALL']
        q_list = raw_qs.index

        pred_cols = ['pred_' + x for x in q_list]
        adj_cols = ['adj_' + x for x in pred_cols]

        q_res = quant_reg_pred(data, in_feats, mod, q_list)

        quants = q_res[pred_cols].apply(lambda row: bound_quant(row, raw_qs), axis=1)

        quants[adj_cols] = quants.apply(lambda row: Series(price_adj(*row[pred_cols].values)), axis=1)

        data = data.join(quants[adj_cols], how='left').reset_index(drop=True)

    data = data.apply(lambda row: calc_mf_opt(row), axis=1)

    return data


def run_pp_redist(comp_quant, opt_quant, source):
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
    cq, _ = spread_comp_quants(compq, quoteq, 'offline')

    # calculate outputs per online flow + spread optimal price
    q_out, tot_stats = build_output(cq, 'offline')

    if source == 'train':
        mape = mean(abs(q_out['bot_spread_OP'] - q_out['price_act'])/q_out['price_act'])
        print('MAPE of post-spread comp-level OP: {}'.format(mape))

    print('Redistribution of price points took {}'.format(dt.now() - t3))

    return q_out, tot_stats


for region in ['NA']:
    t = dt.now()
    print('Starting run on region {}...'.format(region))

    src_path = os_path.join(DATA_PATH, region)

    train_data = load_data(src_path, 'training')
    train_quarters = build_quarter_map(train_data)

    t1 = dt.now()

    train_pn = prep_comp(train_data)

    vs_mod = train_model(build_model(train_pn[MODEL_COLS], formula_fe=config['model']['fe'], formula_re=config['model']['re']))
    train_vs = run_model(vs_mod, train_pn, 'TRAINING')
    train_vs['value_score'] = (1. - train_vs['discount_pred']) * train_vs['list_price']
    train_vs['ln_vs'] = train_vs['value_score'].apply(lambda x: log10(x + 1))
    train_vs.to_csv(os_path.join(src_path, 'train_vs.csv'), index=False)

    print('Training LME model took {}'.format(dt.now() - t))

    params = process_params(vs_mod, train_pn, {}, KNOWN_CATEGORICALS, KNOWN_BINARIES)

    # for test purposes, reassign quoteid values to all BP bids
    # m = dict(zip(zip(train_vs['quoteid'], train_vs['componentid']), range(train_vs.shape[0])))
    # tmp = train_vs.copy()
    # tmp['qid'] = tmp['quoteid'].copy()
    # tmp['quoteid'] = tmp.apply(lambda row: m[(row['quoteid'], row['componentid'])] if (row['indirect'] == 1) else row['quoteid'], axis=1)

    t2 = dt.now()

    train_qt = prep_quote(train_vs)

    grp = train_qt.groupby('leading_brand')['quoteid'].count()
    filt_brand = grp.loc[grp.eq(1)].index
    if len(filt_brand) > 0:
        train_qt = train_qt.loc[~train_qt['leading_brand'].isin(filt_brand)].reset_index(drop=True)

    # Run quote-level segmentation
    train_qt, meta_qt, tier_map_qt = segment(train_qt, segmethod)
    train_qt = train_qt.reset_index(drop=True)
    fname = os_path.join(src_path, 'train_qt.csv')
    train_qt.to_csv(fname, index=False)

    # Convert quote-level segmentation to tree format
    seg_dict_qt = seg_to_dict(train_qt, meta_qt)
    train_qt['segment_id'] = train_qt.apply(lambda row: find_seg_id(row, seg_dict_qt), axis=1)

    # opt_wr = run_wr_training(train_qt)
    # wr_mape = mean(abs(opt_wr['price_opt'] - opt_wr['quoted_price'])/opt_wr['quoted_price'])
    # print('MAPE of WR OP: {}'.format(wr_mape))

    print('Starting quote-level quantile regression training...')
    opt_quant, pred_qs_qt, raw_qs_qt = run_quant_training(train_qt, config, ['quoteid'], grp_cols=['segment_id'])
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
    comp_quant, pred_qs_pn, raw_qs_pn = run_quant_training(train_vs, config, ['quoteid', 'componentid'], grp_cols=[])
    comp_mape = mean(abs(comp_quant['price_opt'] - comp_quant['price_act'])/comp_quant['price_act'])
    print('MAPE of pre-spread comp-level quant OP: {}'.format(comp_mape))

    print('Quantile training took {}'.format(dt.now() - t2))

    # tmp = merge(comp_quant[['quoteid', 'componentid', 'quoted_price']], train_vs[['quoteid', 'componentid', 'com_contrib']], on=['quoteid', 'componentid'], how='inner')
    # tmp = merge(tmp, opt_quant[['quoteid', 'price_opt']], on='quoteid', how='left')
    # tmp['op'] = tmp['price_opt'] * tmp['com_contrib']
    # tmp_mape = mean(abs(tmp['op'] - tmp['quoted_price']) / tmp['quoted_price'])
    # print('MAPE of comp-level quant OP: {}'.format(tmp_mape))

    c_quant = {'raw_qs': raw_qs_pn,
                  'pred_qs': {
                      'models': pred_qs_pn,
                      'in_feats': config['quant_feats']['in_feats'],
                      'target': config['quant_feats']['target']
                  }
                  }

    q_out, tot_stats = run_pp_redist(comp_quant, opt_quant, source='train')

    quant_tot = opt_quant.copy()
    quant_tot = post_process_op(quant_tot, train_quarters)

    # TODO - confirm proper output then update post_hoc_wrapper
    hash_folder = post_hoc_wrapper(train_vs, quant_tot, config, region, summary_dict, output_folder)
    #
    # quant_tot.to_csv(os_path.join(hash_folder, region, 'quant_dat_full.csv'), index=False)

    # outdf_qt.to_csv(os_path.join(hash_folder, region, 'outdf_qt.csv'), index=False)
    # train_op.to_csv(os_path.join(hash_folder, region, 'train_op.csv'), index=False)

    # prep model outputs
    model_dict_pn = component_parse_wrapper(params)

    quant_out = {'qt': q_quant,
                 'pn': c_quant}

    # TEMPORARY
    fpath = os_path.join(hash_folder, region)
    q_out.to_csv(os_path.join(fpath, 'quote_df.csv'))
    tot_stats.to_csv(os_path.join(fpath, 'total_deal_stats.csv'))

    write_models(model_dict_pn, seg_dict_qt, quant_out, hash_folder, region)

    print('Finished training on region {r}. Took {t}'.format(r=region, t=dt.now() - t))

    if RUN_TEST:
        print('Starting testing flow...')
        t_test = dt.now()
        test_data = load_data(src_path, 'testing')
        # test_quarters = build_quarter_map(test_data)
        test_pn = prep_comp(test_data)

        # TODO - build logic to allow run_model() to work with out of sample predictions
        # test_vs = run_model(vs_mod, test_pn, 'TESTING')

        print('Applying LME...')
        t_lme = dt.now()
        test_pn['discount_pred'] = test_pn.apply(lambda row: apply_lme(row, model_dict_pn), axis=1)
        print('Application of lme on {n} samples took {t}'.format(n=test_pn.shape[0], t=dt.now()-t_lme))

        test_pn['value_score'] = (1. - test_pn['discount_pred']) * test_pn['list_price']
        test_pn['ln_vs'] = test_pn['value_score'].apply(lambda x: log10(x + 1))

        test_pn.to_csv(os_path.join(src_path, 'test_vs.csv'), index=False)

        t_qt = dt.now()

        test_qt = prep_quote(test_pn)
        test_qt['segment_id'] = test_qt.apply(lambda row: find_seg_id(row, seg_dict_qt), axis=1)

        print('Applying quantile regression models...')
        test_quote = run_quant_testing(test_qt, q_quant, 'quote')
        test_comp = run_quant_testing(test_pn, c_quant, 'component')

        test_q_out, test_tot_stats = run_pp_redist(test_comp, test_quote, source='test')

        print('Quote-level testing calcs took {}'.format(dt.now() - t_qt))

        print('Entire testing flow took {}'.format(dt.now() - t_test))

        test_q_out.to_csv(os_path.join(fpath, 'quote_df_test.csv'))
        test_tot_stats.to_csv(os_path.join(fpath, 'total_deal_stats_test.csv'))

# build_summary(output_folder)
