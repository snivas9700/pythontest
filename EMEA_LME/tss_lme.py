from __future__ import division

# from collections import OrderedDict
from pandas import set_option, read_csv, read_excel, ExcelWriter, merge, Series, concat
from numpy import absolute, mean, log10, arange, int32, inf
from vs_predict import value_score_predict
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from method_defs import segmethod, regmethod
from modeling_utils import price_adj, calc_quant_op, calc_win_prob, opt_price_ci, bound_quant, bound_op
from modeling_hw import quant_reg_pred, calc_mf_opt, build_model, train_model, run_model, run_quants_offline
from utils import parse_regmethod, load_data, build_quarter_map


# set the pandas output width (i.e. print of df.head() ) to 180 characters
set_option('display.width', 180)

# cost factor table
df_cf = read_excel('Cost_Factors_20180307.xlsx', sheetname='Cost Factors', header=2)
# rename long column title
df_cf.rename(columns={'EUROPE & MEA IOTs\nCOST FACTORS\n(not for use in the\n"Cross-Brand Pricing Tool")':'CostFactor'}, inplace=True)
df_cf = df_cf.loc[:,['Type', 'Model', 'TypeModel', 'Brand', 'CostFactor']]
# Clean up data
df_cf.Type = df_cf.Type.astype('str').str.strip()
df_cf.Model = df_cf.Model.astype('str').str.strip()
df_cf.TypeModel = df_cf.TypeModel.astype('str').str.strip()
df_cf.Brand = df_cf.Brand.astype('str').str.strip()

df_cf.Type = df_cf.Type.str.lower()
df_cf.Model = df_cf.Model.str.lower()
df_cf.TypeModel = df_cf.TypeModel.str.lower()

# hw/tss data
dat = read_csv('hwma_hw.csv')
dat0 = dat.copy()

dat = dat.reset_index()

dat.machine_type = dat.machine_type.astype('str').str.strip()
dat.machine_model = dat.machine_model.astype('str').str.strip()
dat.machine_type = dat.machine_type.str.lower()
dat.machine_model = dat.machine_model.str.lower()

dat_mtm = dat[['index', 'machine_type', 'machine_model']].copy()
df_specific = merge(dat_mtm, df_cf, left_on=['machine_model', 'machine_type'], right_on=['Model', 'Type'], how='left')
df_cf_broad = df_cf[df_cf['Model'] == "all"]
df_broad = merge(df_specific, df_cf_broad, left_on=['machine_type'], right_on=['Type'], how='left')

mask = (df_broad['CostFactor_x'].notnull())
df_broad.loc[mask, 'costfactor'] = df_broad.loc[mask, 'CostFactor_x']
mask = (df_broad['CostFactor_y'].notnull())
df_broad.loc[mask, 'costfactor'] = df_broad.loc[mask, 'CostFactor_y']

df_broad = df_broad[['index', 'costfactor']]
dat = merge(dat, df_broad, on=['index'], how='left')

dat['RA_percentage'] = .3
dat['PTI_0price'] = dat['p_list_hwma'] * (dat['costfactor'] / (1-dat['RA_percentage']))
dat['PTI_5price'] = dat['p_list_hwma'] * (dat['costfactor'] / (1-dat['RA_percentage']-0.05))
# GP = QP - LP * CF - QP * RA
dat['PTI_actual'] = 1 - ((dat['p_list_hwma'] * dat['costfactor']) / dat['p_bid']) - dat['RA_percentage']

dat = dat[(dat['com_category']=='s')]

dat['comp_duration_days'] = int32(dat['comp_duration_days'])
dat['coverage_days'] = int32(dat['coverage_days'])
dat['coverage_hours'] = int32(dat['coverage_hours'])
dat['weekly_hours'] = dat['coverage_days'] * dat['coverage_hours']

dat['lnqp'] = log10(dat['p_bid']+1)
dat['lnlp'] = log10(dat['p_list']+1)
dat['lnhwlp'] = log10(dat['p_list_hw']+1)
dat['lnds'] = log10(dat['p_list_hwma_hw']+1)

dat['committed'] = int32(dat['committed'])
dat['cmda'] = int32(dat['cmda'])
# dat['sale_hwma_attached'] = int32(dat['sale_hwma_attached'])
# dat['sale_hwma_aftermarket'] = int32(dat['sale_hwma_aftermarket'])

dat['level_3'] = dat['level_3'].fillna('missing')
dat['level_2'] = dat['level_2'].fillna('missing')
# dat['brandname'] = dat['brandname'].fillna('missing')
# dat['mcc'] = dat['mcc'].fillna('missing')
# dat['product_category'] = dat['product_category'].fillna('missing')
dat['mtm_id'] = dat['MTM_identifier'].astype('str')

# define the config of each run
# target variable: p_pct_list, lnqp, p_bid
config = {'model':
              {'formula_fe': 'p_pct_list ~ lnlp'
                             '+ lnhwlp + lnds + p_uplift_comm'
                             '+ weekly_hours + response_time_lvl1 + response_time_lvl2 + response_type'
                             # # '+ sl_code'
                             '+ comp_duration_days + committed'
                             '+ level_2'
                             '+ sale_hwma_attached + sale_hwma_aftermarket + auto_renewal_flg'
                             '+ chnl'
                             '+ offer_nm'
                             '+ reg_cd + country'
                             '+ tssincluded + sector + pymnt_opt + tss_type'
               , 'formula_re': '(lnlp||platform/level_3/mtm_id)'
                               # '(lnlp||platform/level_3/mtm_id)'
                               # '+ (lnlp||reg_cd/country)'
                               # '+ (p_list||offer_nm)'
                               # '+ (p_list||platform/level_3/level_2/mtm_id)'
               }
          }

train_pn = dat.copy()
# train_pn=train_pn[:100]
train_pn = train_pn.sample(frac=1.0)

cols = ['platform', 'level_3', 'level_2', 'level_1', 'mtm_id', 'prod_div', 'product_category',
        'reg_cd', 'country', 'attach_rate_hwma',
        'coverage_days', 'coverage_hours', 'weekly_hours',
        'response_time_lvl1', 'response_time_lvl2', 'response_type', 'sl_code', 'committed',
        'comp_duration_days', 'sale_hwma_attached', 'sale_hwma_aftermarket', 'auto_renewal_flg',
        'chnl', 'ff_chnl_sdesc', 'channel_id',
        'offer_nm', 'offering_short_desc',
        'cmda', 'com_category', 'mcc', 'sector_sdesc', 'pymnt_opt', 'srv_dlvr_meth',
        'p_uplift_comm', 'p_list_hw', 'p_list_hwma_hw',
        'lnds', 'lnhwlp', 'lnlp', 'lnqp', 'p_pct_list', 'discount', 'p_list', 'p_bid']
train_pn = train_pn[cols]

train_vs, params, r_model = value_score_predict(data=train_pn, config=config)

train_vs['value_score'] = train_vs['y_pred'] * train_vs['p_list']
# train_vs['value_score'] = 10**train_vs['y_pred']
train_vs['ape'] = (train_vs['p_bid'] - train_vs['value_score']) / train_vs['value_score']
print(mean(absolute(train_vs['ape'])))
#
# test_pn = train_pn.sample(frac=0.2)
# test_vs, params, r_model = value_score_predict(data=test_pn, config=config, mod1=r_model)
# test_vs['value_score'] = test_vs['y_pred'] * train_vs['p_list']
# # test_vs['value_score'] = 10**test_vs['y_pred']
# test_vs['ape'] = (test_vs['p_bid'] - test_vs['value_score']) / test_vs['value_score']
# print(mean(absolute(test_vs['ape'])))

# train_vs['ape'] = (train_vs['p_list'] - mean(train_vs['p_list'])) / mean(train_vs['p_list'])
# print(mean(absolute(train_vs['ape'])))

# ape = train_vs[(train_vs['ape'] > -1) & (train_vs['ape'] < 1)]['ape']
# # histogram of ape
# lb = -1
# hb = 1.0
# step = 0.05
# counts, bins, bars = plt.hist(ape, bins= arange(lb,hb,step))
# print(mean(absolute(ape)))

train_vs.to_csv('in_sample.csv')

# save lme parameters to file
e = ExcelWriter('lme_params.xls')

params['fixed_effects'].to_frame().to_excel(e, sheet_name='fixed')

for hier in params['mixed_effects'].keys():
    ks = params['mixed_effects'][hier].keys()
    key = sorted(ks, key=len, reverse=True)[0]
    key_out = key.replace(':', '_').replace('(', '').replace(')', '')
    key_out = key_out[0:5]
    params['mixed_effects'][hier][key].to_excel(e, sheet_name=key_out)

e.save()

## ols baseline
# md = smf.ols("lncomqp ~ 1 + lncomlp", train_pn).fit()
# ypred = md.predict()
# vspred = 10**ypred
# train_pn['apeols'] = (vspred-train_pn['comp_real_value'])/train_pn['comp_real_value']
# print(md.summary())
# print(mean(absolute(apeols)))


train_vs = read_csv('in_sample_MAPE20.csv')
train_vs.reset_index(inplace=True)
UPLIFT = 1.0
# scale on lower bound, where lower bound = ALPHA*VS
ALPHA = 0.0

config_wr = {'wr_feats': parse_regmethod(regmethod)[1]
            , 'quant_feats': {'in_feats': ['lnlp', 'lnhwlp', 'lnds'], 'target': 'metric'}
            , 'uplift': UPLIFT
            , 'alpha': ALPHA
            , 'misc': 'includes component segmentation'
          }

def run_mf_calcs(train_op, pred_cols, adj_cols, alpha):

    train_op['norm_op'] = train_op.apply(
        lambda row: calc_quant_op(*row[adj_cols].values, cf=0., cv=0., fcw=0., fcl=0.), axis=1)

    train_op['price_opt'] = train_op['norm_op'] * train_op['value_score']  # calc_quant_op predicts QP/VS

    # train_op = bound_op(train_op, alpha)

    train_op['wp_opt'] = train_op.apply(
        lambda row: calc_win_prob(row['norm_op'], *row[adj_cols].values), axis=1)

    train_op['wp_act'] = train_op.apply(
        lambda row: calc_win_prob(row['metric'], *row[adj_cols].values), axis=1)

    # train_op['price_act'] = train_op['quoted_price'].copy()

    # train_op[['ci_low', 'ci_high']] = train_op.apply(lambda row: Series(
    #     opt_price_ci(row['norm_op'], *row[adj_cols].values, Cf=0., Cv=0, FCw=0, FCl=0, Pct=.95)),
    #     axis=1)

    op_cols = []
    for col in pred_cols:
        c = col.split('pred_')[-1]
        c = 'price_' + c
        train_op[c] = train_op[col] * train_op['value_score']
        op_cols.append(c)  # for easy access later

    return train_op

def run_quant_training(data, config, idx_cols=['index'], grp_cols=[], uplift=1., alpha=0.0):
    # prep dependent variable
    data['metric'] = (data['p_bid'] / data['value_score']).replace([-inf, inf], 0.0)  # inf -> 0.0

    # run quantile approaches; quantile regression + raw calculated quantiles
    quants, qs, pred_cols, models = run_quants_offline(
        data.copy(), idx_cols, config['quant_feats']['in_feats'], out_feat='metric', grp_cols=grp_cols)

    # save as variable for easy access
    adj_cols = [x for x in pred_cols]
    #
    # # mix raw quantiles + quantile regressions
    # quants = quants.apply(lambda row: bound_quant(row, qs, grp_cols), axis=1)[pred_cols]

    # run model factory L/M/H price adjustments
    # quants[adj_cols] = quants.apply(lambda row: Series(price_adj(*row[pred_cols].values)), axis=1) * uplift

    # merge quantile calcs into training data
    # train_op = merge(data, quants.reset_index(), on=idx_cols, how='left')

    # run OP, WP, CI calcs
    train_op = run_mf_calcs(quants, pred_cols, adj_cols, alpha)

    return train_op, models, qs

comp_quant, pred_qs_pn, raw_qs_pn = run_quant_training(train_vs, config_wr, ['index'], grp_cols=['platform'], uplift=UPLIFT, alpha=ALPHA)

comp_quant.to_csv('opt_insample.csv')