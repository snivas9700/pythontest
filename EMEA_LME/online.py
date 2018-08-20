from numpy import log10
from pandas import read_pickle, read_excel, read_csv, set_option
import pandas as pd

import json
from os import path as os_path
from os.path import join as os_join, dirname, abspath
from pickle import loads as pkl_loads
from zlib import decompress

from EMEA_LME.spreading_offline import spread_tss, spread_hw
from EMEA_LME.data_prep_online import data_prep_hw_com, data_prep_tss_com, prep_output,prep_output_nonTSS, init_prep, final_prep, swma_prep
from EMEA_LME.modeling_hw import run_quants_online
from EMEA_LME.load_data import load_model_dict
from EMEA_LME.modeling_main import apply_lme
from EMEA_LME.wr_predict import wr_train_hw, wr_train_tss, pti
from EMEA_LME.model_def import config_wr_hw, config_wr_tss, FIELD_MAP
from EMEA_LME.utils import apply_bounds_hw, apply_bounds_tss
from EMEA_LME.segmentation_utils import find_seg_id

from config import final_online_TSS

pd.options.display.max_rows = 4000
set_option('display.width', 180)

import builtins as _b_
from EMEA_LME.scribe import Scribe
_b_.SCRIBE = Scribe('trace')

# scale on L/M/H price points
UPLIFT_TSS = 1.0; UPLIFT_HW = 1.4
# scale on lower bound
ALPHA = 0.0


def process_quote(df):

    # Step A: breakdown incoming quote dataframe into hw and tss parts
    #df = read_csv('./input/before_online_4163203.csv')

    df.reset_index(inplace=True)
    df = df.copy().fillna('')
    df_ = init_prep(df_in=df.copy())

    df_ = df_.rename(columns=FIELD_MAP['input_trans'])
    hw_columns = FIELD_MAP['hw_cols']

    if df_['TSSComponentincluded'][0] == 'N':
        com_df_tss = pd.DataFrame(columns=['index'])
        com_df_tss_nonswma = pd.DataFrame(columns=['index'])
                
    elif df_['TSSComponentincluded'][0] == 'Y':
        tss_columns = FIELD_MAP['tss_cols']
        ra_per = 0.3
        df_["PTI_5price"] = (1-ra_per)/(0.95-ra_per)*df_['PTI_0price']

        df_tss = df_[tss_columns].copy()
        com_df_tss = data_prep_tss_com(df_tss)

        com_df_nonswma = com_df_tss[com_df_tss['servoffcode'].isin(["HW-MAINTENANCE","WSU","TMS","HDR"])]
        com_df_swma = com_df_tss[com_df_tss['servoffcode'].isin(["HW-MAINTENANCE", "WSU", "TMS", "HDR"])==False]

        # SWMA Case Analysis
        com_df_tss_swma = swma_prep(com_df_swma.copy())

        # Non-SWMA Case Analysis
        tss_lme = load_model_dict('./EMEA_LME/output/tssocomp_model.json')
        tss_quant = read_pickle('./EMEA_LME/output/tssocomp_quant.pkl')
    
        # apply LME to get discount, then project back to value_score space
        com_df_nonswma['tss_bundled1.0'] = 1.0
        com_df_nonswma['y_pred'] = com_df_nonswma.apply(lambda row: apply_lme(row, tss_lme), axis=1)
        com_df_nonswma['tss_value_score'] = com_df_nonswma['y_pred'] * com_df_nonswma['p_list_hwma']
        com_df_nonswma['tss_ln_vs'] = com_df_nonswma['tss_value_score'].apply(log10)
    
        # run component-level quantile regression
        com_df_nonswma = com_df_nonswma.apply(lambda row: run_quants_online(row, tss_quant, seg_field='taxon_hw_level_3', uplift=UPLIFT_TSS, alpha=ALPHA), axis=1)

        # winrate calculation
        com_df_nonswma.reset_index(inplace=True)
        com_df_nonswma_wr = wr_train_tss(data=com_df_nonswma.copy(), config=config_wr_tss, uplift=UPLIFT_TSS)
        com_df_nonswma_wr['contract_number'] = 1

        # spreading
        q_tss, com_df_tss_nonswma = spread_tss(com_tss=com_df_nonswma_wr.copy(), TSS_UPLIFT=UPLIFT_TSS, spread_opt=True, tss_cost='GP')

    # HW
    df_hw = df_[hw_columns].copy()
    com_df_hw = data_prep_hw_com(df_hw)
    
    hw_lme = load_model_dict('./EMEA_LME/output/hwcomp_model.json')
    hw_quant = read_pickle('./EMEA_LME/output/hwcomp_quant.pkl')
    
    com_df_hw['y_pred'] = com_df_hw.apply(lambda row: apply_lme(row, hw_lme), axis=1)
    com_df_hw['hw_value_score'] = com_df_hw['y_pred'] * com_df_hw['p_list_hw']
    com_df_hw['hw_ln_vs'] = com_df_hw['hw_value_score'].apply(log10)

    com_df_hw = com_df_hw.apply(lambda row: run_quants_online(row, hw_quant, seg_field='taxon_hw_level_3', uplift=UPLIFT_HW, alpha=ALPHA), axis=1)

    com_df_hw.reset_index(inplace=True)
    com_df_hw_wr = wr_train_hw(data=com_df_hw, config=config_wr_hw, uplift=UPLIFT_HW)

    com_df_hw_wr['quoteid'] = com_df_hw_wr['QuoteID']

    q_hw, com_df_hw = spread_hw(com_hw=com_df_hw_wr.copy(), HW_UPLIFT=UPLIFT_HW, spread_opt=True)

    fpath = './EMEA_LME/output'
    com_df_hw.to_csv(os_path.join(fpath, 'hw_online.csv'), index=False)
    com_df_tss_nonswma.to_csv(os_path.join(fpath, 'tss_online.csv'), index=False)

    if ~('ci_low' in com_df_hw.columns):
        com_df_hw['ci_low_hw'] = 0.9
    if ~('ci_low' in com_df_tss_nonswma.columns):
        com_df_tss_nonswma['ci_low_tss'] = 0.9
    if ~('ci_high' in com_df_hw.columns):
        com_df_hw['ci_high_hw'] = 1.1
    if ~('ci_high' in com_df_tss_nonswma.columns):
        com_df_tss_nonswma['ci_high_tss'] = 1.1

    com_df_hw.rename(columns={'level_0': 'index'}, inplace=True)
    com_df_tss_nonswma.rename(columns={'level_0': 'index'}, inplace=True)
    quote_out_ = com_df_hw.merge(com_df_tss_nonswma, on='index', how='left')

    quote_out_.to_csv(os_path.join(fpath, 'quote_out_tss.csv'), index=False)
    quote_out = quote_out_.merge(df, on='index', how='left')
    quote_out.to_csv(os_path.join(fpath, 'quote_out_tss_hw.csv'), index=False)    
    
    dup_map = {'DealSize_x': 'DealSize'
               , 'committedcharge_x': 'committedcharge'
               , 'servoffcode_x': 'servoffcode'
               , 'pred_L_x': 'pred_L_hw'
               , 'pred_M_x': 'pred_M_hw'
               , 'pred_H_x': 'pred_H_hw'
               , 'pred_L_y': 'pred_L_tss'
               , 'pred_M_y': 'pred_M_tss'
               , 'pred_H_y': 'pred_H_tss'
                      
               , 'pred_L': 'pred_L_hw'
               , 'pred_M': 'pred_M_hw'
               , 'pred_H': 'pred_H_hw'
               , 'Componentid_x':'componentid'
               
               ,'QuoteID_x' : 'QuoteID'
               ,'TSSComponentincluded_x' : 'TSSComponentincluded'
               ,'Indirect(1/0)_x' : 'Indirect(1/0)'
               ,'CustomerSecName_x' : 'CustomerSecName'
               ,'CustomerIndustryName_x' : 'CustomerIndustryName'

               }

    quote_out = quote_out.rename(columns=dup_map)
    quote_out = quote_out.rename(columns={v: k for k, v in FIELD_MAP['input_trans'].items()})
    quote_out = quote_out.loc[:, ~quote_out.columns.duplicated()]
    
    if quote_out['TSSComponentincluded'][0] == 'N':
        quote_out = prep_output_nonTSS(quote_out.copy())
    elif quote_out['TSSComponentincluded'][0] == 'Y':
        quote_out = prep_output(df=quote_out.copy())
        quote_out = quote_out[final_online_TSS]

        # Integrate SWMA Quotes
        com_df_tss_swma = com_df_tss_swma.rename(columns={v: k for k, v in FIELD_MAP['input_trans'].items()})
        quote_out.update(com_df_tss_swma)

    quote_out = final_prep(df_out=quote_out.copy(), df_in=df)

    quote_out.to_csv(os_path.join(fpath, 'quote_out_tss_hw_final.csv'), index=True)

    #print(quote_out[['TSS_AdjComLowPrice', 'TSS_OptimalPriceIntervalLow', 'TSS_OptimalPrice',
    #                 'TSS_OptimalPriceIntervalHigh', 'TSS_AdjComHighPrice']])

    return quote_out
