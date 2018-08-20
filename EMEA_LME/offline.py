from pandas import read_csv, DataFrame, Series, set_option, merge, read_excel, \
    to_numeric, to_pickle, read_pickle, to_datetime, concat
from numpy import inf, intersect1d, mean, log10, isfinite, nan
import pickle
from datetime import datetime as dt
import os, sys
from rpy2.robjects import r

from modeling_hw import run_quants_online
from spreading_offline import spread_tss, spread_hw
from vs_predict import lme_train, lme_predict, process_params
from wr_predict import quantile_train, wr_train_hw, wr_train_tss, pti, pti_tsso
from model_def import config_lme_hw, config_lme_tss, config_lme_tsso, \
    config_wr_hw, config_wr_tss, config_wr_tsso
from utils import apply_bounds_hw, apply_bounds_tss
from conversions import component_parse_wrapper, process_params
from io_utils import write_models
from data_prep_offline import log_transform, prep_hw, prep_tss, map_svl,\
    map_cmda, map_hw_taxon, merge_tss_hw, write_fields, gtms_prep, mss_prep

sys_path_root = os.getcwd()
sys.path.append(sys_path_root)

path_root = sys_path_root
src_path = os.path.join(sys_path_root,"data")
if not os.path.exists(src_path):
    os.makedirs(src_path)
path_out = os.path.join(sys_path_root,"output", "") 
if not os.path.exists(path_out):
    os.makedirs(path_out)
set_option('display.width', 180)

r('memory.limit(size=6000)');


def main(prep=False):

    t0 = dt.now()
    print('load hw and tss data')

    if prep:

        # Load HWMA data
        hwma, process_log_tss = prep_tss(src_path, path_out, label_='INV_EP_CONTRACT', time_sample=False)

        process_log_tss.to_excel(path_out + 'prep_log_hwma.xlsx', index=False)

        file_hw_taxon = 'EACM_product_map.xlsx'

        svl_attribs = ['SLC', 'COV VALUE', 'CNT VALUE', 'CNT UNIT', 'ORT VALUE', 'ORT UNIT',
                       'PAT VALUE', 'PAT UNIT', 'FXT VALUE', 'FXT UNIT', 'TAT VALUE', 'TAT UNIT']

        # Add TSS service level attributes to hardware maintenance data;
        # attributes are stored in a separate file; merge key is sl_code

        hwma_svl, _ = map_svl(hwma, 'tss_service_levels.xlsx',
                              fields_tss=svl_attribs,
                              keys_tss=['sl_code'],
                              path_in=path_root + '/data/',
                              path_out=path_out)

        # Add IBM Systems product taxonomy to hardware maintenance data; merge key is MTM

        hwma_svl_hw_taxon, _ = map_hw_taxon(hwma_svl, file_hw_taxon,
                                            fields_hw=['mtm', 'level_2', 'level_3','level_4'],
                                            keys=['mtm'],
                                            path_in=path_root + '/data/',
                                            path_out=path_out)

        # Some post-processing on HWMA
        hwma_svl_hw_taxon.sort_index(axis=1, inplace=True)
        hwma_svl_hw_taxon['date_inst'] = to_datetime(hwma_svl_hw_taxon['date_inst']).map(lambda x: x.year)
        hwma_svl_hw_taxon = hwma_svl_hw_taxon.loc[hwma_svl_hw_taxon['date_inst'].isin([2016, 2017, 2018])]
        hwma_svl_hw_taxon = hwma_svl_hw_taxon.loc[hwma_svl_hw_taxon['p_pct_list_hwma'] > 0.1]

        hwma_svl_hw_taxon.to_csv(path_out + 'hwma_2016_2018.csv', index=False)

        # Load GTMS data
        file_gtms = 'GTMS_Contract_v20180306.csv'
        file_mtm_map = 'EACM_product_map_automated_3_20_2018.csv'

        gtms = gtms_prep(file_gtms, file_mtm_map, src_path)

        # Add TSS service level attributes to GTMS data
        gtms_svl, _ = map_svl(gtms, 'tss_service_levels.xlsx',
                              fields_tss=svl_attribs,
                              keys_tss=['sl_code'],
                              path_in=path_root + '/data/',
                              path_out=path_out)

        gtms_svl[['sl_contact_hours', 'sl_onsite_hours', 'sl_part_hours']] = 55 * 24.0

        # Load MMS Data
        file_mss = 'GHDR_CONTRACTS_20180319.csv'
        mss = mss_prep(file_mss, file_mtm_map, src_path)
        mss_svl, _ = map_svl(mss, 'tss_service_levels.xlsx',
                              fields_tss=svl_attribs,
                              keys_tss=['sl_code'],
                              path_in=path_root + '/data/',
                              path_out=path_out)

        mss_svl[['sl_contact_hours', 'sl_onsite_hours', 'sl_part_hours']] = 55 * 24.0
        mss_svl['sl_hours_x_days'] = 55.0

        # Join three groups of TSS data and save them

        hwma_gtms_mss = concat([hwma_svl_hw_taxon, gtms_svl, mss_svl])
        hwma_gtms_mss.to_csv(path_out + 'tss_2016_2018.csv', index=False)

        ########################################################################################
        # Call function to load & process the hardware data

        hw, process_log_hw, _ = prep_hw(path_in=src_path, path_out=path_out,
                                        label_='TSS_trainingset',
                                        remove_losses=False, time_sample=False)

        # This is to remove the quotes that are not approved, or waiting for approval
        # hw = hw.loc[hw['approval_status'].isin(['approved', 'does not need approval',
        #                                        'approval expired', 'copra priced', 'approved/upc'])]

        # Add indicator of CMDA sales & pricing; indicator stored in a separate file; merge key is quote_id
        hw = map_cmda(hw, src_path, 'hw_quote_cmda')

        # Add IBM Systems product taxonomy to hardware data
        file_hw_taxon = 'EACM_product_map.xlsx'
        hw_taxon, _ = map_hw_taxon(hw, file_hw_taxon,
                                   fields_hw=['mtm', 'level_2', 'level_3', 'level_4'],
                                   keys=['mtm'], path_in=path_root + '/data/', path_out=path_out)

        # Post Processing and Save HW
        hw_taxon.to_csv(path_out + 'hw.csv', index=False)
        hw_taxon = hw_taxon[hw_taxon.region != 'middle east and africa']
        hw_taxon = hw_taxon[(hw_taxon.date_approval_year <= 2018) & (hw_taxon.date_approval_year >= 2016)]
        hw_taxon.to_csv(path_out + 'hw_2016_2018.csv', index=False)

        write_fields(hw_taxon, 'hw', path_out)

        # Merge TSS & hardware data sets
        hwma_hw, process_log_tss_hw, cntry_match = merge_tss_hw(hwma_svl_hw_taxon, hw, path_out)

        # Transform feature(s) & place in a separate column (leaving original intact)
        hwma_hw = log_transform(hwma_hw,['p_list_hwma', 'p_list_hw','p_list_hwma_hw'])

        # Reorder columns alphabetically
        hwma_hw.sort_index(axis=1, inplace=True)

        # Save the merged data set, record a list of fields therein
        hwma_hw.to_csv(path_out + 'hwma_hw.csv', index=False)
        process_log_tss_hw.to_excel(path_out + 'process_log_hwma_hw.xlsx',index=False)

        # Refresh list of annotated training data fields
        complete_list = read_csv(path_out + 'hwma_hw_cols.txt', header=None)
        complete_list.columns = ['Field']
        annotated_list = read_excel(path_out + 'hwma_hw_fields_explained.xlsx')

        new_list = merge(complete_list, annotated_list, on='Field',how='left')
        new_list.to_excel(path_out + 'hwma_hw_fields_explained_.xlsx',index=False)

        # END TRAINING DATA PREPARATION
        train_hw = hw_taxon.copy()
        train_tsso = hwma_gtms_mss.copy()

    else:
        train_hw = read_csv('./output/hw_2016_2018.csv',low_memory=False)
        train_tsso = read_csv('./output/tss_2016_2018.csv',low_memory=False)


    t1 = dt.now()
    print('lme training')

    ##############################################################################

    ## TSS cost and PTI
    df_cf = read_excel('./input/Cost_Factors_20180307.xlsx', sheet_name='Cost Factors', header=2)
    train_tsso = pti_tsso(train_tsso.copy(), df_cf)
    train_tsso['const'] = 1

    ## HW data engineering
    train_hw['p_pct_list_hw'] = to_numeric(train_hw['p_pct_list_hw'], errors='coerce')
    train_hw = train_hw[isfinite(train_hw['p_pct_list_hw'])]  # 44758, 14%
    train_hw = train_hw.loc[train_hw['p_pct_list_hw'] > .00001]  # 44483, 14%
    train_hw['const'] = 1
    train_hw['p_list_hw_log'] = log10(train_hw['p_list_hw']+2)
    train_hw['p_list_hw_total_log'] = log10(train_hw['p_list_hw_total'] + 2.0)

    train_tsso['p_list_hwma_log'] = log10(train_tsso['p_list_hwma'] + 2.0)

    cols_hw = ['p_pct_list_hw','p_list_hw_log','cost_pct_list_hw','chnl_ep','bundled','date_approval_eoq',
           'upgrade_mes','p_bid_hw_contrib','p_delgl4_hw','p_list_hw_total_log','won','taxon_hw_level_4',
           'taxon_hw_level_3','taxon_hw_level_2','taxon_hw_mtm','sector','industry','country']

    cols_tsso = ['p_pct_list_hwma', 'p_list_hwma_log', 'committed', 'sl_contact_hours', 'sl_onsite_hours',
             'sl_part_hours', 'sl_hours_x_days','tss_bundled', 'tss_type', 'chnl_tss', 'taxon_hw_level_4',
             'taxon_hw_level_3', 'taxon_hw_level_2', 'mtm', 'market', 'country']

    train_hw[cols_hw] = train_hw[cols_hw].replace([-inf, inf], nan)
    # train_tss[cols_tss] = train_tss[cols_tss].replace([-inf, inf], nan)
    train_tsso[cols_tsso] = train_tsso[cols_tsso].replace([-inf, inf], nan)

    train_hw = train_hw.dropna(axis=0, how='any', subset=cols_hw)
    # train_tss = train_tss.dropna(axis=0, how='any', subset=cols_tss)
    train_tsso = train_tsso.dropna(axis=0, how='any', subset=cols_tsso)

    t1 = dt.now()
    print('lme training')

    ## LME train
    model_lme_hw = lme_train(train_data=train_hw.copy(), config=config_lme_hw)
    # model_lme_tss = lme_train(train_data=train_tss.copy(), config=config_lme_tss)
    model_lme_tsso = lme_train(train_data=train_tsso.copy(), config=config_lme_tsso)

    ## Saving results
    with open('./output/lme_hw.pkl', 'wb') as f:
        pickle.dump(model_lme_hw, f)
    # with open('./output/lme_tss.pkl', 'wb') as f:
    #    pickle.dump(model_lme_tss, f)
    with open('./output/lme_tsso.pkl', 'wb') as f:
        pickle.dump(model_lme_tsso, f)

    ##  Build LME model dict
    KNOWN_CATEGORICALS_HW = ['chnl_ep']
    KNOWN_BINARIES_HW = []
    params_lme_hw = process_params(model_lme_hw, train_hw, {}, KNOWN_CATEGORICALS_HW, KNOWN_BINARIES_HW)
    model_dict_hw = component_parse_wrapper(params_lme_hw)

    KNOWN_CATEGORICALS_TSSO = ['tss_type', 'chnl_tss']
    KNOWN_BINARIES_TSSO = []
    params_lme_tsso = process_params(model_lme_tsso, train_tsso, {}, KNOWN_CATEGORICALS_TSSO, KNOWN_BINARIES_TSSO)
    model_dict_tsso = component_parse_wrapper(params_lme_tsso)

    ## value score prediction
    train_hw_vs = lme_predict(train_hw.copy(), mod=model_lme_hw, ind='hw')
    # train_tss_vs = lme_predict(train_tss.copy(), mod=model_lme_tss, ind='tss')
    train_tsso_vs = lme_predict(train_tsso.copy(), mod=model_lme_tsso, ind='tss')

    ## Run quantile regression
    train_hw_vs = train_hw_vs.copy().reset_index(drop=True).reset_index()
    # train_tss_vs = train_tss_vs.copy().reset_index(drop=True).reset_index()
    train_tsso_vs = train_tsso_vs.copy().reset_index(drop=True).reset_index()

    train_quant_hw_won = train_hw_vs[train_hw_vs.won == 1]

    train_quant_hw, model_quant_hw, qs_hw = quantile_train(data=train_quant_hw_won.copy(), config=config_wr_hw)
    # train_quant_tss, model_quant_tss, qs_tss = quantile_train(data=train_tss_vs.copy(), config=config_wr_tss)
    train_quant_tsso, model_quant_tsso, qs_tsso = quantile_train(data=train_tsso_vs.copy(), config=config_wr_tsso)

    ## Build quantile regression model output dict
    hw_quant = {'raw_qs': qs_hw,
               'pred_qs': {
                   'models': model_quant_hw,
                   'in_feats': config_wr_hw['quant_feats']['in_feats'],
                   'target': config_wr_hw['quant_feats']['target']
                   }
               }

    tsso_quant = {'raw_qs': qs_tsso,
                 'pred_qs': {
                     'models': model_quant_tsso,
                     'in_feats': config_wr_tsso['quant_feats']['in_feats'],
                     'target': config_wr_tsso['quant_feats']['target']
                 }
                 }

    ## Calculate win rate curve and optimal price
    UPLIFT_HW = 1.04; UPLIFT_TSSO = 1.0

    # train_quant_hw = train_hw_vs.apply(lambda row: run_quants_online(row, hw_quant, seg_field='taxon_hw_level_3', uplift=UPLIFT_HW, alpha=0), axis=1) ## CW

    wr_fit_hw = wr_train_hw(data=train_quant_hw.copy(), config=config_wr_hw, uplift=UPLIFT_HW)
    # wr_fit_tss = wr_train_tss(data=train_quant_tss.copy(), config=config_wr_tss, uplift=UPLIFT_TSS)
    wr_fit_tsso = wr_train_tss(data=train_quant_tsso.copy(), config=config_wr_tsso, uplift=UPLIFT_TSSO)

    # Apply bounds
    '''
    op_hw_bound = apply_bounds_hw(wr_fit_hw.copy())
    # op_tss_bound = apply_bounds_tss(wr_fit_tss.copy())
    op_tsso_bound = apply_bounds_tss(wr_fit_tsso.copy())

    # write in-sample prediction results
    fpath = './output'

    
    op_hw_bound.to_csv(os_path.join(fpath, 'HW_output_2016_2018_wl.csv'), index=False)
    # op_tss_bound.to_csv(os_path.join(fpath, 'HWMA_HW_output.csv'), index=False)
    op_tsso_bound.to_csv(os_path.join(fpath, 'HWMA_output_2016_2018_uplift05.csv'), index=False)
    '''

    # Quote_level optimization and bottom-line spreading
    tss_cost = 'GP'
    q_hw, com_hw = spread_hw(com_hw=wr_fit_hw.copy(), HW_UPLIFT=UPLIFT_HW, spread_opt=True)
    q_tsso, com_tsso = spread_tss(com_tss=wr_fit_tsso.copy(), TSS_UPLIFT=UPLIFT_TSSO, spread_opt=True, tss_cost=tss_cost)

    # Save results
    q_hw.to_csv('./output/q_hw_2016_2018.csv')
    com_hw.to_csv('./output/com_hw_2016_2018_spd_opt.csv')
    q_tsso.to_csv('./output/q_tss_GTMS_7_27.csv')
    com_tsso.to_csv('./output/com_tss_GTMS_spd_opt_7_27.csv')

    # write model to json and pickle
    write_models(model_dict_hw, hw_quant, './output', 'hw', compressed=False)
    # write_models(model_dict_tss, tss_quant, './output', 'tss', compressed=False)
    write_models(model_dict_tsso, tsso_quant, './output', 'tsso', compressed=False)


main()
