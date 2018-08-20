from pandas import merge, to_numeric, DataFrame, Series
from numpy import log, log10, arange

def data_prep_hw_com(data):
    dat = data.copy()
    dat['chnl_ep'] = dat['Indirect(1/0)']

    hw_nodup = dat.drop_duplicates('componentid').copy()
    dealsize = sum(hw_nodup['p_list_hw'])
    dat['DealSize'] = dealsize
    dat['p_list_hw_total_log'] = log10(dat['DealSize']+1)
    dat['p_bid_hw_contrib'] = dat['p_list_hw']/dat['DealSize']
    dat['sector'] = dat['CustomerSecName'].str.lower()
    dat['industry'] = dat['CustomerIndustryName'].str.lower()
    dat['bundled'] = (dat['TSSComponentincluded'] == 'Y') * 1
    dat['const'] = 1

    return dat


def data_prep_tss_com(data):
    data = data[~data['p_list_hwma'].isin([0])]
    dat = data.copy()
    dat['p_list_hwma_log'] = log10(dat['p_list_hwma'] + 1)
    dat['chnl_tss'] = dat['Indirect(1/0)']

    hw_nodup = dat.drop_duplicates('componentid').copy()
    dealsize_hw = sum(hw_nodup['p_list_hw'])
    dealsize_tss = sum(dat['p_list_hwma'])
    dat['dealsize'] = dealsize_hw + dealsize_tss
    dat['p_list_hwma_hw_log'] = log10(dat['dealsize'] + 1)
    dat['tss_type'] = dat['servoffcode'].str.lower()
    dat['committed'] = (dat['committedcharge'] != 0) * 1

    dat['mtm'] = dat['taxon_hw_mtm']
    dat['sector'] = dat['CustomerSecName'].str.lower()
    dat['industry'] = dat['CustomerIndustryName'].str.lower()
    dat['tss_bundled'] = (dat['TSSComponentincluded'] == 'Y') * 1
    dat['const'] = 1

    return dat


def prep_output(df):
    # HW-specific provide nonzero values
    print('Generating the optimal prices...')
    df['TreeNode'] = None  # CS
    df['GEO_CODE'] = 'EMEA'
    df['DealBotLineSpreadOptimalPrice'] = df['bot_spread_OP_hw']
    df['OptimalPrice'] = df['bot_spread_OP_hw']
    df['OptimalPriceExpectedGP'] = (df['OptimalPrice'] - df['ComTMC']) * df['Win_Rate_at_bot_spread_OP_hw']
    df['OptimalPriceGP'] = df['OptimalPrice'] - df['ComTMC']
    df['OptimalPriceIntervalHigh'] = df['ci_high_hw'] * df['OptimalPrice']
    df['OptimalPriceIntervalLow'] = df['ci_low_hw'] * df['OptimalPrice']
    df['OptimalPricePofL'] = df["OptimalPrice"] / df["ComListPrice"]
    df['OptimalPriceWinProb'] = df['Win_Rate_at_OP_GP_hw']
    df['PredictedQuotePrice'] = df['OptimalPrice']
    df['PredictedQuotePricePofL'] = df['OptimalPricePofL']
    df['QuotePrice'] = df['OptimalPrice']
    df['QuotePriceExpectedGP'] = (df['QuotePrice'] - df["ComTMC"]) * df['Win_Rate_at_OP_GP_hw']
    df['QuotePriceGP'] = df['QuotePrice'] - df["ComTMC"]
    df['QuotePricePofL'] = df['QuotePrice'] / df["ComListPrice"]
    df['QuotePriceWinProb'] = df['Win_Rate_at_OP_GP_hw']
    df['DealSize'] = df[['Componentid', 'ComTMC']].drop_duplicates('Componentid').ComTMC.sum()
    # GZ: try to in line with today's definition in HW data_prep.py
    df['LogDealSize'] = log10(df['DealSize'] + 1)
    df['ComPctContrib'] = df['ComTMC'] / df['DealSize']  # GZ: try to in line with today's definition in HW data_prep.py

    # HW/Low,Med,High-Specific
    df['AdjComHighPrice'] = df['pred_H_hw'] * df['hw_value_score']
    df['AdjComHighPofL'] = df['AdjComHighPrice'] / df['ComListPrice']
    df['AdjComLowPrice'] = df['pred_L_hw'] * df['hw_value_score']
    df['AdjComLowPofL'] = df['AdjComLowPrice'] / df['ComListPrice']
    df['AdjComMedPrice'] = df['pred_M_hw'] * df['hw_value_score']
    df['AdjComMedPofL'] = df['AdjComMedPrice'] / df['ComListPrice']

    df['AdjComHighPrice'] = df[['OptimalPriceIntervalHigh', 'AdjComHighPrice']].max(axis=1)
    df['AdjComLowPrice'] = df[['OptimalPriceIntervalLow', 'AdjComLowPrice']].min(axis=1)

    df['ComMedPrice'] = df['pred_M_hw'] * df['hw_value_score']
    df['ComMedPofL'] = df['ComMedPrice'] / df['ComListPrice']
    df['ComLowPrice'] = df['pred_L_hw'] * df['hw_value_score']
    df['ComLowPofL'] = df['ComLowPrice'] / df['ComListPrice']
    df['ComHighPrice'] = df['pred_H_hw'] * df['hw_value_score']
    df['ComHighPofL'] = df['ComHighPrice'] / df['ComListPrice']

    # TSS-Specific
    df['TSSContduration'].loc[(df['TSSContduration'] > 1825) | (df['TSSContduration'] < 0)] = 1825

    df['TSS_DealBotLineSpreadOptimalPrice'] = df['bot_spread_OP_tss']
    df['TSS_OptimalPrice'] = df['bot_spread_OP_tss']
    df['TSS_OptimalPriceExpectedGP'] = (df['TSS_OptimalPrice'] - df['Cost']) * df['Win_Rate_at_bot_spread_OP_tss']
    df['TSS_OptimalPriceGP'] = df['TSS_OptimalPrice'] - df['Cost']
    df['TSS_OptimalPriceIntervalHigh'] = df['ci_high_tss'] * df['TSS_OptimalPrice']
    df['TSS_OptimalPriceIntervalLow'] = df['ci_low_tss'] * df['TSS_OptimalPrice']
    df['TSS_OptimalPricePofL'] = df['TSS_OptimalPrice'] / df['totalcharge']
    df['TSS_OptimalPriceWinProb'] = df['Win_Rate_at_OP_GP_tss']
    df['pti'] = 1 - df['PTI0'] / df['TSS_OptimalPrice'] * 0.7 - 0.3
    df['hwma_pti'] = df['pti']

    # TSS/Adj-Specific
    df['TSS_AdjComHighPrice'] = df['pred_H_tss'] * df['tss_value_score']
    df['TSS_AdjComHighPofL'] = df['TSS_AdjComHighPrice'] / df['totalcharge']
    df['TSS_AdjComLowPrice'] = df['pred_L_tss'] * df['tss_value_score']
    df['TSS_AdjComLowPofL'] = df['TSS_AdjComLowPrice'] / df['totalcharge']
    df['TSS_AdjComMedPrice'] = df['pred_M_tss'] * df['tss_value_score']
    df['TSS_AdjComMedPofL'] = df['TSS_OptimalPrice'] / df['totalcharge']

    df['TSS_AdjComHighPrice'] = df[['TSS_OptimalPriceIntervalHigh', 'TSS_AdjComHighPrice']].max(axis=1)
    df['TSS_AdjComLowPrice'] = df[['TSS_OptimalPriceIntervalLow', 'TSS_AdjComLowPrice']].min(axis=1)

    df['ComTMCPofL'] = df['ComTMC']/(10**df['ComLogListPrice'])

    # Customized Optimal Price (COP) fields should be removed (not for now)
    # TSS-specific provide nonzero

    return df


def prep_output_nonTSS(df):
    # HW-specific provide nonzero values
    print('Generating the optimal prices...')
    df['TreeNode'] = None  # CS
    df['GEO_CODE'] = 'EMEA'
    df['DealBotLineSpreadOptimalPrice'] = df['bot_spread_OP_hw']
    df['OptimalPrice'] = df['bot_spread_OP_hw']
    df['OptimalPriceExpectedGP'] = (df['OptimalPrice'] - df['ComTMC']) * df['Win_Rate_at_bot_spread_OP_hw']
    df['OptimalPriceGP'] = df['OptimalPrice'] - df['ComTMC']

    df['OptimalPricePofL'] = df["OptimalPrice"] / df["ComListPrice"]
    df['OptimalPriceWinProb'] = df['Win_Rate_at_OP_GP_hw']
    df['PredictedQuotePrice'] = df['OptimalPrice']
    df['PredictedQuotePricePofL'] = df['OptimalPricePofL']

    df['OptimalPriceIntervalHigh'] = df['ci_high_hw'] * df['OptimalPrice']
    df['OptimalPriceIntervalLow'] = df['ci_low_hw'] * df['OptimalPrice']

    df['QuotePrice'] = df['OptimalPrice']
    df['QuotePriceExpectedGP'] = (df['QuotePrice'] - df["ComTMC"]) * df['Win_Rate_at_OP_GP_hw']
    df['QuotePriceGP'] = df['QuotePrice'] - df["ComTMC"]
    df['QuotePricePofL'] = df['QuotePrice'] / df["ComListPrice"]
    df['QuotePriceWinProb'] = df['Win_Rate_at_OP_GP_hw']
    df['DealSize'] = df[['Componentid', 'ComTMC']].drop_duplicates('Componentid').ComTMC.sum()

    # df['DealSize'] = df[['componentid', 'ComTMC']].drop_duplicates('componentid').ComTMC.sum()
    # GZ: try to in line with today's definition in HW data_prep.py
    df['LogDealSize'] = log10(df['DealSize'] + 1)
    df['ComPctContrib'] = df['ComTMC'] / df['DealSize']  # GZ: try to in line with today's definition in HW data_prep.py

    # HW/Low,Med,High-Specific
    df['AdjComHighPrice'] = df['pred_H_hw'] * df['hw_value_score']
    df['AdjComHighPofL'] = df['AdjComHighPrice'] / df['ComListPrice']
    df['AdjComLowPrice'] = df['pred_L_hw'] * df['hw_value_score']
    df['AdjComLowPofL'] = df['AdjComLowPrice'] / df['ComListPrice']
    df['AdjComMedPrice'] = df['pred_M_hw'] * df['hw_value_score']
    df['AdjComMedPofL'] = df['AdjComMedPrice'] / df['ComListPrice']

    df['AdjComHighPrice'] = df[['OptimalPriceIntervalHigh', 'AdjComHighPrice']].max(axis=1)
    df['AdjComLowPrice'] = df[['OptimalPriceIntervalLow', 'AdjComLowPrice']].min(axis=1)

    df['ComMedPrice'] = df['pred_M_hw'] * df['hw_value_score']
    df['ComMedPofL'] = df['ComMedPrice'] / df['ComListPrice']
    df['ComLowPrice'] = df['pred_L_hw'] * df['hw_value_score']
    df['ComLowPofL'] = df['ComLowPrice'] / df['ComListPrice']
    df['ComHighPrice'] = df['pred_H_hw'] * df['hw_value_score']
    df['ComHighPofL'] = df['ComHighPrice'] / df['ComListPrice']
    df['ComTMCPofL'] = df['ComTMC']/(10**df['ComLogListPrice'])

    return df


def init_prep(df_in):
    cols = ['ctry_desc', 'Level_4', 'Level_3', 'Level_1', 'Level_2']
    for col in cols:
        if col in df_in.columns:
            df_in[col] = df_in[col].str.lower()

    if 'imt_code' in df_in.columns:
        df_in['imt_code'] = df_in['imt_code'].str.slice(0,3,1)
    dic1 = {'HW-MAINTENANCE': "hw maintenance", 'WSU': 'wsu'}
    df_in.replace({'servoffcode':dic1})

    return df_in


def final_prep(df_out, df_in):
    cols = ['ctry_desc', 'Level_4', 'Level_3', 'imt_code', 'Level_2', 'Level_1', 'servoffcode']
    for col in cols:
        if col in df_in.columns:
            df_out[col] = df_in[col].copy()

    return df_out


def swma_prep(fixed_quotes):
    fixed_quotes['TSS_AdjComLowPrice'] = 0.4*fixed_quotes['p_list_hwma'] - 1
    fixed_quotes['TSS_AdjComMedPrice'] = fixed_quotes['TSS_AdjComLowPrice'] + 1  # make sure AdjComLowPrice is less than AdjComMedPrice
    fixed_quotes['TSS_AdjComHighPrice'] = fixed_quotes['TSS_AdjComLowPrice'] + 2  # make sure AdjComHighPrice is greater than AdjComMedPrice

    # optimal_price is same as the discounted_price
    fixed_quotes['TSS_OptimalPrice'] = fixed_quotes['TSS_AdjComMedPrice']  # make sure OptimalPrice equals to AdjComMedPrice
    fixed_quotes['TSS_ComLowPofL'] = fixed_quotes['TSS_AdjComLowPrice'] / fixed_quotes.loc[:, 'p_list_hwma']
    fixed_quotes['TSS_ComMedPofL'] = fixed_quotes['TSS_AdjComMedPrice'] / fixed_quotes.loc[:, 'p_list_hwma']
    # fixed_quotes['ComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc[:, 'p_list_hwma']
    # fixed_quotes['ComMedPrice'] = fixed_quotes['AdjComMedPrice']

    # The tree node has no significance as we are not using pricing engine and segmentation
    # setting to a unique number -999 just to indicate that this is not a number set by engine
    fixed_quotes['TreeNode'] = -999
    #fixed_quotes['ComTMCPofL'] = fixed_quotes.loc[:, 'cost'] / fixed_quotes.loc[:, 'p_list_hwma']
    fixed_quotes['TSS_AdjComLowPofL'] = fixed_quotes['TSS_AdjComLowPrice'] / fixed_quotes.loc[:, 'p_list_hwma']
    fixed_quotes['TSS_AdjComMedPofL'] = fixed_quotes['TSS_AdjComMedPrice'] / fixed_quotes.loc[:, 'p_list_hwma']
    fixed_quotes['TSS_AdjComHighPofL'] = fixed_quotes['TSS_AdjComHighPrice'] / fixed_quotes.loc[:, 'p_list_hwma']
    fixed_quotes['TSS_OptimalPricePofL'] = fixed_quotes['TSS_OptimalPrice'] / fixed_quotes.loc[:, 'p_list_hwma']

    # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
    fixed_quotes['TSS_OptimalPriceWinProb'] = 0.5
    fixed_quotes['TSS_OptimalPriceGP'] = fixed_quotes['TSS_OptimalPrice'] - fixed_quotes.loc[:, 'cost']
    fixed_quotes['TSS_OptimalPriceExpectedGP'] = fixed_quotes['TSS_OptimalPriceGP'] * 0.5

    # setting intervalLow and intervalHigh also as same as discounted_price because this is a fixed discount anyways
    fixed_quotes['TSS_OptimalPriceIntervalLow'] = fixed_quotes['TSS_OptimalPrice']
    fixed_quotes['TSS_OptimalPriceIntervalHigh'] = fixed_quotes['TSS_OptimalPrice']
    fixed_quotes['TSS_DealBotLineSpreadOptimalPrice'] = fixed_quotes['TSS_OptimalPrice']
    fixed_quotes['pti'] = 1 - fixed_quotes['PTI_0price'] / fixed_quotes['TSS_OptimalPrice'] * 0.7 - 0.3
    fixed_quotes['hwma_pti'] = fixed_quotes['pti']
    # fixed_quotes['QuotePrice'] = fixed_quotes['p_bid_hwma']
    # fixed_quotes['QuotePricePofL'] = fixed_quotes['QuotePrice']/fixed_quotes['p_list_hwma']

    # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
    # fixed_quotes['QuotePriceWinProb'] = 0.5
    # fixed_quotes['QuotePriceGP'] = fixed_quotes['QuotePrice'] - fixed_quotes.loc[:, 'cost']
    # fixed_quotes['QuotePriceExpectedGP'] = fixed_quotes['QuotePriceGP'] * 0.5
    # fixed_quotes['PredictedQuotePricePofL'] = fixed_quotes['QuotePrice'] / fixed_quotes.loc[:, 'p_list_hwma']
    # fixed_quotes['PredictedQuotePrice'] = fixed_quotes['QuotePrice']

    return fixed_quotes
