# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:21:25 2018

"""
import numpy as np
import pandas as pd
import copy
from copy import deepcopy
import BasePricingFunctions as BPF
from geodispatcher import GeoDispatcher

class CappingRules:
    def __init__(self):
        pass
    def capping_rules(self,quote_df_all):
        
        if ( quote_df_all.shape[0] < 1 ):
            quote_df_all = deepcopy(quote_df_all)
            return quote_df_all
        
        #if (quote_df_all['Countrycode'][0] == 'JP'):
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than OptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if (quote_df_all.loc[i, 'OptimalPrice'] > quote_df_all.loc[i, 'ComListPrice']):
                quote_df_all.loc[i, 'OptimalPrice'] = quote_df_all.loc[i, 'ComListPrice']

                # For EMEA quotes, also adjust the intervals
                if ('GEO_CODE' in quote_df_all and quote_df_all['GEO_CODE'][0] == 'EMEA'):
                    quote_df_all['OptimalPriceIntervalLow'] = quote_df_all['OptimalPrice'] * 0.9
                    quote_df_all['OptimalPriceIntervalHigh'] = quote_df_all['OptimalPrice'] * 1.1
            #since ComListPrice was coming greater than DealBotLineSpreadOptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if (quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] > quote_df_all.loc[i, 'ComListPrice']):
                quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] = quote_df_all.loc[i, 'ComListPrice']

                #For EMEA quotes, also adjust deal bottom line interval
                if ('GEO_CODE' in quote_df_all and quote_df_all['GEO_CODE'][0] == 'EMEA'):
                    quote_df_all['DealBotLineOptimalPriceIntervalLow'] = quote_df_all['OptimalPrice'] * 0.9
                    quote_df_all['DealBotLineOptimalPriceIntervalHigh'] = quote_df_all['OptimalPrice'] * 1.1

        if (GeoDispatcher().is_BP(quote_df_all) & (quote_df_all['Countrycode'][0] in ('CA','US'))):
            '''for i in range(len(quote_df_all)): #since ComListPrice was coming greater than DealBotLineSpreadOptimalPrice. So, to correct that capping done on 26th Mar 2018.
                if ((quote_df_all.loc[i, 'ComTMC'] < quote_df_all.loc[i, 'ComListPrice']) & (quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] < quote_df_all.loc[i, 'ComTMC'])):
                    quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] = quote_df_all.loc[i, 'ComTMC']'''
            
            for i in range(len(quote_df_all)): #since ComListPrice was coming greater than OptimalPrice. So, to correct that capping done on 26th Mar 2018.
                if ((quote_df_all.loc[i, 'ComTMC'] < quote_df_all.loc[i, 'ComListPrice']) & (quote_df_all.loc[i, 'OptimalPrice'] < quote_df_all.loc[i, 'ComTMC'])):
                    quote_df_all.loc[i, 'OptimalPrice'] = quote_df_all.loc[i, 'ComTMC']
                                  
            #Optimal price is less ComTMC and ComTMC is less then ListPrice then cap optimal price with ComTMC and recalucate the intervals
            #for i in range(len(quote_df_all)):
                #quote_df_all.loc[i, 'OptimalPriceIntervalLow'], quote_df_all.loc[i, 'OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'OptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
    
        for i in range(len(quote_df_all)):
            
            if ('GEO_CODE' in quote_df_all and quote_df_all['GEO_CODE'][0] != 'EMEA'):
                quote_df_all.loc[i, 'L1'], quote_df_all.loc[i, 'H1'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'OptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
                quote_df_all.loc[i, 'L2'], quote_df_all.loc[i, 'H2'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
                quote_df_all.loc[i,'OptimalPriceIntervalLow'] = min(quote_df_all.loc[i, 'L1'],quote_df_all.loc[i, 'L2'])
                quote_df_all.loc[i,'OptimalPriceIntervalHigh'] = max(quote_df_all.loc[i, 'H1'],quote_df_all.loc[i, 'H2'])
            
                del quote_df_all['L1']
                del quote_df_all['L2']
                del quote_df_all['H1']
                del quote_df_all['H2']
    
        quote_df_all = deepcopy(quote_df_all)
        
        return quote_df_all
    
    def tss_capping_rules(self,final_output_normal):
        
        final_output_normal['TSS_OptimalPrice'] = pd.to_numeric(final_output_normal['TSS_OptimalPrice'],errors='coerce')
        final_output_normal['TSS_DealBotLineSpreadOptimalPrice'] = pd.to_numeric(final_output_normal['TSS_DealBotLineSpreadOptimalPrice'],errors='coerce')
        cond_cap_1 = final_output_normal.TSS_OptimalPrice > final_output_normal.totalcharge
        cond_cap_2 = final_output_normal.TSS_DealBotLineSpreadOptimalPrice > final_output_normal.totalcharge
        final_output_normal.loc[cond_cap_1, 'TSS_OptimalPrice'] = final_output_normal.loc[cond_cap_1, 'totalcharge']
        final_output_normal.loc[cond_cap_2, 'TSS_DealBotLineSpreadOptimalPrice'] = final_output_normal.loc[cond_cap_2, 'totalcharge']
        final_output_normal.loc[cond_cap_1, 'TSS_OptimalPriceIntervalLow'] = final_output_normal.loc[cond_cap_1,'TSS_OptimalPrice'] * 0.9
        final_output_normal.loc[cond_cap_1, 'TSS_OptimalPriceIntervalHigh'] = final_output_normal.loc[cond_cap_1, 'TSS_OptimalPrice'] * 1.1

        final_output_normal = deepcopy(final_output_normal)
        return final_output_normal
                