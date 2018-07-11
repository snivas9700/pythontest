# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:21:25 2018

"""
import numpy as np
import pandas as pd
import copy
from copy import deepcopy
import BasePricingFunctions as BPF

class CappingRules:
    def __init__(self):
        pass
    def capping_rules(self,quote_df_all):
        
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than DealBotLineSpreadOptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if (quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] > quote_df_all.loc[i, 'ComListPrice']):
                quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] = quote_df_all.loc[i, 'ComListPrice']
        #quote_df_all.to_csv(data_path + 'spread_optimal_price.csv', index=False)
                
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than DealBotLineSpreadOptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if ((quote_df_all.loc[i, 'ComTMC'] < quote_df_all.loc[i, 'ComListPrice']) & (quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] < quote_df_all.loc[i, 'ComTMC'])):
                quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] = quote_df_all.loc[i, 'ComTMC']
        
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than OptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if (quote_df_all.loc[i, 'OptimalPrice'] > quote_df_all.loc[i, 'ComListPrice']):
                quote_df_all.loc[i, 'OptimalPrice'] = quote_df_all.loc[i, 'ComListPrice']
        #quote_df_all.to_csv(data_path + 'spread_optimal_price.csv', index=False)
                
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than OptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if ((quote_df_all.loc[i, 'ComTMC'] < quote_df_all.loc[i, 'ComListPrice']) & (quote_df_all.loc[i, 'OptimalPrice'] < quote_df_all.loc[i, 'ComTMC'])):
                quote_df_all.loc[i, 'OptimalPrice'] = quote_df_all.loc[i, 'ComTMC']
        #Optimal price is less ComTMC and ComTMC is less then ListPrice then cap optimal price with ComTMC and recalucate the intervals
        for i in range(len(quote_df_all)):
            quote_df_all.loc[i, 'OptimalPriceIntervalLow'], quote_df_all.loc[i, 'OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'OptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])

        for i in range(len(quote_df_all)):
            quote_df_all.loc[i, 'L1'], quote_df_all.loc[i, 'H1'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'OptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
            quote_df_all.loc[i, 'L2'], quote_df_all.loc[i, 'H2'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
            quote_df_all.loc[i,'OptimalPriceIntervalLow'] = min(quote_df_all.loc[i, 'L1'],quote_df_all.loc[i, 'L2'])
            quote_df_all.loc[i,'OptimalPriceIntervalHigh'] = max(quote_df_all.loc[i, 'H1'],quote_df_all.loc[i, 'H2'])

        if(quote_df_all.shape[0] > 0):
            del quote_df_all['L1']
            del quote_df_all['L2']
            del quote_df_all['H1']
            del quote_df_all['H2'] 
        
        quote_df_all = deepcopy(quote_df_all)
        
        return quote_df_all
        