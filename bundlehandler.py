# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:29:09 2017

@author: Ravindranath
"""

import numpy as np
import pandas as pd
import copy
from copy import deepcopy

#set paths and import COPRA_OP .py modules
import sys
sys.path.insert(
    0, '.'
)  #this is the path to the directory that holds the python optimal pricing code

import BasePricingFunctions as BPF
import PricingEngine as PricingEngine
#import needed libraries
import pandas as pd
import numpy as np
import operator
#import matplotlib.pyplot as plt
#from scipy import stats
from pandas import Series, DataFrame, pivot_table
import time


class bundlehandler:
    """class for calculating Optimal Prices for bundled products"""

    def __init__(self):
        pass

    def bundle_flat_spreading(
            self,
            bundled_quotes_df,
    ):
        """class for calculating Optimal Prices for bundled products"""
        """model_ID = model ID to be used by engine, quote_df = quote dataframe, COP values for Low, Median, High"""

        if bundled_quotes_df.empty:
            return bundled_quotes_df
        else:
            #bundled_quotes_df.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/quote_df_bundled.csv')
            bundled_quotes_df['bundle1'] = bundled_quotes_df['Bundle'].str[1:]
            #print(bundled_quotes_df['bundle1'])
            #print(type(bundled_quotes_df['bundle1']))
            bundled_quotes_df['bundle2'] = pd.to_numeric(
                bundled_quotes_df['bundle1'])
            #print(type(bundled_quotes_df['bundle2'][0]))
            l = bundled_quotes_df['bundle1'].tolist()
            l = [int(i) for i in l]
            l = list(set(l))
            #print(l)
            m = max(l)
            bundled_quotes_df_out = pd.DataFrame()
            temp_df = pd.DataFrame()

            for i in l:
                #print bundled_quotes_df[bundled_quotes_df['bundle2'] == i]
                temp_df = bundled_quotes_df[bundled_quotes_df['bundle2'] == i]
                AdjComLowPriceSum = temp_df['AdjComLowPrice'].sum()
                AdjComMedPriceSum = temp_df['AdjComMedPrice'].sum()
                OptimalPriceSum = temp_df['OptimalPrice'].sum()
                OptimalPriceIntervalLowSum = temp_df['OptimalPriceIntervalLow'].sum()                
                OptimalPriceIntervalHighSum = temp_df['OptimalPriceIntervalHigh'].sum()
                AdjComHighPriceSum = temp_df['AdjComHighPrice'].sum()
                ComListPriceSum = temp_df['ComListPrice'].sum()
                temp_df.index = range(len(temp_df))
                for j in range(len(temp_df)):

                    temp_df.loc[j,'AdjComLowPofL'] = AdjComLowPriceSum / ComListPriceSum
                    temp_df.loc[j,'AdjComMedPofL'] = AdjComMedPriceSum / ComListPriceSum                    
                    temp_df.loc[j,'AdjComHighPofL'] = AdjComHighPriceSum / ComListPriceSum                                        
                    
                    temp_df.loc[j,'AdjComLowPrice'] = temp_df.loc[
                        j, 'ComListPrice'] * AdjComLowPriceSum / ComListPriceSum
                    temp_df.loc[j,'AdjComMedPrice'] = temp_df.loc[
                        j, 'ComListPrice'] * AdjComMedPriceSum / ComListPriceSum
                    temp_df.loc[j, 'AdjComHighPrice'] = temp_df.loc[
                        j, 'ComListPrice'] * AdjComHighPriceSum / ComListPriceSum
                        
                    temp_df.loc[j,'OptimalPricePofL'] = OptimalPriceSum / ComListPriceSum
                    temp_df.loc[j,'OptimalPrice'] = temp_df.loc[
                        j, 'ComListPrice'] * OptimalPriceSum / ComListPriceSum
                        
                    temp_df.loc[j, 'OptimalPriceIntervalLow'] = temp_df.loc[
                        j, 'ComListPrice'] * OptimalPriceIntervalLowSum / ComListPriceSum
                    temp_df.loc[j, 'OptimalPriceIntervalHigh'] = temp_df.loc[
                        j, 'ComListPrice'] * OptimalPriceIntervalHighSum / ComListPriceSum

                bundled_quotes_df_out = pd.concat(
                    [bundled_quotes_df_out, temp_df])

            bundled_quotes_df_out1 = deepcopy(bundled_quotes_df_out)
            return bundled_quotes_df_out1
