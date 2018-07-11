# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:23:16 2017
@author: Ravindranath
"""
"""Provide optimal prices info for normal products.
    :param df: output pandas.df (normal part) from FixedProdDetector/BundleDetector.
    :retur a new pandas.df
    """
import os
import sys
sys.path.insert(0, '.')
import BasePricingFunctions as BPF
import PricingEngine as PricingEngine
from config import *
from geodispatcher import GeoDispatcher

import NA_Direct.online_main as OL_Direct
import NA_BP.online as OL_BP

import pandas as pd
import io
import copy
from copy import deepcopy
import numpy as np

data_path = os.path.dirname(os.path.realpath(__file__))



class NormalHandler():
    def __init__(self):
        pass

    def handle_EMEA(self,
                    modelId,
                    deployedmodelrules,
                    quote_df,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0):
        """Function to deal with Normal cases in EMEA quote
        """
        if quote_df.empty:

            quote_df = pd.DataFrame(0, index=np.arange(len(quote_df)), 
                                    columns=NORMALHANDLER_EMEA)

            quote_df1 = deepcopy(quote_df)
            return quote_df1

        else:
            if quote_df['GEO_CODE'][0] == 'EMEA':

                quote_df_out, total_deal_stats = PricingEngine.optimal_pricing_engine(
                    modelId, deployedmodelrules, quote_df, 0, 0, 0)
                quote_df_out1 = deepcopy(quote_df_out)
                total_deal_stats1 = deepcopy(total_deal_stats)
                return quote_df_out1

    def handle_NA(self,
                  modelId,
                  deployedmodelrules,
                  quote_df,
                  data_path,
                  COP_l=0,
                  COP_m=0,
                  COP_h=0,
                  loaded_models=None):
        """Function to deal with Normal cases in NA Non-BP quote
        """
        if quote_df.empty:
            
            quote_df = pd.DataFrame(0, index=np.arange(len(quote_df)), 
                                    columns=NORMALHANDLER_NA)
            
            quote_df1 = deepcopy(quote_df)
            return quote_df1

        else:
            if ((quote_df['GEO_CODE'][0] == 'NA') & (not GeoDispatcher().is_BP(quote_df))):

                quote_df = pd.DataFrame(quote_df, columns=NORMALHANDLER_NA)

                # quote_df_out, total_deal_stats = OL.process_quote(quote_df)
                quote_df['Componentid_Orig'] = quote_df['Componentid']
                quote_df_out = OL_Direct.process_quote(quote_df, loaded_models)
                quote_df_out['Componentid'] = quote_df['Componentid_Orig']
                del quote_df['Componentid_Orig']
                quote_df_out1 = deepcopy(quote_df_out)
                # total_deal_stats1 = deepcopy(total_deal_stats)
                return quote_df_out1


    def handle_NA_BP(self,
                  modelId,
                  deployedmodelrules,
                  quote_df,
                  data_path,
                  COP_l=0,
                  COP_m=0,
                  COP_h=0,
                  loaded_models=None):
        """Function to deal with Normal cases in NA BP quote
        """
        if quote_df.empty:
            quote_df = pd.DataFrame(0, index=np.arange(len(quote_df)),columns=NORMALHANDLER_NA_BP)

            quote_df1 = deepcopy(quote_df)
            return quote_df1

        else:
            if ((quote_df['GEO_CODE'][0] == 'NA') & (GeoDispatcher().is_BP(quote_df))):

                quote_df = pd.DataFrame(quote_df, columns=NORMALHANDLER_NA_BP)

                #quote_df_out, total_deal_stats = OL.process_quote(quote_df)
                quote_df['Componentid_Orig'] = quote_df['Componentid']
                quote_df_out = OL_BP.process_quote(quote_df, loaded_models)
                quote_df_out['Componentid'] = quote_df['Componentid_Orig'] 
                del quote_df['Componentid_Orig']
                quote_df_out1 = deepcopy(quote_df_out)
                #total_deal_stats1 = deepcopy(total_deal_stats)
                return quote_df_out1
            
    def handle_JP(self,
                  quote_df,
                  data_path,
                  COP_l=0,
                  COP_m=0,
                  COP_h=0,
                  loaded_models=None):
        """Function to deal with Normal cases in JP Non-BP quote
        """
        if quote_df.empty:
            
            quote_df = pd.DataFrame(0, index=np.arange(len(quote_df)), 
                                    columns=NORMALHANDLER_JP)
            
            quote_df1 = deepcopy(quote_df)
            return quote_df1

        else:
            if ((quote_df['GEO_CODE'][0] == 'JP') & (not GeoDispatcher().is_BP(quote_df))):

                quote_df = pd.DataFrame(quote_df, columns=NORMALHANDLER_JP)

                # quote_df_out, total_deal_stats = OL.process_quote(quote_df)
                quote_df['Componentid_Orig'] = quote_df['Componentid']
                quote_df_out = OL_Direct.process_quote(quote_df, loaded_models)
                quote_df_out['Componentid'] = quote_df['Componentid_Orig']
                del quote_df['Componentid_Orig']
                quote_df_out1 = deepcopy(quote_df_out)
                # total_deal_stats1 = deepcopy(total_deal_stats)
                return quote_df_out1
            
    def handle_JP_BP(self,
                  quote_df,
                  data_path,
                  COP_l=0,
                  COP_m=0,
                  COP_h=0,
                  loaded_models=None):
        """Function to deal with Normal cases in JP BP quote
        """
        if quote_df.empty:
            quote_df = pd.DataFrame(0, index=np.arange(len(quote_df)),columns=NORMALHANDLER_JP_BP)

            quote_df1 = deepcopy(quote_df)
            return quote_df1

        else:
            if ((quote_df['GEO_CODE'][0] == 'JP') & (GeoDispatcher().is_BP(quote_df))):

                quote_df = pd.DataFrame(quote_df, columns=NORMALHANDLER_JP_BP)

                quote_df['Componentid_Orig'] = quote_df['Componentid']
                quote_df_out = OL_Direct.process_quote(quote_df, loaded_models)
                quote_df_out['Componentid'] = quote_df['Componentid_Orig'] 
                del quote_df['Componentid_Orig']
                quote_df_out1 = deepcopy(quote_df_out)
                return quote_df_out1
