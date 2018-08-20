# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:01:02 2017

@author: Ravindranath
"""
"""Connect all classes/objects together in a system (see arrows in the Python Web Service architecture)
    """
import sys
import os
import re
import copy
from copy import deepcopy
import numpy as np
import pandas as pd
#import pandasql
#from pandasql import *
import time
from config import *
from geomaptablereader import GeoMapTableReader
from xml2dfconverter import XML2dfConverter
from prodconsolidator import ProdConsolidator
from zeropricefilter import ZeroPriceFilter
from geodispatcher import GeoDispatcher
from datafieldsfilter import DataFieldsFilter
from fixedprodtablereader import FixedProdTableReader
from fixedproddetector import FixedProdDetector
from fixedprodhandler import FixedProdHandler
from normalhandler import NormalHandler
from bundletablereader import bundletablereader
from bundledetector import BundleDetector
from bundlehandler import bundlehandler
from pricespreader import PriceSpreader
from reporter import Reporter
from df2xmlconverter import Df2XMLConverter
from HWReader import HWReader
from HWValidator import HWValidator
from HWValidProdHandler import HWValidProdHandler
from cappingrules import CappingRules
from tss_fixed_discount import TssFixedDiscounts
import BasePricingFunctions as BPF

from os import path as os_path
from os.path import join as os_join, dirname, abspath


data_path = DATA_PATH


#fixed_discounts_path = data_path

class System:
    """Connect all classes/objects together in a system (see arrows in the Python Web Service architecture)
    """

    def __init__(self):
        """read geomap table        
        """
        self._geomap = GeoMapTableReader().read(GEOMAP)
        self._geomap = self._geomap.fillna('NA') #since Python was considering NA as NAN  
        #print (self._geomap)
        #print ('ASIA++++++')
        """read fixed table for EMEA & NA
        """
        self._discounting_rules_EMEA = FixedProdTableReader().read_EMEA(FIXEDPRODTABLE_EMEA)                
        self._discounting_rules_NA = FixedProdTableReader().read_NA(FIXEDPRODTABLE_NA)
        self._discounting_rules_JP = FixedProdTableReader().read_NA(FIXEDPRODTABLE_JP)
        """read bundle table for NA
        """
        self._bundle_table_NA = bundletablereader().read(BUNDLETABLE, data_path)
    
    def run_logic(self, modelId, quote_xml,loaded_models):
        """param input_quote_xml
           retur: output_quote_xml
        """

        #geomap = GeoMapTableReader().read(GEOMAP)
        #print (quote_xml)
        #print (self._geomap)
        ComRevDivCd_orgi, ComRevDivCd_orgi2, final_output, out_PricePoint_data = XML2dfConverter().xmltodf(
                quote_xml, self._geomap) 

        final_output = ProdConsolidator().consolidate(final_output)
          
        # checking quotes for TSS components to add TSS related col to ZeroPriceFilter DataFrame
        if 'TSSComponentincluded' not in final_output or final_output['TSSComponentincluded'][0] == 'N':
            
            df_zero_price, final_output = ZeroPriceFilter().zero_price(
                final_output)
            
        elif (final_output['TSSComponentincluded'][0] == 'Y'):
            df_zero_price, final_output = ZeroPriceFilter().zero_price(
                final_output)
               
        final_output = GeoDispatcher().dispatch(final_output)
        df_zero_price = GeoDispatcher().dispatch(df_zero_price)
                                 
        if (len(final_output) == 0):
            
            quote_df_out, total_deal_stats, PricePoint_data = Reporter(
            ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
                     ComRevDivCd_orgi2, final_output)
            
        #check Quote if it is EMEA_old,EMEA_TSS,EMEA_NEW and selecting datafields accordingly
        elif (final_output['GEO_CODE'][0] == 'EMEA'):
            
            # handle old EMEA quote for old optimalprice calculation
            if (GeoDispatcher().is_EMEA_old(final_output)):
                print ("%%%%%%%% &&&&&&& ****** EMEA OLD MODEL")    
                ######################### changes##############################################      
                #if ('TSSComponentincluded' not in final_output.columns) and (final_output['GEO_CODE'][0] == 'EMEA'):
                final_output = DataFieldsFilter().get_EMEA(final_output)
                data_fields = final_output
                
                fixed_quotes, non_fixed_quotes = FixedProdDetector().split_EMEA(
                self._discounting_rules_EMEA, data_fields, self._geomap)
                #fixed_quotes.to_csv('./output_results/fixed_quotes.csv')
                #non_fixed_quotes.to_csv('./output_results/non_fixed_quotes.csv')
                                                               
                final_output_validator = FixedProdHandler().process_EMEA(fixed_quotes)
                #final_output_fixed.to_csv('./output_results/final_output_fixed.csv')
                
                final_output_normal = NormalHandler().handle_EMEA(
                    modelId,
                    DEPLOYEDMODELRULE_EMEA,
                    non_fixed_quotes,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0)
                non_fixed_quotes = final_output_normal
                #non_fixed_quotes.to_csv('./output_results/non_fixed_quotes.csv')                
                non_fixed_quotes = CappingRules().capping_rules(non_fixed_quotes)

            # handle EMEA TSS quote for new optimalprice calculation
            elif (not GeoDispatcher().is_EMEA_old(final_output)): 
                print ("%%%%%%%% &&&&&&& ****** is_EMEA_TSS LME AND NON TSS LME")                               
                                                               
                final_output_normal = NormalHandler().handle_EMEA_LME(
                                modelId,
                                DEPLOYEDMODELRULE_EMEA,
                                final_output,
                                data_path,
                                COP_l=0,
                                COP_m=0,
                                COP_h=0,
                                loaded_models=loaded_models)
                #final_output_normal.to_csv('./output_results/final_output_online.csv')
                
                if final_output_normal['TSSComponentincluded'][0] == 'Y':
                    final_output_normal = TssFixedDiscounts().apply(final_output_normal)
                    final_output_normal = CappingRules().tss_capping_rules(final_output_normal)
                #final_output_normal.to_csv('./output_results/final_online_aft_TSSdiscount_capping.csv')
                
                quote_df_all = final_output_normal.copy()
                discounts_df = HWReader().read_table(final_output['GEO_CODE'][0])

                #Detect the validator products and split them into 2 dataframes 
                validator_quotes, final_output_normal = HWValidator().detect_and_split(discounts_df , quote_df_all, self._geomap)
    
                #Apply appropriate rules to the validator products
                final_output_validator = HWValidProdHandler().apply_rules(validator_quotes)
                non_fixed_quotes = CappingRules().capping_rules(final_output_normal)                
                
                #final_output_validator.to_csv('./output_results/final_output_validator.csv')            
                #non_fixed_quotes.to_csv('./output_results/final_output_nonfixed_TSS.csv')
                
            else:
                raise ValueError('Invalid Quote')
                                        
            quote_df_out, total_deal_stats, PricePoint_data = Reporter(
            ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
                 ComRevDivCd_orgi2, final_output_validator,
                 non_fixed_quotes)
            #total_deal_stats.to_csv('./output_results/total_deal_stats_reporter.csv')
                                                        
        elif (final_output['GEO_CODE'][0] == 'NA'):

            final_output = DataFieldsFilter().get_NA(final_output)
            data_fields = final_output
            
            if (not GeoDispatcher().is_BP(data_fields)):            
                print ("%%%%%%%% &&&&&&& ****** Going to NA DIRECT")                
                final_output_normal = NormalHandler().handle_NA(
                    modelId,
                    DEPLOYEDMODELRULE_NA,
                    data_fields,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0,
                    loaded_models=loaded_models["Direct"])
                    
            elif (GeoDispatcher().is_BP(data_fields)):
                print ("%%%%%%%% &&&&&&& ****** Going to BP")
                final_output_normal = NormalHandler().handle_NA_BP(modelId,
                    DEPLOYEDMODELRULE_NA,
                    data_fields,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0,
                    loaded_models=loaded_models["BP"])

            else:

                raise ValueError('Invalid QuoteType')
                             
                       
            bundled_quotes, non_bundled_quotes = BundleDetector().detect(
                self._bundle_table_NA, final_output_normal)

            final_output_bundled = bundlehandler().bundle_flat_spreading(
                bundled_quotes)

            quote_df_all = PriceSpreader().spread_botline_optimal_price(0, 0, 0, final_output_bundled, non_bundled_quotes)
            
            discounts_df = HWReader().read_table(final_output['GEO_CODE'][0])

            #Detect the validator products and split them into 2 dataframes 
            validator_quotes, final_output_normal = HWValidator().detect_and_split(discounts_df , quote_df_all, self._geomap)

            #Apply appropriate rules to the validator products
            final_output_validator = HWValidProdHandler().apply_rules(validator_quotes)
            final_output_normal = CappingRules().capping_rules(final_output_normal)
                        
            quote_df_out, total_deal_stats, PricePoint_data = Reporter(
            ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
                     ComRevDivCd_orgi2, final_output_validator, final_output_normal)
            
        elif (final_output['GEO_CODE'][0] == 'JP'):
            
            if GeoDispatcher().is_BP(final_output):
                final_output = DataFieldsFilter().get_JP_BP(final_output)
            else:
                final_output = DataFieldsFilter().get_JP(final_output)
                
            data_fields = copy.deepcopy(final_output)
            
            #Pass all the components to the processing engine
            if (not GeoDispatcher().is_BP(final_output)):            
            
                all_output_normal = NormalHandler().handle_JP(
                    data_fields,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0,
                    loaded_models=loaded_models["Direct"])
                    
            elif (GeoDispatcher().is_BP(final_output)):
                
                all_output_normal = NormalHandler().handle_JP_BP(
                    data_fields,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0,
                    loaded_models=loaded_models["BP"])

            else:

                raise ValueError('Invalid QuoteType')
            
            #load the validator csv into a pandas dataframe
            discounts_df = HWReader().read_table(final_output['GEO_CODE'][0])
            
            #Detect the validator products and split them into 2 dataframes 
            validator_quotes, final_output_normal = HWValidator().detect_and_split(discounts_df , all_output_normal, self._geomap)
            
            #Apply appropriate rules to the validator products
            final_output_validator = HWValidProdHandler().apply_rules(validator_quotes)
            final_output_normal = CappingRules().capping_rules(final_output_normal)
            
            quote_df_out, total_deal_stats, PricePoint_data = Reporter(
            ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
                     ComRevDivCd_orgi2, final_output_validator, final_output_normal)
            
        else:
            raise ValueError('GEO not updated yet.')

        #quote_df_out['X'] = quote_df_out['ComListPrice'].map(str) + quote_df_out['ComMTM'] + quote_df_out['ComQuotePrice'].map(str)

        #if ((quote_df_out['GEO_CODE'][0] == 'EMEA') & (not GeoDispatcher().is_EMEA_old)):
        if 'TSSComponentincluded' in quote_df_out:
            if quote_df_out['TSSComponentincluded'][0] == 'Y':
                quote_df_out = quote_df_out.drop_duplicates(subset=['Componentid','TSScomid'], keep='first')
                quote_df_out.reset_index(drop=True, inplace=True)
            elif quote_df_out['TSSComponentincluded'][0] == 'N':
                quote_df_out = quote_df_out.drop_duplicates(
                subset=['Componentid'], keep='first')
                quote_df_out.reset_index(drop=True, inplace=True)
        else:
            quote_df_out = quote_df_out.drop_duplicates(
                subset=['Componentid'], keep='first')
            quote_df_out.reset_index(drop=True, inplace=True)

        quote_df_out1 = pd.DataFrame(quote_df_out)
       # _, i = np.unique(quote_df_out1.columns, return_index=True)
       # quote_df_out1 = quote_df_out1.iloc[:, i]

        quote_df_out_response = Df2XMLConverter().df_to_xml(
            quote_df_out1, total_deal_stats, PricePoint_data,
            'quote_df_out_response.xml')
        
        print ('5001 port code running+++++++++++++++++++++++++++++')
                
        return quote_df_out_response
