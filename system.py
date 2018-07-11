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
        

    def run_logic(self, modelId, quote_xml, loaded_models):
        """param input_quote_xml
           retur: output_quote_xml
        """

        #geomap = GeoMapTableReader().read(GEOMAP)
        #print (quote_xml)
        #print (self._geomap)
        ComRevDivCd_orgi, ComRevDivCd_orgi2, final_output, out_PricePoint_data = XML2dfConverter().xmltodf(
            quote_xml, self._geomap)
        
        quote_df_out = final_output

        final_output = ProdConsolidator().consolidate(final_output)

        df_zero_price, final_output = ZeroPriceFilter().zero_price(
            final_output)
        
        final_output = GeoDispatcher().dispatch(final_output)
        
        # zero_indicator
        if (len(final_output) == 0):
            
            quote_df_out, total_deal_stats, PricePoint_data = Reporter(
            ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
                     ComRevDivCd_orgi2, final_output)

        elif (final_output['GEO_CODE'][0] == 'EMEA'):
            
            final_output = DataFieldsFilter().get_EMEA(final_output)
            data_fields = final_output

            #discounting_rules = FixedProdTableReader().read(FIXEDPRODTABLE_EMEA)

            fixed_quotes, non_fixed_quotes = FixedProdDetector().split_EMEA(
                self._discounting_rules_EMEA, data_fields, self._geomap)

            final_output_fixed = FixedProdHandler().process_EMEA(fixed_quotes)

            final_output_normal = NormalHandler().handle_EMEA(
                modelId,
                DEPLOYEDMODELRULE_EMEA,
                non_fixed_quotes,
                data_path,
                COP_l=0,
                COP_m=0,
                COP_h=0)
            #final_output_normal.to_csv('C:/Users/IBM_ADMIN/Documents/Chaitra/$user/COPRA/NA/WS_Phase2/final_output_normal_137.csv')                 


            quote_df_out, total_deal_stats, PricePoint_data = Reporter(
            ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
                     ComRevDivCd_orgi2, final_output_fixed,
                     final_output_normal)


        elif (final_output['GEO_CODE'][0] == 'NA'):
            
            final_output = DataFieldsFilter().get_NA(final_output)
            data_fields = final_output
            #final_output.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/datafieldsfilter.csv')

            #discounting_rules = FixedProdTableReader().read(FIXEDPRODTABLE_NA)

            fixed_quotes, non_fixed_quotes = FixedProdDetector().split_NA(
                self._discounting_rules_NA, data_fields, self._geomap)
            #fixed_quotes.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/fixed_quotes.csv')

            final_output_fixed = FixedProdHandler().process_NA(fixed_quotes)
            #final_output_fixed.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/final_output_fixed.csv')
            #print(non_fixed_quotes['ComMedPrice'])
            #input("test");

            if (not GeoDispatcher().is_BP(final_output)):            
            
                final_output_normal = NormalHandler().handle_NA(
                    modelId,
                    DEPLOYEDMODELRULE_NA,
                    non_fixed_quotes,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0,
                    loaded_models=loaded_models["Direct"])
                    
            elif (GeoDispatcher().is_BP(final_output)):
                
                final_output_normal = NormalHandler().handle_NA_BP(modelId,
                    DEPLOYEDMODELRULE_NA,
                    non_fixed_quotes,
                    data_path,
                    COP_l=0,
                    COP_m=0,
                    COP_h=0,
                    loaded_models=loaded_models["BP"])

            else:

                raise ValueError('Invalid QuoteType')
                             
            #bundle_table = bundletablereader().read(BUNDLETABLE, data_path)

            bundled_quotes, non_bundled_quotes = BundleDetector().detect(
                self._bundle_table_NA, final_output_normal)
            #bundled_quotes.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/bundled_quotes.csv')
            #non_bundled_quotes.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/non_bundled_quotes.csv')

            final_output_bundled = bundlehandler().bundle_flat_spreading(
                bundled_quotes)

            #quote_df_all.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/quote_dff_all.csv')
            quote_df_all = PriceSpreader().spread_botline_optimal_price(0, 0, 0, final_output_bundled, non_bundled_quotes)
            
            #final_output_bundled.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/final_output_bundled.csv')
                
            quote_df_out, total_deal_stats, PricePoint_data = Reporter(
            ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
                     ComRevDivCd_orgi2, final_output_fixed, quote_df_all)
            
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
        
        quote_df_out = quote_df_out.drop_duplicates(
                subset=['Componentid'], keep='first')
        quote_df_out.reset_index(drop=True, inplace=True)
        
        quote_df_out1 = pd.DataFrame(quote_df_out, columns =[
            'QuoteID', 'Countrycode', 'ChannelID', 'CustomerNumber',
            'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4',
            'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup',
            'ComFamily', 'ComMT', 'ComMTM', 'Year', 'Month', 'EndOfQtr',
            'ClientSegCd', 'ClientSeg_E', 'ComLogListPrice', 'Indirect_1_0',
            'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL',
            'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice',
            'DealSize', 'LogDealSize', 'ComPctContrib', 'ModelID',
            'RequestingApplicationID', 'Version', 'Componentid', 'TreeNode',
            'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL',
            'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice',
            'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb',
            'OptimalPriceGP', 'OptimalPriceExpectedGP',
            'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh',
            'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice',
            'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP',
            'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice',
            'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL',
            'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice',
            'COPOptimalPricePofL', 'COPOptimalPriceWinProb',
            'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP',
            'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh',
            'COPQuotePriceWinProb', 'COPQuotePriceGP',
            'COPQuotePriceExpectedGP'
        ])        
              
        quote_df_out_response = Df2XMLConverter().df_to_xml(
            quote_df_out1, total_deal_stats, PricePoint_data,
            'quote_df_out_response.xml')
    
        quote_df_out.to_csv('C:\\Model_Factory_JP\\CSV\\quote_out.csv')
        return quote_df_out_response
