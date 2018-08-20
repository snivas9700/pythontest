# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 05:32:15 2018

@author: harish
"""

import pandas as pd
import copy
import numpy as np
import BasePricingFunctions as BPF

def validator_conditions(row):
        """
        This function decides which condition need to be applied for each validator components
        Input : Each row of validator dataframe
        Output : Condition Type (String)
        """
        
        row['max_price'] = int(row['max_price'])
        row['min_price'] = int(row['min_price'])
        
        if (row['max_price'] != -999 and row['min_price'] == -999):
            condition = 'validate_max'
        if (row['max_price'] != -999 and row['min_price'] != -999):
            condition = 'validate_fixed'
        if (row['max_price'] == -999 and row['min_price'] != -999):
            condition = 'validate_min'
        
        return condition
    

def validator_updates(fixed_quotes):
        """
        This function updates the values of each row of validator dataframe. 
              First by applying the validator conditions
              And then recalculates the values based on the new values
        Input : Each row of validator dataframe
        Output : Updated row
        """
        
        #Store original OP value
        OP_Orig = fixed_quotes['OptimalPrice']
        
        #Apply the capping rules based on the validator rules
        if (fixed_quotes.validator_type == 'validate_min'):
            # If High, Optimal and Low > min_price no adjustment needed
            test_cond = fixed_quotes['AdjComLowPrice'] > fixed_quotes['min_price'] and fixed_quotes['AdjComHighPrice'] > fixed_quotes['min_price'] and fixed_quotes['OptimalPrice'] > fixed_quotes['min_price']
            if ( test_cond ):
                fixed_quotes['AdjComLowPrice'] = fixed_quotes['AdjComLowPrice']  # No change required
            else:
                fixed_quotes['AdjComLowPrice'] = fixed_quotes['min_price']
            fixed_quotes['AdjComHighPrice'] = fixed_quotes['AdjComHighPrice'] 
            if (fixed_quotes['AdjComHighPrice'] <= fixed_quotes['AdjComLowPrice'] ):
                fixed_quotes['AdjComHighPrice'] = fixed_quotes['AdjComLowPrice'] + 2
        elif (fixed_quotes.validator_type == 'validate_max'):
            fixed_quotes['AdjComHighPrice'] = fixed_quotes['max_price'] 
            fixed_quotes['AdjComLowPrice'] = fixed_quotes['AdjComLowPrice']
            if (fixed_quotes['AdjComLowPrice'] >= fixed_quotes['AdjComHighPrice'] ):
                fixed_quotes['AdjComLowPrice'] = fixed_quotes['AdjComHighPrice'] - 2
        elif (fixed_quotes.validator_type == 'validate_fixed'):
            fixed_quotes['AdjComHighPrice'] = fixed_quotes['max_price'] 
            fixed_quotes['AdjComLowPrice'] = fixed_quotes['min_price']
        
        #To handle scenario when AdjComHighPrice and AdjComLowPrice price are same
        if (fixed_quotes['AdjComHighPrice'] == fixed_quotes['AdjComLowPrice'] ):
            if ( ( fixed_quotes['AdjComLowPrice'] == fixed_quotes['ComListPrice'] ) | (fixed_quotes['Countrycode'] == 'JP') ) :
                fixed_quotes['AdjComLowPrice'] = fixed_quotes['AdjComLowPrice']  - 1
            fixed_quotes['AdjComHighPrice'] = fixed_quotes['AdjComLowPrice'] + 2
            fixed_quotes['AdjComMedPrice']  = fixed_quotes['AdjComLowPrice'] + 1
            fixed_quotes['OptimalPrice']  = fixed_quotes['AdjComLowPrice'] + 1
            fixed_quotes['DealBotLineSpreadOptimalPrice'] = fixed_quotes['AdjComLowPrice'] + 1
            
        
        #Adjust Optimal rules to respect the rules
        if (fixed_quotes['OptimalPrice'] >= fixed_quotes['AdjComHighPrice']):
                fixed_quotes['OptimalPrice'] = fixed_quotes['AdjComHighPrice'] - 1.0
        elif (fixed_quotes['OptimalPrice'] <= fixed_quotes['AdjComLowPrice']):
                fixed_quotes['OptimalPrice'] = fixed_quotes['AdjComLowPrice'] + 1.0
        
        #Adjust Optimal rules to respect the rules
        if (fixed_quotes['AdjComMedPrice'] >= fixed_quotes['AdjComHighPrice']):
                fixed_quotes['AdjComMedPrice'] = fixed_quotes['AdjComHighPrice'] - 1.0
        elif (fixed_quotes['AdjComMedPrice'] <= fixed_quotes['AdjComLowPrice']):
                fixed_quotes['AdjComMedPrice'] = fixed_quotes['AdjComLowPrice'] + 1.0
        
        #Adjustments for all possible recomputed values
        fixed_quotes['ComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc['ComListPrice']
        fixed_quotes['ComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc['ComListPrice']
        fixed_quotes['ComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc['ComListPrice']
        fixed_quotes['ComMedPrice'] = fixed_quotes['AdjComMedPrice']

        fixed_quotes['AdjComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc['ComListPrice']
        fixed_quotes['AdjComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc['ComListPrice']
        fixed_quotes['AdjComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc['ComListPrice']
        fixed_quotes['OptimalPricePofL'] = fixed_quotes['OptimalPrice'] / fixed_quotes.loc['ComListPrice']

        # Make use of probability computed by engine
        fixed_quotes['OptimalPriceGP'] = fixed_quotes['OptimalPrice'] - fixed_quotes.loc['ComTMC']
        fixed_quotes['OptimalPriceExpectedGP'] = fixed_quotes['OptimalPriceGP'] * fixed_quotes['OptimalPriceWinProb']

        
        #Adjustments for OptimalPriceIntervalLow & OptimalPriceIntervalHigh
        #Call this only if OP has been changed or when OP is oustide interval ranges
        if ( fixed_quotes['OptimalPrice'] != OP_Orig or (fixed_quotes['OptimalPrice'] < fixed_quotes['OptimalPriceIntervalLow'] or fixed_quotes['OptimalPrice'] > fixed_quotes['OptimalPriceIntervalHigh'])):
            fixed_quotes['OptimalPriceIntervalLow'], fixed_quotes['OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(fixed_quotes['OptimalPrice'], fixed_quotes['AdjComLowPrice'], fixed_quotes['AdjComMedPrice'], fixed_quotes['AdjComHighPrice'], fixed_quotes['ComTMC'])
        
        #Adjust OptimalPriceIntervalHigh and OptimalPriceIntervalLow price
        if ( (fixed_quotes['OptimalPriceIntervalHigh'] > fixed_quotes['AdjComHighPrice']) or (fixed_quotes['OptimalPriceIntervalHigh'] <= fixed_quotes['AdjComLowPrice'])):
                fixed_quotes['OptimalPriceIntervalHigh'] = fixed_quotes['AdjComHighPrice']
        if ( (fixed_quotes['OptimalPriceIntervalLow'] >= fixed_quotes['AdjComHighPrice']) or (fixed_quotes['OptimalPriceIntervalLow'] < fixed_quotes['AdjComLowPrice'])):
                fixed_quotes['OptimalPriceIntervalLow'] = fixed_quotes['AdjComLowPrice']
                
        #Adjust DealBotLineSpreadOptimalPrice to respect the rules
        if (fixed_quotes['DealBotLineSpreadOptimalPrice'] >= fixed_quotes['OptimalPriceIntervalHigh']):
                fixed_quotes['DealBotLineSpreadOptimalPrice'] = fixed_quotes['OptimalPriceIntervalHigh'] - 1.0
        elif (fixed_quotes['DealBotLineSpreadOptimalPrice'] <= fixed_quotes['OptimalPriceIntervalLow']):
                fixed_quotes['DealBotLineSpreadOptimalPrice'] = fixed_quotes['OptimalPriceIntervalLow'] + 1.0
        
        return fixed_quotes
        
        
class HWValidProdHandler:
    """Provide optimal prices info for fixed-discounted products.

    """

    def __init__(self):
        pass
        
    
    def apply_rules(self, validator_quotes):
        """
        This function is to apply validation rules on the validator components and update the values
        :param df: output pandas.df (validator part) from HWValidator
        :return a new pandas.df
        """

        if validator_quotes.empty:
            validator_quotes_out = copy.deepcopy(validator_quotes)
            return validator_quotes_out
        
        #Identify the rules to be applied
        validator_quotes['validator_type']   =   validator_quotes.apply(validator_conditions,axis=1)
        
        
        #Adjust Optimal , High & Low values based on the validator rules
        validator_quotes  =   validator_quotes.apply(validator_updates,axis=1)
        
        #del validator_quotes['min_price']
        #del validator_quotes['max_price']
        
        '''
        For Defect 1697177, OptimalPrice was coming greater then ComListPrice for Fixed Discount Products,
        So, to resolve this, done capping which will equate OptimalPrice and ComListPrice
        whenever OptimalPrice > ComListPrice. 
        '''
        validator_quotes['OptimalPrice'] = np.where((validator_quotes['OptimalPrice'] > validator_quotes['ComListPrice'])
                     , validator_quotes['ComListPrice'], validator_quotes['OptimalPrice'])
        
        #Capping for DealBotLineSpreadOptimalPrice when it is higher than List Price
        validator_quotes['DealBotLineSpreadOptimalPrice'] = np.where((validator_quotes['DealBotLineSpreadOptimalPrice'] > validator_quotes['ComListPrice'])
                     , validator_quotes['ComListPrice'], validator_quotes['DealBotLineSpreadOptimalPrice'])
       
        validator_quotes_out = copy.deepcopy(validator_quotes)

        return validator_quotes_out
