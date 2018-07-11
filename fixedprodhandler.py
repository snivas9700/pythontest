import pandas as pd
import copy
import numpy as np
import BasePricingFunctions as BPF


class FixedProdHandler:
    """Provide optimal prices info for fixed-discounted products.

    """

    def __init__(self):
        pass

    def process_EMEA(self, fixed_quotes):
        """
        :param df: output pandas.df (fixed part) from FixedProdDetector
        :return a new pandas.df
        """

        if fixed_quotes.empty:
            fixed_quotes_out = copy.deepcopy(fixed_quotes)
            return fixed_quotes_out

        # The AdjComLowPrice, Med, and High prices are set to same value as the discounted_price as the discounts are fixed.
        fixed_quotes['AdjComLowPrice'] = fixed_quotes['DiscountedPrice']
        fixed_quotes['AdjComMedPrice'] = fixed_quotes[
                                             'DiscountedPrice'] + 1  # make sure AdjComLowPrice is less than AdjComMedPrice
        fixed_quotes['AdjComHighPrice'] = fixed_quotes[
                                              'DiscountedPrice'] + 2  # make sure AdjComHighPrice is greater than AdjComMedPrice

        # optimal_price is same as the discounted_price
        fixed_quotes['OptimalPrice'] = fixed_quotes['AdjComMedPrice']  # make sure OptimalPrice equals to AdjComMedPrice
        fixed_quotes['ComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComMedPrice'] = fixed_quotes['AdjComMedPrice']

        # The tree node has no significance as we are not using pricing engine and segmentation
        # setting to a unique number -999 just to indicate that this is not a number set by engine
        fixed_quotes['TreeNode'] = -999
        fixed_quotes['ComTMCPofL'] = fixed_quotes.loc[:, 'ComTMC'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['OptimalPricePofL'] = fixed_quotes['OptimalPrice'] / fixed_quotes.loc[:, 'ComListPrice']

        # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
        fixed_quotes['OptimalPriceWinProb'] = 0.5
        fixed_quotes['OptimalPriceGP'] = fixed_quotes['OptimalPrice'] - fixed_quotes.loc[:, 'ComTMC']
        fixed_quotes['OptimalPriceExpectedGP'] = fixed_quotes['OptimalPriceGP'] * 0.5

        # setting intervalLow and intervalHigh also as same as discounted_price because this is a fixed discount anyways
        fixed_quotes['OptimalPriceIntervalLow'] = fixed_quotes['OptimalPrice']
        fixed_quotes['OptimalPriceIntervalHigh'] = fixed_quotes['OptimalPrice']
        fixed_quotes['DealBotLineSpreadOptimalPrice'] = fixed_quotes['OptimalPrice']
        fixed_quotes['QuotePrice'] = fixed_quotes['ComQuotePrice']
        fixed_quotes['QuotePricePofL'] = fixed_quotes['ComQuotePricePofL']

        # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
        fixed_quotes['QuotePriceWinProb'] = 0.5
        fixed_quotes['QuotePriceGP'] = fixed_quotes['QuotePrice'] - fixed_quotes.loc[:, 'ComTMC']
        fixed_quotes['QuotePriceExpectedGP'] = fixed_quotes['QuotePriceGP'] * 0.5
        fixed_quotes['PredictedQuotePricePofL'] = fixed_quotes['QuotePrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['PredictedQuotePrice'] = fixed_quotes['QuotePrice']

        del fixed_quotes['DiscountedPrice']
        
        '''
        For Defect 1697177, OptimalPrice was coming greater then ComListPrice for Fixed Discount Products,
        So, to resolve this, done capping which will equate OptimalPrice and ComListPrice
        whenever OptimalPrice > ComListPrice. 
        '''
        fixed_quotes['OptimalPrice'] = np.where((fixed_quotes['OptimalPrice'] > fixed_quotes['ComListPrice'])
                     , fixed_quotes['ComListPrice'], fixed_quotes['OptimalPrice'])
        
        '''
        Since, after doing the above hotfix OptimalPrice was not coming in OptimalPriceIntervalLow and 
        OptimalPriceIntervalHigh, so applied the below hotfix to recalculate the Confidence Intervals.
        DealBotLineSpreadOptimalPrice for Fixed Discount Products should always be equal to Optimal Price,
        thats'why below we have equated both. 
        '''
        
        fixed_quotes.index = range(len(fixed_quotes))
        for i in range(len(fixed_quotes)):
            fixed_quotes.loc[i, 'OptimalPriceIntervalLow'], fixed_quotes.loc[i, 'OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(fixed_quotes.loc[i, 'OptimalPrice'], fixed_quotes.loc[i,'AdjComLowPrice'], fixed_quotes.loc[i,'AdjComMedPrice'], fixed_quotes.loc[i,'AdjComHighPrice'], fixed_quotes.loc[i, 'ComTMC'])
            fixed_quotes.loc[i,'DealBotLineSpreadOptimalPrice'] = fixed_quotes.loc[i, 'OptimalPrice']

        fixed_quotes_out = copy.deepcopy(fixed_quotes)

        return fixed_quotes_out


    def process_NA(self, fixed_quotes):
        """
        :param df: output pandas.df (fixed part) from FixedProdDetector
        :retur a new pandas.df
        """

        if fixed_quotes.empty:
            fixed_quotes_out = copy.deepcopy(fixed_quotes)
            return fixed_quotes_out

        # The AdjComLowPrice, Med, and High prices are set to same value as the discounted_price as the discounts are fixed.
        #fixed_quotes.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/fixed_quotes_out_fixedprod_Start_83.csv')
        fixed_quotes['AdjComLowPrice'] = fixed_quotes['DiscountedPrice']
        fixed_quotes['AdjComMedPrice'] = fixed_quotes[
                                             'DiscountedPrice'] + 1  # make sure AdjComLowPrice is less than AdjComMedPrice
        fixed_quotes['AdjComHighPrice'] = fixed_quotes[
                                              'DiscountedPrice'] + 2  # make sure AdjComHighPrice is greater than AdjComMedPrice

        # optimal_price is same as the discounted_price
        fixed_quotes['OptimalPrice'] = fixed_quotes['AdjComMedPrice']  # make sure OptimalPrice equals to AdjComMedPrice
        fixed_quotes['ComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComMedPrice'] = fixed_quotes['AdjComMedPrice']

        # The tree node has no significance as we are not using pricing engine and segmentation
        # setting to a unique number -999 just to indicate that this is not a number set by engine
        fixed_quotes['TreeNode'] = -999
        fixed_quotes['ComTMCPofL'] = fixed_quotes.loc[:, 'ComTMC'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['OptimalPricePofL'] = fixed_quotes['OptimalPrice'] / fixed_quotes.loc[:, 'ComListPrice']

        # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
        fixed_quotes['OptimalPriceWinProb'] = 0.5
        fixed_quotes['OptimalPriceGP'] = fixed_quotes['OptimalPrice'] - fixed_quotes.loc[:, 'ComTMC']
        fixed_quotes['OptimalPriceExpectedGP'] = fixed_quotes['OptimalPriceGP'] * 0.5

        # setting intervalLow and intervalHigh also as same as discounted_price because this is a fixed discount anyways
        fixed_quotes['OptimalPriceIntervalLow'] = fixed_quotes['OptimalPrice']
        fixed_quotes['OptimalPriceIntervalHigh'] = fixed_quotes['OptimalPrice']
        fixed_quotes['DealBotLineSpreadOptimalPrice'] = fixed_quotes['OptimalPrice']
        fixed_quotes['QuotePrice'] = fixed_quotes['ComQuotePrice']
        fixed_quotes['QuotePricePofL'] = fixed_quotes['ComQuotePricePofL']

        # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
        fixed_quotes['QuotePriceWinProb'] = 0.5
        fixed_quotes['QuotePriceGP'] = fixed_quotes['QuotePrice'] - fixed_quotes.loc[:, 'ComTMC']
        fixed_quotes['QuotePriceExpectedGP'] = fixed_quotes['QuotePriceGP'] * 0.5
        fixed_quotes['PredictedQuotePricePofL'] = fixed_quotes['QuotePrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['PredictedQuotePrice'] = fixed_quotes['QuotePrice']

        del fixed_quotes['DiscountedPrice']
        
        '''
        For Defect 1697177, OptimalPrice was coming greater then ComListPrice for Fixed Discount Products,
        So, to resolve this, done capping which will equate OptimalPrice and ComListPrice
        whenever OptimalPrice > ComListPrice. 
        '''
        fixed_quotes['OptimalPrice'] = np.where((fixed_quotes['OptimalPrice'] > fixed_quotes['ComListPrice'])
                     , fixed_quotes['ComListPrice'], fixed_quotes['OptimalPrice']) 
        
        '''
        Since, after doing the above hotfix OptimalPrice was not coming in OptimalPriceIntervalLow and 
        OptimalPriceIntervalHigh, so applied the below hotfix to recalculate the Confidence Intervals.
        DealBotLineSpreadOptimalPrice for Fixed Discount Products should always be equal to Optimal Price,
        thats'why below we have equated both. 
        '''
        
        fixed_quotes.index = range(len(fixed_quotes))
        for i in range(len(fixed_quotes)):
            fixed_quotes.loc[i, 'OptimalPriceIntervalLow'], fixed_quotes.loc[i, 'OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(fixed_quotes.loc[i, 'OptimalPrice'], fixed_quotes.loc[i,'AdjComLowPrice'], fixed_quotes.loc[i,'AdjComMedPrice'], fixed_quotes.loc[i,'AdjComHighPrice'], fixed_quotes.loc[i, 'ComTMC'])
            fixed_quotes.loc[i,'DealBotLineSpreadOptimalPrice'] = fixed_quotes.loc[i, 'OptimalPrice']
        
        
        fixed_quotes_out = copy.deepcopy(fixed_quotes)
        #fixed_quotes_out.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/fixed_quotes_out_fixedprod_127_last.csv')

        return fixed_quotes_out
    
    def process_JP(self, fixed_quotes):
        """
        :param df: output pandas.df (fixed part) from FixedProdDetector
        :return a new pandas.df
        """

        if fixed_quotes.empty:
            fixed_quotes_out = copy.deepcopy(fixed_quotes)
            return fixed_quotes_out

        # The AdjComLowPrice, Med, and High prices are set to same value as the discounted_price as the discounts are fixed.
        fixed_quotes['AdjComLowPrice'] = fixed_quotes['DiscountedPrice']
        fixed_quotes['AdjComMedPrice'] = fixed_quotes[
                                             'DiscountedPrice'] + 1  # make sure AdjComLowPrice is less than AdjComMedPrice
        fixed_quotes['AdjComHighPrice'] = fixed_quotes[
                                              'DiscountedPrice'] + 2  # make sure AdjComHighPrice is greater than AdjComMedPrice

        # optimal_price is same as the discounted_price
        fixed_quotes['OptimalPrice'] = fixed_quotes['AdjComMedPrice']  # make sure OptimalPrice equals to AdjComMedPrice
        fixed_quotes['ComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['ComMedPrice'] = fixed_quotes['AdjComMedPrice']

        # The tree node has no significance as we are not using pricing engine and segmentation
        # setting to a unique number -999 just to indicate that this is not a number set by engine
        fixed_quotes['TreeNode'] = -999
        fixed_quotes['ComTMCPofL'] = fixed_quotes.loc[:, 'ComTMC'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComLowPofL'] = fixed_quotes['AdjComLowPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComMedPofL'] = fixed_quotes['AdjComMedPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['AdjComHighPofL'] = fixed_quotes['AdjComHighPrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['OptimalPricePofL'] = fixed_quotes['OptimalPrice'] / fixed_quotes.loc[:, 'ComListPrice']

        # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
        fixed_quotes['OptimalPriceWinProb'] = 0.5
        fixed_quotes['OptimalPriceGP'] = fixed_quotes['OptimalPrice'] - fixed_quotes.loc[:, 'ComTMC']
        fixed_quotes['OptimalPriceExpectedGP'] = fixed_quotes['OptimalPriceGP'] * 0.5

        # setting intervalLow and intervalHigh also as same as discounted_price because this is a fixed discount anyways
        fixed_quotes['OptimalPriceIntervalLow'] = fixed_quotes['OptimalPrice']
        fixed_quotes['OptimalPriceIntervalHigh'] = fixed_quotes['OptimalPrice']
        fixed_quotes['DealBotLineSpreadOptimalPrice'] = fixed_quotes['OptimalPrice']
        fixed_quotes['QuotePrice'] = fixed_quotes['ComQuotePrice']
        fixed_quotes['QuotePricePofL'] = fixed_quotes['ComQuotePricePofL']

        # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
        fixed_quotes['QuotePriceWinProb'] = 0.5
        fixed_quotes['QuotePriceGP'] = fixed_quotes['QuotePrice'] - fixed_quotes.loc[:, 'ComTMC']
        fixed_quotes['QuotePriceExpectedGP'] = fixed_quotes['QuotePriceGP'] * 0.5
        fixed_quotes['PredictedQuotePricePofL'] = fixed_quotes['QuotePrice'] / fixed_quotes.loc[:, 'ComListPrice']
        fixed_quotes['PredictedQuotePrice'] = fixed_quotes['QuotePrice']

        del fixed_quotes['DiscountedPrice']
        
        '''
        For Defect 1697177, OptimalPrice was coming greater then ComListPrice for Fixed Discount Products,
        So, to resolve this, done capping which will equate OptimalPrice and ComListPrice
        whenever OptimalPrice > ComListPrice. 
        '''
        fixed_quotes['OptimalPrice'] = np.where((fixed_quotes['OptimalPrice'] > fixed_quotes['ComListPrice'])
                     , fixed_quotes['ComListPrice'], fixed_quotes['OptimalPrice'])
        
        '''
        Since, after doing the above hotfix OptimalPrice was not coming in OptimalPriceIntervalLow and 
        OptimalPriceIntervalHigh, so applied the below hotfix to recalculate the Confidence Intervals.
        DealBotLineSpreadOptimalPrice for Fixed Discount Products should always be equal to Optimal Price,
        thats'why below we have equated both. 
        '''
        
        fixed_quotes.index = range(len(fixed_quotes))
        for i in range(len(fixed_quotes)):
            fixed_quotes.loc[i, 'OptimalPriceIntervalLow'], fixed_quotes.loc[i, 'OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(fixed_quotes.loc[i, 'OptimalPrice'], fixed_quotes.loc[i,'AdjComLowPrice'], fixed_quotes.loc[i,'AdjComMedPrice'], fixed_quotes.loc[i,'AdjComHighPrice'], fixed_quotes.loc[i, 'ComTMC'])
            fixed_quotes.loc[i,'DealBotLineSpreadOptimalPrice'] = fixed_quotes.loc[i, 'OptimalPrice']

        fixed_quotes_out = copy.deepcopy(fixed_quotes)

        return fixed_quotes_out
