# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:02:37 2017
@author: Sai
"""

import unittest
import pandas as pd
from fixedproddetector import FixedProdDetector
from pandas.util.testing import assert_frame_equal


class TestFixedProdDetector(unittest.TestCase):
    """
    Class to unit test the FixedProdDetector class
    """

    def test_fixed_prod_detector_NA(self):
        """
        Test that the dataframe read in equals what you expect
        """

        quote_df = pd.DataFrame(columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP', 'COPQuotePriceExpectedGP'],
                            data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '3403a', '3403-M01a', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], ['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '8247a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']])
        quote_df1 = pd.DataFrame(
            columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss',
                     'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM',
                     'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL',
                     'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize',
                     'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL',
                     'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice',
                     'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP',
                     'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh',
                     'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb',
                     'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice',
                     'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL',
                     'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb',
                     'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow',
                     'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP',
                     'COPQuotePriceExpectedGP'],
            data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM',
                   'POWER LINUX', '3403a', '3403-M01a', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728,
                   0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1,
                   '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210',
                   1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333,
                   0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859,
                   31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524,
                   16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                  ['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM',
                   'POWER LINUX', '8247a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728,
                   0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1,
                   '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210',
                   1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333,
                   0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859,
                   31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524,
                   16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']])
        discount_df = pd.DataFrame(columns=['Condition', 'DiscountedPrice', 'Discount', 'Comments'], data=[["df.loc[df.ComMTM == '3403-M01a']", "item['ComTMC']*0.9", 0.1, "TMC GP 10%"]])

        fixed_df = pd.DataFrame(columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP', 'COPQuotePriceExpectedGP', 'DiscountedPrice'],
                             data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '3403a', '3403-M01a', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 22582.685493]])

        non_fixed_df = pd.DataFrame(
            columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss',
                     'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM',
                     'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL',
                     'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize',
                     'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL',
                     'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice',
                     'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP',
                     'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh',
                     'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb',
                     'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice',
                     'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL',
                     'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb',
                     'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow',
                     'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP',
                     'COPQuotePriceExpectedGP'],
            data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM',
                   'POWER LINUX', '8247a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728,
                   0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1,
                   '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210',
                   1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333,
                   0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859,
                   31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524,
                   16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']])
        empty_discount_df = pd.DataFrame(columns=['Condition', 'DiscountedPrice', 'Discount', 'Comments'])
        geomap = pd.DataFrame()

        fixed_quotes, non_fixed_quotes = FixedProdDetector().split_NA(discount_df, quote_df, geomap)
        fixed_quotes = fixed_quotes.reset_index()
        non_fixed_quotes = non_fixed_quotes.reset_index()

        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        assert_frame_equal(fixed_df.reindex_axis(sorted(fixed_df.columns),
                           axis=1),
                           fixed_quotes.reindex_axis(sorted(fixed_df.columns),
                           axis=1))

        assert_frame_equal(non_fixed_df.reindex_axis(sorted(non_fixed_df.columns),
                           axis=1),
                           non_fixed_quotes.reindex_axis(sorted(non_fixed_df.columns),
                           axis=1))

        fixed_df = pd.DataFrame(
            columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss',
                     'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM',
                     'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL',
                     'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize',
                     'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL',
                     'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice',
                     'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP',
                     'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh',
                     'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb',
                     'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice',
                     'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL',
                     'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb',
                     'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow',
                     'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP',
                     'COPQuotePriceExpectedGP', 'DiscountedPrice'])

        fixed_df = fixed_df.reset_index()
        fixed_quotes, non_fixed_quotes = FixedProdDetector().split_NA(empty_discount_df, quote_df, geomap)
        fixed_quotes = fixed_quotes.reset_index()

        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        assert_frame_equal(fixed_df.reindex_axis(sorted(fixed_df.columns),
                                             axis=1),
                       fixed_quotes.reindex_axis(sorted(fixed_df.columns),
                                                 axis=1))

        assert_frame_equal(non_fixed_quotes.reindex_axis(sorted(non_fixed_quotes.columns),
                                                 axis=1),
                       quote_df.reindex_axis(sorted(non_fixed_quotes.columns),
                                                     axis=1))


    def test_fixed_prod_detector_EMEA(self):
        """
        Test that the dataframe read in equals what you expect
        """

        quote_df = pd.DataFrame(columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP', 'COPQuotePriceExpectedGP'],
                            data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '8247a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], ['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '8248a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']])

        discount_df = pd.DataFrame(columns=['Condition', 'DiscountedPrice', 'Discount', 'Comments'], data=[["df.loc[df.ComMT.isin(['8248a'])]", "item['ComListPrice']*0.77", 0.23, 'Discount 23%']])

        fixed_df = pd.DataFrame(columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP', 'COPQuotePriceExpectedGP', 'DiscountedPrice'],
                             data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '8248a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 51658.7702554]])

        non_fixed_df = pd.DataFrame(columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP', 'COPQuotePriceExpectedGP'],
                             data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '8247a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.3120225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']])
        geomap = pd.DataFrame()


        empty_discount_df = pd.DataFrame(columns=['Condition', 'DiscountedPrice', 'Discount', 'Comments'])



        fixed_quotes, non_fixed_quotes = FixedProdDetector().split_EMEA(discount_df, quote_df, geomap)
        fixed_quotes = fixed_quotes.reset_index()
        non_fixed_quotes = non_fixed_quotes.reset_index()

        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        assert_frame_equal(fixed_df.reindex_axis(sorted(fixed_df.columns),
                           axis=1),
                           fixed_quotes.reindex_axis(sorted(fixed_df.columns),
                           axis=1))

        assert_frame_equal(non_fixed_df.reindex_axis(sorted(non_fixed_df.columns),
                           axis=1),
                           non_fixed_quotes.reindex_axis(sorted(non_fixed_df.columns),
                           axis=1))

        fixed_df = pd.DataFrame(
            columns=['Countrycode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss',
                     'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM',
                     'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL',
                     'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize',
                     'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL',
                     'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice',
                     'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP',
                     'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh',
                     'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb',
                     'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice',
                     'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL',
                     'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb',
                     'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow',
                     'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP',
                     'COPQuotePriceExpectedGP', 'DiscountedPrice'])

        fixed_df = fixed_df.reset_index()
        fixed_quotes, non_fixed_quotes = FixedProdDetector().split_EMEA(empty_discount_df, quote_df, geomap)
        fixed_quotes = fixed_quotes.reset_index()
        """
               reindex columns to ensure that columns are sorted and comparison
               doesn't result in failure
               """

        assert_frame_equal(fixed_df.reindex_axis(sorted(fixed_df.columns),
                                                 axis=1),
                           fixed_quotes.reindex_axis(sorted(fixed_df.columns),
                                                     axis=1))

        assert_frame_equal(non_fixed_quotes.reindex_axis(sorted(non_fixed_quotes.columns),
                                                         axis=1),
                           quote_df.reindex_axis(sorted(non_fixed_quotes.columns),
                                                 axis=1))
