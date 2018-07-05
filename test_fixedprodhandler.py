
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 16:53:25 2017
@author: Sai
"""

import unittest
import pandas as pd
import fixedprodhandler as fp
from pandas.util.testing import assert_frame_equal, assert_series_equal


class fixedProdHandlerTester(unittest.TestCase):
    """
    Class to unit test the FixedProdHandler class
    """

    def test_fixed_prod_handler_EMEA(self):
        """
        Test that the dataframe read in equals what you expect
        """

        input_df = pd.DataFrame(columns=['CountryCode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL' , 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP', 'COPQuotePriceExpectedGP', 'DiscountedPrice'],
                             data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '8248a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.312 0225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 51658.7702554]])

        expected_df = pd.DataFrame(columns=[u'CountryCode', u'Quantity', u'ComListPrice', u'ComTMC', u'ComQuotePrice', u'ComDelgPriceL4', u'WinLoss', u'ComRevCat', u'ComRevDivCd', u'ComBrand', u'ComGroup', u'ComFamily', u'ComMT', u'ComMTM', u'ComLogListPrice', u'UpgMES', u'ComQuotePricePofL', u'ComDelgPriceL4PofL', u'ComCostPofL', u'ComLowPofL', u'ComMedPofL', u'ComHighPofL', u'ComMedPrice', u'DealSize', u'LogDealSize', u'ComPctContrib', u'Componentid', u'concat', u'TreeNode', u'ComTMCPofL', u'AdjComLowPofL', u'AdjComMedPofL', u'AdjComHighPofL', u'AdjComLowPrice', u'AdjComMedPrice', u'AdjComHighPrice', u'OptimalPricePofL', u'OptimalPrice', u'OptimalPriceWinProb', u'OptimalPriceGP', u'OptimalPriceExpectedGP', u'OptimalPriceIntervalLow', u'OptimalPriceIntervalHigh', u'DealBotLineSpreadOptimalPrice', u'QuotePricePofL', u'QuotedPrice', u'QuotePriceWinProb', u'QuotePriceGP', u'QuotePriceExpectedGP', u'PredictedQuotePricePofL', u'PredictedQuotePrice', u'COPComLowPrice', u'COPComMedPrice', u'COPComHighPrice', u'COPComLowPofL', 'COPComMedPofL', u'COPComHighPofL', u'COPOptimalPrice', u'COPOptimalPricePofL', u'COPOptimalPriceWinProb', u'COPOptimalPriceGP', u'COPOptimalPriceExpectedGP', u'COPOptimalPriceIntervalLow', u'COPOptimalPriceIntervalHigh', u'COPQuotePriceWinProb', u'COPQuotePriceGP', u'COPQuotePriceExpectedGP', u'QuotePrice'],
                             data = [['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '8248a', '8247-22La', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.77, 0.7700149055038708, 0.7700298110077415, 51659.7702554, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.312 0225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', -999, 0.3740070066975774, 0.77, 0.7700149055038708, 0.7700298110077415, 51658.7702554, 51659.7702554, 51660.7702554, 0.7700149055038708, 51659.7702554, 0.5, 26567.897485399997, 13283.948742699999, 51659.7702554, 51659.7702554, 51659.7702554, 0.05805849, 3895.10412, 0.5, -21196.76865, -10598.384325, 0.05805848953763053, 3895.10412, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 3895.10412]])

        test_result = fp.FixedProdHandler().process_EMEA(input_df)
        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        assert_frame_equal(
            expected_df.reindex_axis(sorted(expected_df.columns), axis=1),
            test_result.reindex_axis(sorted(expected_df.columns), axis=1))

    def test_fixed_prod_handler_NA(self):
        """
        Test that the dataframe read in equals what you expect
        """

        input_df = pd.DataFrame(columns=['CountryCode', 'Quantity', 'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComLogListPrice', 'UpgMES', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL' , 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'Componentid', 'concat', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow', 'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice', 'QuotePricePofL', 'QuotedPrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow', 'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP', 'COPQuotePriceExpectedGP', 'DiscountedPrice'],
                             data=[['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '3403a', '3403-M01a', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.070366728, 0.219135802, 0.498540279, 14701.67023, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.312 0225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', 1, 0.374007007, 0.073181397, 0.258364382, 0.518481891, 4909.68956, 17333.48862, 34784.59333, 0.470925885, 31594.09365, 0.082751645, 6502.220882, 538.0694737, 29805.30051, 33748.30859, 31594.65786, 0.05805849, 3895.10412, 0.964382112, -21196.76865, -20441.78452, 0.247868524, 16629.32876, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 51658.7702554]])

        expected_df = pd.DataFrame(columns=[u'CountryCode', u'Quantity', u'ComListPrice', u'ComTMC', u'ComQuotePrice', u'ComDelgPriceL4', u'WinLoss', u'ComRevCat', u'ComRevDivCd', u'ComBrand', u'ComGroup', u'ComFamily', u'ComMT', u'ComMTM', u'ComLogListPrice', u'UpgMES', u'ComQuotePricePofL', u'ComDelgPriceL4PofL', u'ComCostPofL', u'ComLowPofL', u'ComMedPofL', u'ComHighPofL', u'ComMedPrice', u'DealSize', u'LogDealSize', u'ComPctContrib', u'Componentid', u'concat', u'TreeNode', u'ComTMCPofL', u'AdjComLowPofL', u'AdjComMedPofL', u'AdjComHighPofL', u'AdjComLowPrice', u'AdjComMedPrice', u'AdjComHighPrice', u'OptimalPricePofL', u'OptimalPrice', u'OptimalPriceWinProb', u'OptimalPriceGP', u'OptimalPriceExpectedGP', u'OptimalPriceIntervalLow', u'OptimalPriceIntervalHigh', u'DealBotLineSpreadOptimalPrice', u'QuotePricePofL', u'QuotedPrice', u'QuotePriceWinProb', u'QuotePriceGP', u'QuotePriceExpectedGP', u'PredictedQuotePricePofL', u'PredictedQuotePrice', u'COPComLowPrice', u'COPComMedPrice', u'COPComHighPrice', u'COPComLowPofL', 'COPComMedPofL', u'COPComHighPofL', u'COPOptimalPrice', u'COPOptimalPricePofL', u'COPOptimalPriceWinProb', u'COPOptimalPriceGP', u'COPOptimalPriceExpectedGP', u'COPOptimalPriceIntervalLow', u'COPOptimalPriceIntervalHigh', u'COPQuotePriceWinProb', u'COPQuotePriceGP', u'COPQuotePriceExpectedGP', u'QuotePrice'],
                             data = [['DE', 2, 67089.31202, 25091.87277, 3895.10412, 0, 1, 'H', '9R', 'OTHER POWER', 'POWER SYSTEM', 'POWER LINUX', '3403a', '3403-M01a', 4.826653338, 0, 0.05805849, 0, 0.374007007, 0.77, 0.7700149055038708, 0.7700298110077415, 51659.7702554, 22254.07, 4.34740945, 0.660628381, 1, '4145484-DEDEO724-322769267089.312 0225091.872773895.104120.01HXYZOTHERPOWERPOWERSYSTEMPOWERLINUX8247a8247-22La201770B04.82665333834000.05805848953760.00.374007006698CHW_DEEPRICER210', -999, 0.3740070066975774, 0.77, 0.7700149055038708, 0.7700298110077415, 51658.7702554, 51659.7702554, 51660.7702554, 0.7700149055038708, 51659.7702554, 0.5, 26567.897485399997, 13283.948742699999, 51659.7702554, 51659.7702554, 51659.7702554, 0.05805849, 3895.10412, 0.5, -21196.76865, -10598.384325, 0.05805848953763053, 3895.10412, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 3895.10412]])

        test_result = fp.FixedProdHandler().process_NA(input_df)
        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        assert_frame_equal(
            expected_df.reindex_axis(sorted(expected_df.columns), axis=1),
            test_result.reindex_axis(sorted(expected_df.columns), axis=1))
