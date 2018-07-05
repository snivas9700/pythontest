# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:02:37 2017

@author: Sanket Maheshwari
"""

import unittest
import pandas as pd
import sys, os
import datafieldsfilter as dff
# for testing dataframes
from pandas.util.testing import assert_frame_equal
sys.path.append(os.path.realpath('..'))


class DataFieldsFilterTest(unittest.TestCase):
    """
    Class to unit test the datafieldsfilter class
    """

    def test_fieldsfilter(self):
        """
        Test that the dataframe read in equals what you expect
        """

        inpu = pd.DataFrame(columns=["QuoteID", "ModelID",
                                     "Version", "Countrycode",
                                     "Channeltype", "Channelidcode",
                                     "Year", "Month",
                                     "EndOfQtr", "Indirect",
                                     "CustomerNumber", "CustomerNumber_CCMS",
                                     "ClientSegCd", "ClientSeg",
                                     "CustomerSecName", "CustomerIndustryName",
                                     "Componentid", "HWPlatformid",
                                     "ComMT", "ComMTM", "ComMTMDesc",
                                     "ComMTMDescLocal",
                                     "ComQuotePricePofL", "ComDelgPriceL4PofL",
                                     "ComCostPofL", "ComLogListPrice",
                                     "ComRevDivCd",
                                     "ComCategory", "ComSubCategory",
                                     "UpgMES",
                                     "Quantity", "ComListPrice",
                                     "ComTMC", "ComQuotePrice",
                                     "ComDelgPriceL4", "ComRevCat",
                                     "ComSpclBidCode1", "ComSpclBidCode2",
                                     "ComSpclBidCode3", "ComSpclBidCode4",
                                     "ComSpclBidCode5", "ComSpclBidCode6",
                                     "FeatureId", "FeatureQuantity",
                                     "ComBrand", "ComGroup", "ComFamily",
                                     "DomBuyerGrpID", "DomBuyerGrpName",
                                     "RequestingApplicationID", "PricePointId",
                                     "PricePointName", "Price", 'GEO_CODE'],
                            data=[[5300242, 'CHW_NA', 2, 'NA', 'I',
                                  'J', 2017, 9, 0, 0,
                                   93008, 123456, 'E', 0,
                                   'Public', 'Government', 9, 0,
                                   '8247a', '3584L55',
                                   'TS4500 HD2 Base Frame',
                                   'TS4500 HD2 Base Frame',
                                   0.058058, 0, 0.374007,
                                   4.826653, 'XYZ', 'H', 'P',
                                   4, 1, 16578.5, 1.207031,
                                   4254.044, 1.963244, 'S',
                                   541, 961, 950, 671, 543, 657, 3, 267,
                                   'OTHER POWER', 'STORWIZE V7000',
                                   'V3700 SFF CONTROL',
                                   'DB5023M5', 'TCU FINANCIAL GROUP',
                                   'EPRICER', 45, 'Original Quoted Price',
                                   76297.11987, 'NA']])

        expected = pd.DataFrame(columns=["QuoteID", "ModelID", "Version",
                                         "Countrycode", "Channeltype",
                                         "Channelidcode", "Year", "Month",
                                         "EndOfQtr", "Indirect",
                                         "CustomerNumber",
                                         "CustomerNumber_CCMS",
                                         "ClientSegCd",
                                         "ClientSeg", "CustomerSecName",
                                         "CustomerIndustryName", "Componentid",
                                         "HWPlatformid", "ComMT", "ComMTM",
                                         "ComMTMDesc", "ComMTMDescLocal",
                                         "ComQuotePricePofL",
                                         "ComDelgPriceL4PofL",
                                         "ComCostPofL", "ComLogListPrice",
                                         "ComRevDivCd",
                                         "ComCategory", "ComSubCategory",
                                         "UpgMES", "Quantity", "ComListPrice",
                                         "ComTMC", "ComQuotePrice",
                                         "ComDelgPriceL4", "ComRevCat",
                                         "ComSpclBidCode1", "ComSpclBidCode2",
                                         "ComSpclBidCode3", "ComSpclBidCode4",
                                         "ComSpclBidCode5", "ComSpclBidCode6",
                                         "FeatureId", "FeatureQuantity",
                                         "ComBrand", "ComGroup", "ComFamily",
                                         "DomBuyerGrpID", "DomBuyerGrpName",
                                         "RequestingApplicationID",
                                         "PricePointId",
                                         "PricePointName", "Price",
                                         'GEO_CODE'],
                                data=[[5300242, 'CHW_NA', 2, 'NA', 'I', 'J',
                                       2017, 9, 0, 0, 93008, 123456, 'E', 0,
                                       'Public', 'Government', 9, 0, '8247a',
                                       '3584L55', 'TS4500 HD2 Base Frame',
                                       'TS4500 HD2 Base Frame', 0.058058, 0,
                                       0.374007, 4.826653, 'XYZ', 'H',
                                       'P', 4, 1,
                                       16578.5, 1.207031, 4254.044, 1.963244,
                                       'S', 541, 961, 950, 671, 543,
                                       657, 3, 267,
                                       'OTHER POWER', 'STORWIZE V7000',
                                       'V3700 SFF CONTROL', 'DB5023M5',
                                       'TCU FINANCIAL GROUP', 'EPRICER', 45,
                                       'Original Quoted Price',
                                       76297.11987, 'NA']])

        test_result = dff.DataFieldsFilter.get_NA(inpu)

        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        assert_frame_equal(expected.reindex_axis(sorted(expected.columns),
                           axis=1),
                           test_result.reindex_axis(sorted(expected.columns),
                           axis=1))
