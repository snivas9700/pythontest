# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 15:50:58 2018

@author: Ravindranath
"""

DATA_PATH = './Models_in_Use/'
DEPLOYEDMODELRULE_NA = './Models_in_Use/DeployedModelRules_NA_2018-04-27.csv'
DEPLOYEDMODELRULE_EMEA = './Models_in_Use/DeployedModelRules_EMEA 2018-06-06.csv'
FIXEDPRODTABLE_NA = './Models_in_Use/FixedDiscounting_Rules_NA.csv'
FIXEDPRODTABLE_EMEA = './Models_in_Use/FixedDiscounting_Rules_EMEA.csv'
FIXEDPRODTABLE_JP = './Models_in_Use/FixedDiscounting_Rules_JP.csv'
VALIDATORPRODTABLE_NA = './Models_in_Use/HW_Validator_Rules_NA.csv'
VALIDATORPRODTABLE_EMEA = './Models_in_Use/HW_Validator_Rules_EMEA.csv'
VALIDATORPRODTABLE_JP = './Models_in_Use/HW_Validator_Rules_JP.csv'
BUNDLETABLE = 'BundleTableList.csv'
GEOMAP = './Models_in_Use/Country_SubReg_Geo_Mapping.csv'

# Defined list variables which are used in xmltodf,Reporter classes
TAG_LIST = [
    "Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
    "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComRevDivCd", 'ComBrand',
    "ComGroup", "ComFamily", "ComMT", "ComMTM", "ComQuotePricePofL",
    "ComDelgPriceL4PofL", "ComCostPofL", "ComLogListPrice", "ComMTMDesc",
    "ComMTMDescLocal", "ComCategory", "ComSubCategory", "ComSpclBidCode1",
    "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4", "ComSpclBidCode5",
    "ComSpclBidCode6", "HWPlatformid", "FeatureId", "FeatureQuantity",
    "Level_1", "Level_2", "Level_3", "Level_4"
]
TAG_QUOTE_LIST = [
    "Countrycode", "Channelidcode", "Year", "Month", "EndOfQtr", "Indirect",
    "DomBuyerGrpID", "DomBuyerGrpName" ,"QuoteType"
]
TAG_HEADER_LIST = ['RequestingApplicationID', 'QuoteID', 'ModelID', 'Version']
PRICEPOINT_TAG_LIST = ['PricePointId', 'PricePointName', 'Price']

CUSTOMERINFORMATION_TAG_LIST = [
    'CustomerNumber', 'ClientSegCd', 'ClientSeg', 'CCMScustomerNumber',
    'CustomerSecName', 'CustomerIndustryName'
]

OUTPUT_COLUMNS = [
    "CustomerNumber", "ClientSegCd", "ClientSeg=E", "CCMScustomerNumber",
    "CustomerSecName", "CustomerIndustryName", "RequestingApplicationID",
    "QuoteID", "ModelID", "Version", "Countrycode", "ChannelID", "Year",
    "Month", "EndOfQtr", "Indirect(1/0)", "DomBuyerGrpID", "DomBuyerGrpName",
    "Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
    "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComRevDivCd",
    "ComRevDivCd_Orig", "ComBrand", "ComGroup", "ComFamily", "ComMT", "ComMTM",
    "ComQuotePricePofL", "ComDelgPriceL4PofL", "ComCostPofL",
    "ComLogListPrice", "ComMTMDesc", "ComMTMDescLocal", "ComCategory",
    "ComSubCategory", "ComSpclBidCode1", "ComSpclBidCode2", "ComSpclBidCode3",
    "ComSpclBidCode4", "ComSpclBidCode5", "ComSpclBidCode6", 
    "Level_1", "Level_2", "Level_3", "Level_4", "HWPlatformid", "FeatureId",
    "FeatureQuantity", "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL",
    "ComHighPofL", "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib","QuoteType"
]
INT64LIST = [
    'Year', 'Month', 'EndOfQtr', 'Indirect(1/0)', 'ClientSeg=E', 'UpgMES',
    'Quantity', 'WinLoss', 'Componentid'
]
FLOAT64LIST = [
    'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4',
    'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLogListPrice'
]

COLUMNS = [
    "CustomerNumber", "ClientSegCd", "ClientSeg", "CCMScustomerNumber",
    "CustomerSecName", "CustomerIndustryName", "RequestingApplicationID",
    "quoteidnew", "ModelID", "Version", "Countrycode", "Channelidcode", "Year",
    "Month", "EndOfQtr", "Indirect", "DomBuyerGrpID", "DomBuyerGrpName",
    "Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
    "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComRevDivCd",
    "ComRevDivCd_Orig", "ComBrand", "ComGroup", "ComFamily", "ComMT", "ComMTM",
    "ComQuotePricePofL", "ComDelgPriceL4PofL", "ComCostPofL",
    "ComLogListPrice", "ComMTMDesc", "ComMTMDescLocal", "ComCategory",
    "ComSubCategory", "ComSpclBidCode1", "ComSpclBidCode2", "ComSpclBidCode3",
    "ComSpclBidCode4", "ComSpclBidCode5", "ComSpclBidCode6", 
    "Level_1", "Level_2", "Level_3", "Level_4", "HWPlatformid", "FeatureId",
    "FeatureQuantity", "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL",
    "ComHighPofL", "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib","QuoteType"
]

NORMALHANDLER_EMEA = [
    'GEO_CODE', 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL', 'AdjComMedPofL',
    'AdjComHighPofL', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice',
    'OptimalPricePofL', 'OptimalPrice', 'OptimalPriceWinProb',
    'OptimalPriceGP', 'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow',
    'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice',
    'QuotePricePofL', 'QuotePrice', 'QuotePriceWinProb', 'QuotePriceGP',
    'QuotePriceExpectedGP', 'PredictedQuotePricePofL', 'PredictedQuotePrice',
    'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL',
    'COPComMedPofL', 'COPComHighPofL', 'COPOptimalPrice',
    'COPOptimalPricePofL', 'COPOptimalPriceWinProb', 'COPOptimalPriceGP',
    'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow',
    'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP',
    'COPQuotePriceExpectedGP'
]

NORMALHANDLER_NA = [
    "CustomerNumber", "ClientSegCd", "ClientSeg=E", "CCMScustomerNumber",
    "CustomerSecName", "CustomerIndustryName", "RequestingApplicationID",
    "QuoteID", "ModelID", "Version", "Countrycode", "ChannelID", "Year",
    "Month", "EndOfQtr", "Indirect(1/0)", "DomBuyerGrpID", "DomBuyerGrpName",
    "Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
    "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComRevDivCd", "ComBrand",
    "ComGroup", "ComFamily", "ComMT", "ComMTM", "ComQuotePricePofL",
    "ComDelgPriceL4PofL", "ComCostPofL", "ComLogListPrice", "ComMTMDesc",
    "ComMTMDescLocal", "ComCategory", "ComSubCategory", "ComSpclBidCode1",
    "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4", "ComSpclBidCode5",
    "ComSpclBidCode6", "Level_1", "Level_2", "Level_3", "Level_4",
    "HWPlatformid", "FeatureId", "FeatureQuantity",
    "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL", "ComHighPofL",
    "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib"
]

NORMALHANDLER_NA_BP = [
    "CustomerNumber", "ClientSegCd", "ClientSeg=E", "CCMScustomerNumber",
    "CustomerSecName", "CustomerIndustryName", "RequestingApplicationID",
    "QuoteID", "ModelID", "Version", "Countrycode", "ChannelID", "Year",
    "Month", "EndOfQtr", "Indirect(1/0)", "DomBuyerGrpID", "DomBuyerGrpName",
    "Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
    "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComRevDivCd", "ComBrand",
    "ComGroup", "ComFamily", "ComMT", "ComMTM", "ComQuotePricePofL",
    "ComDelgPriceL4PofL", "ComCostPofL", "ComLogListPrice", "ComMTMDesc",
    "ComMTMDescLocal", "ComCategory", "ComSubCategory", "ComSpclBidCode1",
    "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4", "ComSpclBidCode5",
    "ComSpclBidCode6","Level_1", "Level_2","Level_3", "Level_4",
    "HWPlatformid", "FeatureId", "FeatureQuantity",
    "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL", "ComHighPofL",
    "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib","QuoteType"
]

NON_ZERO_PRICE = [
    "CustomerNumber","ClientSegCd","ClientSeg=E","CCMScustomerNumber",
    "CustomerSecName","CustomerIndustryName","RequestingApplicationID",
    "QuoteID","ModelID","Version","Countrycode","ChannelID","Year","Month",
    "EndOfQtr","Indirect(1/0)","DomBuyerGrpID","DomBuyerGrpName","Componentid",
    "Quantity","UpgMES","ComListPrice","ComTMC","ComQuotePrice","ComDelgPriceL4",
    "ComRevCat","ComRevDivCd","ComRevDivCd_Orig","ComBrand","ComGroup","ComFamily",
    "ComMT","ComMTM","ComQuotePricePofL","ComDelgPriceL4PofL","ComCostPofL",
    "ComLogListPrice","ComMTMDesc","ComMTMDescLocal","ComCategory","ComSubCategory",
    "ComSpclBidCode1","ComSpclBidCode2","ComSpclBidCode3","ComSpclBidCode4",
    "ComSpclBidCode5","ComSpclBidCode6","Level_1", "Level_2", "Level_3", "Level_4",
    "HWPlatformid","FeatureId","FeatureQuantity","quoteidnew","WinLoss","ComLowPofL",
    "ComMedPofL","ComHighPofL","ComMedPrice","DealSize","LogDealSize","ComPctContrib","QuoteType"
    ]

ZERO_PRICE = ["QuoteID", "ModelID", "Version", "Countrycode",
                 "ChannelID", "Year", "Month", "EndOfQtr", "Indirect(1/0)",
                 "CustomerNumber", "CCMScustomerNumber", "ClientSegCd",
                 "ClientSeg=E", "CustomerSecName", "CustomerIndustryName",
                 "Componentid", "HWPlatformid", "ComMT", "ComMTM", "ComMTMDesc",
                 "ComMTMDescLocal", "ComQuotePricePofL", "ComDelgPriceL4PofL",
                 "ComCostPofL", "ComLowPofL", "ComMedPofL", "ComHighPofL", "ComLogListPrice", "ComRevDivCd","ComRevDivCd_Orig", "ComCategory",
                 "ComSubCategory", "UpgMES", "Quantity", "ComListPrice", "ComTMC",
                 "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComSpclBidCode1",
                 "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4",
                 "ComSpclBidCode5", "ComSpclBidCode6","Level_1", "Level_2", "Level_3", "Level_4","FeatureId",
                 "FeatureQuantity", "ComBrand", "ComGroup","WinLoss", "ComFamily",
                 "DomBuyerGrpID", "DomBuyerGrpName", "RequestingApplicationID",
                 'TreeNode', 'ComTMCPofL', 'AdjComLowPofL',
                'AdjComMedPofL', 'AdjComHighPofL', 'AdjComLowPrice',
                'AdjComMedPrice', 'AdjComHighPrice', 'OptimalPricePofL',
                'OptimalPrice', 'OptimalPriceWinProb', 'OptimalPriceGP',
                'OptimalPriceExpectedGP', 'OptimalPriceIntervalLow',
                'OptimalPriceIntervalHigh', 'DealBotLineSpreadOptimalPrice',
                'QuotePricePofL', 'QuotePrice', 'QuotePriceWinProb',
                'QuotePriceGP', 'QuotePriceExpectedGP', 'PredictedQuotePricePofL',
                'PredictedQuotePrice', 'COPComLowPrice', 'COPComMedPrice',
                'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL',
                'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL',
                'COPOptimalPriceWinProb', 'COPOptimalPriceGP',
                'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow',
                'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb',
                'COPQuotePriceGP', 'COPQuotePriceExpectedGP','QuoteType'
            ]

REPORTER_COLUMNS =[
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
            'COPQuotePriceExpectedGP']

NORMALHANDLER_JP = [
    "CustomerNumber", "ClientSegCd", "ClientSeg=E", "CCMScustomerNumber",
    "CustomerSecName", "CustomerIndustryName", "RequestingApplicationID",
    "QuoteID", "ModelID", "Version", "Countrycode", "ChannelID", "Year",
    "Month", "EndOfQtr", "Indirect(1/0)", "DomBuyerGrpID", "DomBuyerGrpName",
    "Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
    "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComRevDivCd", "ComBrand",
    "ComGroup", "ComFamily", "ComMT", "ComMTM", "ComQuotePricePofL",
    "ComDelgPriceL4PofL", "ComCostPofL", "ComLogListPrice", "ComMTMDesc",
    "ComMTMDescLocal", "ComCategory", "ComSubCategory", "ComSpclBidCode1",
    "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4", "ComSpclBidCode5",
    "ComSpclBidCode6", "Level_1", "Level_2", "Level_3", "Level_4",
    "HWPlatformid", "FeatureId", "FeatureQuantity",
    "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL", "ComHighPofL",
    "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib"
]

NORMALHANDLER_JP_BP = [
    "CustomerNumber", "ClientSegCd", "ClientSeg=E", "CCMScustomerNumber",
    "CustomerSecName", "CustomerIndustryName", "RequestingApplicationID",
    "QuoteID", "ModelID", "Version", "Countrycode", "ChannelID", "Year",
    "Month", "EndOfQtr", "Indirect(1/0)", "DomBuyerGrpID", "DomBuyerGrpName",
    "Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
    "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComRevDivCd", "ComBrand",
    "ComGroup", "ComFamily", "ComMT", "ComMTM", "ComQuotePricePofL",
    "ComDelgPriceL4PofL", "ComCostPofL", "ComLogListPrice", "ComMTMDesc",
    "ComMTMDescLocal", "ComCategory", "ComSubCategory", "ComSpclBidCode1",
    "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4", "ComSpclBidCode5",
    "ComSpclBidCode6","Level_1", "Level_2","Level_3", "Level_4",
    "HWPlatformid", "FeatureId", "FeatureQuantity",
    "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL", "ComHighPofL",
    "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib","QuoteType"
]
