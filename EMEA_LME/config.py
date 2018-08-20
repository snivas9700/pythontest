# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 15:50:58 2018

@author: Ravindranath
"""

DATA_PATH = './Models_in_Use/'
DEPLOYEDMODELRULE_NA = './Models_in_Use/DeployedModelRules_NA_2018-02-07.csv'
DEPLOYEDMODELRULE_EMEA = './Models_in_Use/DeployedModelRules_EMEA 2018-06-06.csv'
FIXEDPRODTABLE_NA = './Models_in_Use/FixedDiscounting_Rules_NA.csv'
FIXEDPRODTABLE_EMEA = './Models_in_Use/FixedDiscounting_Rules_EMEA.csv'
BUNDLETABLE = 'BundleTableList.csv'
GEOMAP = './Models_in_Use/Country_SubReg_Geo_Mapping.csv'
#data_path_TSS = './pricing_tss/'
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

#added "TSSComponentincluded" +++++++++++++++++++++++++++++++++++++++++++++
TAG_QUOTE_LIST = [
    "Countrycode", "Channelidcode", "Year", "Month", "EndOfQtr", "Indirect",
    "DomBuyerGrpID", "DomBuyerGrpName" ,"QuoteType"
]

#added "CHANGEPERIODSTOPDATE" as per Fabienne
TAG_QUOTE_TSS_LIST = ["TSSComponentincluded",
                      "TSSContstartdate","TSSContenddate","TSSContduration", 
                      "TSSPricerefdate", 
                      "TSSPricoptdescription", "TSSFrameOffering","ChangePeriodStopDate"     
                     ]

TAG_QUOTE_LIST_OLD = [
    "Countrycode", "Channelidcode", "Year", "Month", "EndOfQtr", "Indirect",
    "DomBuyerGrpID", "DomBuyerGrpName" ,"QuoteType"
]

TAG_HEADER_LIST = ['RequestingApplicationID', 'QuoteID', 'ModelID', 'Version']
PRICEPOINT_TAG_LIST = ['PricePointId', 'PricePointName', 'Price']

CUSTOMERINFORMATION_TAG_LIST = [
    'CustomerNumber', 'ClientSegCd', 'ClientSeg', 'CCMScustomerNumber',
    'CustomerSecName', 'CustomerIndustryName'
]

#added TSS_COMPONENTS_TAG_LIST +++++++++++++++++++++++++++++++++++++++++++++
#added "CHARGESTARTDATE","CHARGEENDDATE" , 'ctry_desc'as per Fabienne,Gloria
#additional fields created for TSS discount ("TssComponentType", "hwma_pti", "swma_discounted_price")
#removed "flex","ecs","p_uplift_comm","level_1", "level_2",	 "level_3","level_4", 
   
TSS_COMPONENTS_TAG_LIST = [
    "Componentid","TSScomid","type","model","quantity","warranty",
    "warrantyperiod", "servoffdesc","servoffcode","servstartdate","servenddate",
    "serviceduration","inststartdate","instenddate","serial","servlvldesc",
    "servlvlcode","basecharge","committedcharge","totalcharge","PTI0",
    "CMDAprice","Cost", "imt_code","coverage_hours_days",	
    "coverage_hours","coverage_days","sl_cntct", "sl_fix_time",
    "sl_onsite", "sl_part_time", "TssComponentType", "hwma_pti", "swma_discounted_price",
    "ChargeStartDate","ChargeEndDate",'ctry_desc'
] 

"""# removed "hw_level1", "hw_level2",	
"hw_level3","hw_level4" from TSS_COMPONENT_TAG LIST as per Gloria's note """
#removed "flex","ecs","p_uplift_comm", 
TSS_RETURN_FIELDS = ['TSS_AdjComLowPofL','TSS_AdjComMedPofL',
                     'TSS_AdjComHighPofL', 'TSS_AdjComLowPrice',
                     'TSS_AdjComMedPrice', 'TSS_AdjComHighPrice', 
                     'TSS_OptimalPricePofL','TSS_OptimalPrice',
                     'TSS_OptimalPriceWinProb','TSS_OptimalPriceGP',
                     'TSS_OptimalPriceExpectedGP','TSS_OptimalPriceIntervalLow',
                     'TSS_OptimalPriceIntervalHigh',
                     'TSS_DealBotLineSpreadOptimalPrice', "imt_code",
                     "coverage_hours_days", "coverage_hours",
                     "coverage_days","sl_cntct", "sl_fix_time","sl_onsite", 
                     "sl_part_time",	"ParentMapping_ComponentID",
                     "TssComponentType", "hwma_pti", "swma_discounted_price",
                     "ChargeStartDate","ChargeEndDate"
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

#added++++++++++++++++++++++++
RENAME_COL = {
        'Channelidcode' : 'ChannelID' ,
        'ClientSeg' : 'ClientSeg=E',
        'Indirect' : 'Indirect(1/0)',
        'quantity' : 'TSS_quantity',
        'quoteidnew' : 'QuoteID',
        'level_1' : 'Level_1',
        'level_2' : 'Level_2',
        'level_3' : 'Level_3',
        'level_4' : 'Level_4'              
        }

#added++++++++++++++++++++++++,"ComRevDivCd_Orig"
floatlist = ["TSSContduration","ComSpclBidCode1","ComSpclBidCode2","ComSpclBidCode3",
             "ComSpclBidCode4","ComSpclBidCode5","ComSpclBidCode6","FeatureQuantity",
             "warrantyperiod","serviceduration","ComLowPofL","ComMedPofL","ComHighPofL",
             "ComMedPrice","DealSize","LogDealSize","ComPctContrib"]
#added++++++++++++++++++++++++
INT64LIST_TSS = [
    'Year', 'Month', 'EndOfQtr', 'Indirect(1/0)', 'ClientSeg=E', 'UpgMES',
    'Quantity','TSS_quantity', 'WinLoss', 'Componentid']


#'TSS_quantity','TSScomid','warrantyperiod','serviceduration'
#added++++++++++++++++++++++++
FLOAT64LIST_TSS = [
    'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4',
    'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLogListPrice',
    'basecharge','committedcharge','totalcharge','PTI0','CMDAprice','Cost',
    "coverage_hours_days",	"coverage_hours",	"coverage_days",	
    "sl_cntct",	"sl_fix_time",	"sl_onsite",	"sl_part_time"
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
    "ComHighPofL", "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib","QuoteType",
    "TSSComponentincluded"
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

NORMALHANDLER_NA_LME = [
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
    "ComSpclBidCode6", "HWPlatformid", "FeatureId", "FeatureQuantity",
    "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL", "ComHighPofL",
    "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib","Level_1",
    "Level_2","Level_3","Level_4","GEO_CODE"
]

"""
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
    "ComSpclBidCode6", "HWPlatformid", "FeatureId", "FeatureQuantity",
    "quoteidnew", "WinLoss", "ComLowPofL", "ComMedPofL", "ComHighPofL",
    "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib"
]
"""

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
    "ComMedPrice", "DealSize", "LogDealSize", "ComPctContrib","QuoteType","GEO_CODE"
]

NON_ZERO_PRICE_old = [
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
# added++++++++++++++++++++++++++
NON_ZERO_PRICE = [
    'CustomerNumber', 'ClientSegCd', 'ClientSeg=E', 'CCMScustomerNumber',
    'CustomerSecName', 'CustomerIndustryName', 'RequestingApplicationID',
    'ModelID', 'Version', 'Countrycode', 'ChannelID', 'Year', 'Month',
    'EndOfQtr', 'Indirect(1/0)', 'DomBuyerGrpID', 'DomBuyerGrpName',
    'QuoteType', 'TSSComponentincluded', 'Quantity', 'UpgMES',
    'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4',
    'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily',
    'ComMT', 'ComMTM', 'ComQuotePricePofL', 'ComDelgPriceL4PofL',
    'ComCostPofL', 'ComLogListPrice', 'ComMTMDesc', 'ComMTMDescLocal',
    'ComCategory', 'ComSubCategory', 'ComSpclBidCode1', 'ComSpclBidCode2',
    'ComSpclBidCode3', 'ComSpclBidCode4', 'ComSpclBidCode5',
    'ComSpclBidCode6', 'HWPlatformid', 'FeatureId', 'FeatureQuantity',
    'Level_1', 'Level_2', 'Level_3', 'Level_4', 'QuoteID', 'WinLoss',
    'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize',
    'LogDealSize', 'ComPctContrib', 'ComRevDivCd_Orig', 'Componentid'
    ]
# added++++++++++++++++++++++++++
NON_ZERO_PRICE_TSS = [
    'CustomerNumber', 'ClientSegCd', 'ClientSeg=E', 'CCMScustomerNumber',
    'CustomerSecName', 'CustomerIndustryName', 'RequestingApplicationID',
    'ModelID', 'Version', 'Countrycode', 'ChannelID', 'Year', 'Month',
    'EndOfQtr', 'Indirect(1/0)', 'DomBuyerGrpID', 'DomBuyerGrpName',
    'QuoteType', 'TSSComponentincluded', 'Quantity', 'UpgMES',
    'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4',
    'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily',
    'ComMT', 'ComMTM', 'ComQuotePricePofL', 'ComDelgPriceL4PofL',
    'ComCostPofL', 'ComLogListPrice', 'ComMTMDesc', 'ComMTMDescLocal',
    'ComCategory', 'ComSubCategory', 'ComSpclBidCode1', 'ComSpclBidCode2',
    'ComSpclBidCode3', 'ComSpclBidCode4', 'ComSpclBidCode5',
    'ComSpclBidCode6', 'HWPlatformid', 'FeatureId', 'FeatureQuantity',
    'TSScomid', 'type', 'model','Level_1', 'Level_2', 'Level_3', 'Level_4',
    'TSS_quantity', 'warranty', 'warrantyperiod', 'servoffdesc',
    'servoffcode', 'servstartdate', 'servenddate', 'serviceduration',
    'inststartdate', 'instenddate', 'serial', 'servlvldesc', 'servlvlcode',
    'basecharge', 'committedcharge', 'totalcharge', 'PTI0', 'CMDAprice',
    'Cost', 'QuoteID', 'WinLoss', 'ComLowPofL', 'ComMedPofL', 'ComHighPofL',
    'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib',
    'ComRevDivCd_Orig', 'Componentid', "ParentMapping_ComponentID" 
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
            'COPQuotePriceExpectedGP', "GEO_CODE"]
"""
#removed "flex","ecs",,	"p_uplift_comm"
REPORTER_COLUMNS_TSS = [
        "CustomerNumber","ClientSegCd","ClientSeg_E","CCMScustomerNumber",
        "CustomerSecName","CustomerIndustryName","RequestingApplicationID",
        "ModelID","Version","Countrycode","ChannelID","Year","Month","EndOfQtr",
        "Indirect_1_0","DomBuyerGrpID","DomBuyerGrpName","QuoteType","TSSComponentincluded",
        "TSSContstartdate","TSSContenddate","TSSContduration", "TSSPricerefdate", 
        "TSSPricoptdescription", "TSSFrameOffering", "ParentMapping_ComponentID",
        "Quantity","UpgMES","ComListPrice",
        "ComTMC","ComQuotePrice","ComDelgPriceL4","ComRevCat","ComRevDivCd","ComBrand","ComGroup",
        "ComFamily","ComMT","ComMTM","ComQuotePricePofL","ComDelgPriceL4PofL","ComCostPofL",
        "ComLogListPrice","ComMTMDesc","ComMTMDescLocal","ComCategory","ComSubCategory",
        "ComSpclBidCode1","ComSpclBidCode2","ComSpclBidCode3","ComSpclBidCode4","ComSpclBidCode5",
        "ComSpclBidCode6","HWPlatformid","FeatureId","FeatureQuantity","Level_1","Level_2",
        "Level_3","Level_4","TSScomid","type","model","TSS_quantity","warranty",
        "warrantyperiod","servoffdesc","servoffcode","servstartdate","servenddate",
        "serviceduration","inststartdate","instenddate","serial","servlvldesc","servlvlcode",
        "basecharge","committedcharge","totalcharge","PTI0","CMDAprice","Cost","QuoteID",
        "WinLoss","ComLowPofL","ComMedPofL","ComHighPofL","ComMedPrice","DealSize","LogDealSize",
        "ComPctContrib","ComRevDivCd_Orig","Componentid","GEO_CODE","TreeNode","ComTMCPofL",
        "AdjComLowPofL","AdjComMedPofL","AdjComHighPofL","AdjComLowPrice","AdjComMedPrice",
        "AdjComHighPrice","OptimalPricePofL","OptimalPrice","OptimalPriceWinProb",
        "OptimalPriceGP","OptimalPriceExpectedGP","OptimalPriceIntervalLow",
        "OptimalPriceIntervalHigh","DealBotLineSpreadOptimalPrice","QuotePricePofL",
        "QuotePrice","QuotePriceWinProb","QuotePriceGP","QuotePriceExpectedGP",
        "PredictedQuotePricePofL","PredictedQuotePrice","COPComLowPrice","COPComMedPrice",
        "COPComHighPrice","COPComLowPofL","COPComMedPofL","COPComHighPofL","COPOptimalPrice",
        "COPOptimalPricePofL","COPOptimalPriceWinProb","COPOptimalPriceGP","COPOptimalPriceExpectedGP",
        "COPOptimalPriceIntervalLow","COPOptimalPriceIntervalHigh","COPQuotePriceWinProb",
        "COPQuotePriceGP","COPQuotePriceExpectedGP", 'TSS_AdjComLowPofL','TSS_AdjComMedPofL',
                     'TSS_AdjComHighPofL', 'TSS_AdjComLowPrice',
                     'TSS_AdjComMedPrice', 'TSS_AdjComHighPrice', 
                     'TSS_OptimalPricePofL','TSS_OptimalPrice',
                     'TSS_OptimalPriceWinProb','TSS_OptimalPriceGP',
                     'TSS_OptimalPriceExpectedGP','TSS_OptimalPriceIntervalLow',
                     'TSS_OptimalPriceIntervalHigh',
                     'TSS_DealBotLineSpreadOptimalPrice',
                     "imt_code",
                     "coverage_hours_days", "coverage_hours",
                     "coverage_days","sl_cntct", "sl_fix_time","sl_onsite", 
                     "sl_part_time", "ParentMapping_ComponentID"
] """
#removed  'ecs', 'flex','p_uplift_comm', 
REPORTER_COLUMNS_TSS = ['CustomerNumber', 'AdjComHighPofL', 'AdjComHighPrice', 
                        'AdjComLowPofL',
       'AdjComLowPrice', 'AdjComMedPofL', 'AdjComMedPrice',
       'CCMScustomerNumber', 'CMDAprice', 'COPComHighPofL',
       'COPComHighPrice', 'COPComLowPofL', 'COPComLowPrice',
       'COPComMedPofL', 'COPComMedPrice', 'COPOptimalPrice',
       'COPOptimalPriceExpectedGP', 'COPOptimalPriceGP',
       'COPOptimalPriceIntervalHigh', 'COPOptimalPriceIntervalLow',
       'COPOptimalPricePofL', 'COPOptimalPriceWinProb',
       'COPQuotePriceExpectedGP', 'COPQuotePriceGP',
       'COPQuotePriceWinProb', 'ChannelID', 'ClientSegCd', 'ClientSeg_E',
       'ComBrand', 'ComCostPofL', 'ComDelgPriceL4',
       'ComDelgPriceL4PofL', 'ComFamily', 'ComGroup', 'ComHighPofL',
       'ComListPrice', 'ComLogListPrice', 'ComLowPofL', 'ComMT', 'ComMTM',
       'ComMedPofL', 'ComMedPrice',
       'ComPctContrib', 'ComQuotePrice', 'ComQuotePricePofL', 'ComRevCat',
       'ComRevDivCd', 'ComRevDivCd_Orig',  'ComTMC',
       'ComTMCPofL', 'Componentid', 'Cost', 'Countrycode',
       'CustomerIndustryName' , 'CustomerSecName',
       'DealBotLineSpreadOptimalPrice', 'DealSize', 'DomBuyerGrpID',
       'DomBuyerGrpName', 'EndOfQtr',
       'GEO_CODE', 'Indirect_1_0', 'LogDealSize', 'ModelID', 'Month',
       'OptimalPrice', 'OptimalPriceExpectedGP', 'OptimalPriceGP',
       'OptimalPriceIntervalHigh', 'OptimalPriceIntervalLow',
       'OptimalPricePofL', 'OptimalPriceWinProb', 'PTI0',
       'PredictedQuotePrice',
       'PredictedQuotePricePofL', 'Quantity', 'QuoteID', 'QuotePrice',
       'QuotePriceExpectedGP', 'QuotePriceGP', 'QuotePricePofL',
       'QuotePriceWinProb', 'QuoteType', 'RequestingApplicationID',
       'TSSComponentincluded', 'TSSContduration', 'TSSContenddate',
       'TSSContstartdate', 'TSSFrameOffering', 'TSSPricerefdate',
       'TSSPricoptdescription', 'TSS_AdjComHighPofL',
       'TSS_AdjComHighPrice', 'TSS_AdjComLowPofL', 'TSS_AdjComLowPrice',
       'TSS_AdjComMedPofL', 'TSS_AdjComMedPrice',
       'TSS_DealBotLineSpreadOptimalPrice', 'TSS_OptimalPrice',
       'TSS_OptimalPriceExpectedGP', 'TSS_OptimalPriceGP',
       'TSS_OptimalPriceIntervalHigh', 'TSS_OptimalPriceIntervalLow',
       'TSS_OptimalPricePofL', 'TSS_OptimalPriceWinProb', 'TSS_quantity',
       'TSScomid', 'TreeNode', 'UpgMES', 'Version', 'WinLoss', 'Year',
       'basecharge', 'committedcharge', 'coverage_days', 'coverage_hours',
       'coverage_hours_days','imt_code', 'instenddate',
       'inststartdate', 'model', 'serial', 'servenddate',
       'serviceduration', 'servlvlcode', 'servlvldesc', 'servoffcode',
       'servoffdesc', 'servstartdate', 'sl_cntct', 'sl_fix_time',
       'sl_onsite', 'sl_part_time', 'totalcharge', 'type', 'warranty',
       'warrantyperiod', "hwma_pti", "swma_discounted_price","ChangePeriodStopDate",
       "ParentMapping_ComponentID","ChargeStartDate","ChargeEndDate",'ctry_desc'
       ]
#delete TSS fields from hardware component list
DELETE_LIST = ["TSSComponentincluded", 'TSS_AdjComHighPofL',
                          "TSSContstartdate","TSSContenddate","TSSContduration", "TSSPricerefdate", 
                          "TSSPricoptdescription", "TSSFrameOffering","TSScomid",
                          "type","model", "TSS_quantity", "warranty", "warrantyperiod",
                          "servoffdesc", "servoffcode", "servstartdate", "servenddate", 
                          "serviceduration","inststartdate","instenddate","serial",
                          "servlvldesc", "servlvlcode", "basecharge","committedcharge",
                          "totalcharge", "PTI0", "CMDAprice","Cost",
                          'TSS_AdjComHighPrice', 'TSS_AdjComLowPofL', 'TSS_AdjComLowPrice',
       'TSS_AdjComMedPofL', 'TSS_AdjComMedPrice',
       'TSS_DealBotLineSpreadOptimalPrice', 'TSS_OptimalPrice',
       'TSS_OptimalPriceExpectedGP', 'TSS_OptimalPriceGP',
       'TSS_OptimalPriceIntervalHigh', 'TSS_OptimalPriceIntervalLow',
       'TSS_OptimalPricePofL', 'TSS_OptimalPriceWinProb', 'imt_code', 'instenddate',
       'inststartdate', 'model', 'serial', 'servenddate',
       'serviceduration', 'servlvlcode', 'servlvldesc', 'servoffcode',
       'servoffdesc', 'servstartdate', 'sl_cntct', 'sl_fix_time',
       'sl_onsite', 'sl_part_time', 'totalcharge', 'type', 'warranty',
       'warrantyperiod', 
       'ChangePeriodStopDate','ChargeStartDate', 'ChargeEndDate','hwma_pti','swma_discounted_price',
       'coverage_days','coverage_hours','coverage_hours_days','ctry_desc'
        ]

final_online_TSS= [
"QuoteID","Componentid","ComLogListPrice","ComTMC","ComCostPofL","ComPctContrib",
"DealSize","LogDealSize","TSSComponentincluded","Indirect(1/0)","EndOfQtr","ComDelgPriceL4",
"WinLoss","CustomerSecName","CustomerIndustryName","ComQuotePrice","totalcharge","PTI0","Cost",
"committedcharge","TSSContduration","coverage_hours_days","sl_cntct","sl_onsite","sl_part_time",
"servoffdesc","imt_code","basecharge","dealsize","CustomerNumber","ClientSegCd","ClientSeg=E",
"CCMScustomerNumber","RequestingApplicationID","ModelID","Version","Countrycode","ChannelID",
"Year","Month","DomBuyerGrpID","DomBuyerGrpName","QuoteType","TSSContstartdate","TSSContenddate",
"TSSPricerefdate","TSSPricoptdescription","TSSFrameOffering","ChangePeriodStopDate","Quantity",
"UpgMES","ComListPrice","ComRevCat","ComRevDivCd","ComBrand","ComGroup","ComFamily","ComMT","ComMTM",
"ComQuotePricePofL","ComDelgPriceL4PofL","ComMTMDesc","ComMTMDescLocal","ComCategory","ComSubCategory",
"ComSpclBidCode1","ComSpclBidCode2","ComSpclBidCode3","ComSpclBidCode4","ComSpclBidCode5","ComSpclBidCode6",
"HWPlatformid","FeatureId","FeatureQuantity","Level_1","Level_2","Level_3","Level_4","TSScomid",
"type","model","TSS_quantity","warranty","warrantyperiod","servoffcode","servstartdate","servenddate",
"serviceduration","inststartdate","instenddate","serial","servlvldesc","servlvlcode","CMDAprice",
"coverage_hours","coverage_days","sl_fix_time","TssComponentType","hwma_pti","ChargeStartDate","ChargeEndDate",
"ctry_desc","ComLowPofL","ComMedPofL","ComHighPofL","ComMedPrice","ComRevDivCd_Orig","GEO_CODE","TreeNode",
"DealBotLineSpreadOptimalPrice","OptimalPrice","OptimalPriceExpectedGP","OptimalPriceGP","OptimalPriceIntervalHigh",
"OptimalPriceIntervalLow","OptimalPricePofL","OptimalPriceWinProb","PredictedQuotePrice","PredictedQuotePricePofL",
"QuotePrice","QuotePriceExpectedGP","QuotePriceGP","QuotePricePofL","QuotePriceWinProb","AdjComHighPrice",
"AdjComHighPofL","AdjComLowPrice","AdjComLowPofL","AdjComMedPrice","AdjComMedPofL","ComLowPrice","ComHighPrice",
"TSS_DealBotLineSpreadOptimalPrice","TSS_OptimalPrice","TSS_OptimalPriceExpectedGP","TSS_OptimalPriceGP",
"TSS_OptimalPriceIntervalHigh","TSS_OptimalPriceIntervalLow","TSS_OptimalPricePofL","TSS_OptimalPriceWinProb",
"pti","TSS_AdjComHighPrice","TSS_AdjComHighPofL","TSS_AdjComLowPrice","TSS_AdjComLowPofL","TSS_AdjComMedPrice",
"TSS_AdjComMedPofL", "ComTMCPofL"
]
