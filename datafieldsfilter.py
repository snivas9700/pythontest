import copy
from copy import deepcopy
from geodispatcher import GeoDispatcher

# Data Fields for EMEA
EMEA_FIELDS = ["Componentid", "Quantity", "UpgMES", "ComListPrice", "ComTMC",
               "ComQuotePrice", "ComDelgPriceL4", "WinLoss", "ComRevCat",
               "ComRevDivCd", 'ComBrand', "ComGroup", "ComFamily", "ComMT",
               "ComMTM", "ComQuotePricePofL",
               "ComDelgPriceL4PofL", "ComCostPofL", "ComLogListPrice",
               "Countrycode", "ChannelID", "Year", "Month",
               "EndOfQtr", "Indirect(1/0)", 'RequestingApplicationID', 'QuoteID',
               'ModelID', 'Version', 
               'CustomerNumber', 'ClientSegCd', 'ClientSeg=E']

# Data Fields for North America 
NA_FIELDS = ["QuoteID", "ModelID", "Version", "Countrycode", 'WinLoss',
             "ChannelID", "Year", "Month", "EndOfQtr", "Indirect(1/0)",
             "CustomerNumber", "CCMScustomerNumber", "ClientSegCd",
             "ClientSeg=E", "CustomerSecName", "CustomerIndustryName",
             "Componentid", "HWPlatformid", "ComMT", "ComMTM", "ComMTMDesc",
             "ComMTMDescLocal", "ComQuotePricePofL", "ComDelgPriceL4PofL",
             "ComCostPofL", "ComLogListPrice", "ComRevDivCd","ComRevDivCd_Orig",
             "ComCategory", "ComSubCategory", "UpgMES", "Quantity", 
             "ComListPrice", "ComTMC",
             "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComBrand",
             "ComGroup", "ComFamily","DomBuyerGrpID", "DomBuyerGrpName",
             "RequestingApplicationID","Level_1", "Level_2", "Level_3", "Level_4","QuoteType"]

# Data Fields for North America BP
# Extra added fields:
# Part Map: Level_0, Level_1, Level_2, Level_3, Level_4 [This is for NEW partmap table]
# UFCs : FeatureID and Feature Quantity
# SBCs : ComSpclBidCode1, ComSpclBidCode2, ComSpclBidCode3, ComSpclBidCode4, ComSpclBidCode5, ComSpclBidCode6
NA_FIELDS_BP = ["QuoteID", "ModelID", "Version", "Countrycode", 'WinLoss',
             "ChannelID", "Year", "Month", "EndOfQtr", "Indirect(1/0)",
             "CustomerNumber", "CCMScustomerNumber", "ClientSegCd",
             "ClientSeg=E", "CustomerSecName", "CustomerIndustryName",
             "Componentid", "HWPlatformid", "ComMT", "ComMTM", "ComMTMDesc",
             "ComMTMDescLocal", "ComQuotePricePofL", "ComDelgPriceL4PofL",
             "ComCostPofL", "ComLogListPrice", "ComRevDivCd", "ComCategory",
             "ComSubCategory", "UpgMES", "Quantity", "ComListPrice", "ComTMC",
             "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComSpclBidCode1",
             "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4",
             "ComSpclBidCode5", "ComSpclBidCode6", "FeatureId",
             "FeatureQuantity","ComBrand", "ComGroup", "ComFamily",
             "DomBuyerGrpID", "DomBuyerGrpName", "RequestingApplicationID",
             "Level_1", "Level_2", "Level_3", "Level_4","QuoteType"]
#Fileds list for Japan
JP_FIELDS = ["QuoteID", "ModelID", "Version", "Countrycode", 'WinLoss',
             "ChannelID", "Year", "Month", "EndOfQtr", "Indirect(1/0)",
             "CustomerNumber", "CCMScustomerNumber", "ClientSegCd",
             "ClientSeg=E", "CustomerSecName", "CustomerIndustryName",
             "Componentid", "HWPlatformid", "ComMT", "ComMTM", "ComMTMDesc",
             "ComMTMDescLocal", "ComQuotePricePofL", "ComDelgPriceL4PofL",
             "ComCostPofL", "ComLogListPrice", "ComRevDivCd", "ComCategory",
             "ComSubCategory", "UpgMES", "Quantity", "ComListPrice", "ComTMC",
             "ComQuotePrice", "ComDelgPriceL4", "ComRevCat","FeatureId",
             "FeatureQuantity","ComBrand", "ComGroup", "ComFamily",
             "DomBuyerGrpID", "DomBuyerGrpName", "RequestingApplicationID",
             "Level_1", "Level_2", "Level_3", "Level_4","QuoteType"]

#Fileds list for Japan BP
JP_FIELDS_BP = ["QuoteID", "ModelID", "Version", "Countrycode", 'WinLoss',
             "ChannelID", "Year", "Month", "EndOfQtr", "Indirect(1/0)",
             "CustomerNumber", "CCMScustomerNumber", "ClientSegCd",
             "ClientSeg=E", "CustomerSecName", "CustomerIndustryName",
             "Componentid", "HWPlatformid", "ComMT", "ComMTM", "ComMTMDesc",
             "ComMTMDescLocal", "ComQuotePricePofL", "ComDelgPriceL4PofL",
             "ComCostPofL", "ComLogListPrice", "ComRevDivCd", "ComCategory",
             "ComSubCategory", "UpgMES", "Quantity", "ComListPrice", "ComTMC",
             "ComQuotePrice", "ComDelgPriceL4", "ComRevCat", "ComSpclBidCode1",
             "ComSpclBidCode2", "ComSpclBidCode3", "ComSpclBidCode4",
             "ComSpclBidCode5", "ComSpclBidCode6", "FeatureId",
             "FeatureQuantity","ComBrand", "ComGroup", "ComFamily",
             "DomBuyerGrpID", "DomBuyerGrpName", "RequestingApplicationID",
             "Level_1", "Level_2", "Level_3", "Level_4","QuoteType"]



class DataFieldsFilter():
    def __init__(self):
        pass
    """Keep only model needed data fields.
    :param df: output pandas.df from GeoDispatcher
    :return a new pandas.df
    """
    def get_EMEA(self, df):
        """Keep only EMEA model needed data fields.
        """
        if df['GEO_CODE'][0] == 'EMEA':
            DF = df[EMEA_FIELDS]
            DF.loc[:, 'GEO_CODE'] = 'EMEA'
            DF = deepcopy(DF)
            return DF
        else:
            pass
    
    def get_NA(self, df):
        """Keep only NA model needed data fields.
        """
        if df['GEO_CODE'][0] == 'NA':
            DF = df[NA_FIELDS]
            DF.loc[:, 'GEO_CODE'] = 'NA'
            DF = deepcopy(DF)
            return DF
        else:
            pass
        
    def get_NA_BP(self, df):
        """Keep only NA BP model needed data fields.
        :param df: quote (pandas.df)
        :return df_NA_BP (pandas.df), deepcopy should be applied.
        """
        if GeoDispatcher().is_BP(df): # For testing BP Quote
            df_NA_BP = df[NA_FIELDS_BP]
            df_NA_BP.loc[:,'GEO_CODE'] = 'NA'
            df_NA_BP = deepcopy(df_NA_BP)
            return df_NA_BP
        else:
            raise ValueError("The Quote is not of BP!")
            
    def get_JP(self, df):
        """Keep only JP model needed data fields.
        :param df: quote (pandas.df)
        :return DF (pandas.df), deepcopy should be applied.
        """
        if df['GEO_CODE'][0] == 'JP':
            DF = df[JP_FIELDS]
            DF.loc[:, 'GEO_CODE'] = 'JP'
            DF = deepcopy(DF)
            return DF
        else:
            pass
        
    def get_JP_BP(self, df):
        """Keep only JP BP model needed data fields.
        :param df: quote (pandas.df)
        :return df_JP_BP (pandas.df), deepcopy should be applied.
        """
        if GeoDispatcher().is_BP(df): # For testing BP Quote
            df_JP_BP = df[JP_FIELDS_BP]
            df_JP_BP.loc[:,'GEO_CODE'] = 'JP'
            df_JP_BP = deepcopy(df_JP_BP)
            return df_JP_BP
        else:
            raise ValueError("The Quote is not of BP!")
