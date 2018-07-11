"""Created on Sun Dec 03 22:25:40 2017

@author: vinita

"""
from config import *
import pandas as pd
import xml.etree.ElementTree as etree
import numpy as np
from copy import deepcopy

tag_list = TAG_LIST
tag_quote_list = TAG_QUOTE_LIST
tag_header_list = TAG_HEADER_LIST
PricePoint_tag_list = PRICEPOINT_TAG_LIST
CustomerInformation_tag_list = CUSTOMERINFORMATION_TAG_LIST
'''
output_columns = [
    'QuoteID', 'Countrycode', 'ChannelID', 'CustomerNumber', 'Quantity',
    'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'WinLoss',
    'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup', 'ComFamily', 'ComMT',
    'ComMTM', 'Year', 'Month', 'EndOfQtr', 'ClientSegCd', 'ClientSeg=E',
    'ComLogListPrice', 'Indirect(1/0)', 'UpgMES', 'ComQuotePricePofL',
    'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLowPofL', 'ComMedPofL',
    'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib', 'CCMScustomerNumber',
    'ModelID', 'RequestingApplicationID', 'Version', 'Componentid'
]
'''
# Assigned output_columns value with config file variable value
output_columns = OUTPUT_COLUMNS

RETURNCODE = [0]
ERRORCODE = [""]
ERRORDESCRIPTION = [""]


class XML2dfConverter:
    """Convert XML to pandas dataframe
    :param xml: xml from http request
    :retur a new pandas.df
    """

    def __init__(self):
        pass

    def get_com_info(self, com_get, node_list):
        info = []

        for tag in node_list:
            node = com_get.find(tag)

            try:
                info.append(node.text)
            except:
                RETURNCODE[0] = 1
                ERRORCODE[0] = 2
                ERRORDESCRIPTION[
                    0] = "Required data is missing for [" + tag + "] Tag"
                info.append(0)
        return info

    def getdata(self, node_find, node_list):
        # Get node data
        out_data = []
        for com in node_find:
            com_info = self.get_com_info(com, node_list)
            if com_info:
                out_data.append(com_info)
        return out_data

    def dataorganized(self, df_find, df_tag_list, out_data):
        # Organized the data and get component level information
        df_list = self.get_com_info(df_find, df_tag_list)
        df_list = pd.DataFrame(df_list).T
        df_list.columns = df_tag_list

        for i in np.arange(len(out_data)):
            df_list.ix[i] = df_list.ix[0]

        df_list.reset_index()
        return df_list

    def df_error(self):
        # Generate the error code if case of failure
        # Dynamic computes the columns length
        final_output = pd.DataFrame(
            np.arange(len(output_columns) * 2).reshape(2, len(output_columns)),
            columns=output_columns,
            dtype=object)
        #final_output["DealSize"] = np.array(
        # Assined QuoteID = "1-t" if input xml is not in correct format/data
        final_output['QuoteID'] = "1-t"
        data = {
            'PricePointId':
            [45, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 40, 41, 44, 46, 50, 53],
            'PricePointName': [
                "Original Quoted Price", "Optimal Price",
                "Internal Optimal Price", "Approved Price",
                "Delegation Level 0", "Delegation Level 1",
                "Delegation Level 2", "Delegation Level 3",
                "Delegation Level 4", "Delegation Level A", "TMC Breakeven",
                "Value Seller Price", "Best Price", "Warning free",
                "AGOG Special", "List Price", "P4V BP Proposed Price"
            ],
            'Price':
            [45, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 40, 41, 44, 46, 50, 53]
        }
        out_PricePoint_data = pd.DataFrame(data)
        return final_output, out_PricePoint_data

    def xmltodf(self, quote_xml, geomap):
        # Convert xml to dataframe
        try:
            root = etree.fromstring(quote_xml)
            #print ('INDIA++++++++++++++')
            #print (root)
            #input("test");

            RETURNCODE[0] = 0
            ERRORCODE[0] = ""
            ERRORDESCRIPTION[0] = ""

            QuoteInformation_find = root.find('QuoteInformation')
            #print (QuoteInformation_find)
            #print ('India')
            #input("test");
            RequestHeader_find = root.find('RequestHeader')
            component_find = QuoteInformation_find.getiterator('Component')
            #print ('Component+++++++')
            #input("test");
            PricePoint_find = QuoteInformation_find.getiterator('PricePoint')
            #print ('PricePoint')
            #print ('test')
            CustomerInformation_find = root.find('CustomerInformation')
            #print ('Customer')
            #input ("test");

            out_data = self.getdata(component_find, tag_list)
            out_data = pd.DataFrame(out_data)
            out_data.columns = tag_list
            #print (out_data)
            #input ("test");
            #print ('India')

            out_PricePoint_data = self.getdata(PricePoint_find,
                                               PricePoint_tag_list)
            out_PricePoint_data = pd.DataFrame(out_PricePoint_data)
            out_PricePoint_data.columns = PricePoint_tag_list

            QuoteInformation_list = self.dataorganized(
                QuoteInformation_find, tag_quote_list, out_data)
            #print (QuoteInformation_list)
            #print ('India')
            merge_output = pd.concat((QuoteInformation_list, out_data), axis=1)
            #print (merge_output)
            #print ('India')

            RequestHeader_list = self.dataorganized(
                RequestHeader_find, tag_header_list, merge_output)
            #print (RequestHeader_list)
            #print ('India')
            merge_output_RequestHeader = pd.concat(
                (RequestHeader_list, merge_output), axis=1)
            #print(merge_output_RequestHeader)
            #print ('India')

            CustomerInformation_list = self.dataorganized(
                CustomerInformation_find, CustomerInformation_tag_list,
                merge_output_RequestHeader)
            #print (CustomerInformation_list)
            #print ('India')
            final_output = pd.concat(
                (CustomerInformation_list, merge_output_RequestHeader), axis=1)
            #print (final_output)
            #input ("test");
            #print ('India')

            final_output[
                'quoteidnew'] = final_output['QuoteID'] + '-' + final_output['Countrycode']
            final_output['WinLoss'] = 1
            final_output['ComLowPofL'] = ''
            final_output['ComMedPofL'] = ''
            final_output['ComHighPofL'] = ''
            final_output['ComMedPrice'] = ''
            final_output['DealSize'] = ''
            final_output['LogDealSize'] = ''
            final_output['ComPctContrib'] = ''
            # print(final_output)
            #final_output.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/quote_df_out_xml2df.csv')
            # print ('India')

            final_output = pd.DataFrame(final_output, columns=COLUMNS)
           
            final_output.columns = output_columns

            final_output.reset_index()
            
            if (final_output["ComBrand"].any() != 0):
                final_output['ComBrand'] = final_output['ComBrand'].str.replace(
                    '&amp;amp;', '&')
                final_output['ComBrand'] = final_output['ComBrand'].str.replace(
                    '&amp;', '&')
           
            int64list = INT64LIST
            float64list = FLOAT64LIST

            for i in np.arange(len(int64list)):
                final_output[int64list[i]] = np.array(
                    final_output[int64list[i]], dtype='int64')
                
                
            for i in np.arange(len(float64list)):
                final_output[float64list[i]] = np.array(
                    final_output[float64list[i]], dtype='float64')
                    
            
        except:
            RETURNCODE[0] = 1
            ERRORCODE[0] = 1
            ERRORDESCRIPTION[0] = "Invalid XML Format"

            final_output, out_PricePoint_data = self.df_error()

        ### Hotfix - ComRevDivCd (temporal, need to remove when having permanent solution)
        ### get the original ComRevDivCd and hard-coded to dummy XYZ
        #final_output['ComRevDivCd'] = str(final_output['ComRevDivCd'])
        # final_output['ComRevDivCd'] = 'XYZ'

        # 10/03/2017, Jerry Yang
        # Hard-coded ComRevDivCd to Storage if in ['2D','2K','2W','U5','Y4','72'],
        # else hard-coded ComRevDivCd to Power
        final_output['ComRevDivCd'] = final_output['ComRevDivCd'].map(str)
                
        ComRevDivCd_orgi = '-'.join(final_output['ComRevDivCd'])

        ComRevDivCd_orgi2 = final_output[['ComRevDivCd', 'Componentid']]
        ComRevDivCd_orgi2 = pd.DataFrame(ComRevDivCd_orgi2)
        ComRevDivCd_orgi2.drop_duplicates(keep='first', inplace=True)

        #Hotfix fix is only implimented for EMEA
        geomap_df = geomap
        #if (final_output['Countrycode'].all() in [x for x in geomap_df['Countrycode']]):
        Country_Code = final_output['Countrycode'].all()
        #print (Country_Code)
        #print ('IBM+++++++++')
        geomap_df = geomap_df.fillna(
            'NA'
        )  #Since Python was considering NA country code as NAN/blank. Hence, replacing NAN/blank with NA.
        result_df = geomap_df[(geomap_df['Countrycode'] == Country_Code)
                              & (geomap_df['Geo'].isin(['EMEA', 'NA','JP']))] #Hotfix implimented for Japan
        final_output['ComRevDivCd_Orig'] = final_output['ComRevDivCd']
        if (len(result_df) != 0):
            final_output.loc[final_output.ComRevDivCd.isin([
                '2D', '2K', '2W', 'U5', 'Y4', '72'
            ]), 'ComRevDivCd'] = 'Storage'
            final_output.loc[~(final_output.ComRevDivCd == 'Storage'),
                             'ComRevDivCd'] = 'Power'
        
        ComRevDivCd_orgi1 = deepcopy(ComRevDivCd_orgi)
        
        if final_output.empty:
            final_output = final_output.head(1)

        final_output1 = deepcopy(final_output)
        out_PricePoint_data1 = deepcopy(out_PricePoint_data)

        return ComRevDivCd_orgi1, ComRevDivCd_orgi2, final_output1, out_PricePoint_data1
