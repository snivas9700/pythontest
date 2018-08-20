"""Created on Sun Dec 03 22:25:40 2017

@author: vinita

"""
from config import *
import pandas as pd
import xml.etree.ElementTree as etree
import numpy as np
from copy import deepcopy

tag_list = TAG_LIST
#added "TSSComponentincluded" +++++++++++++++++++++++++++++++++++++++++++++
tag_quote_list = TAG_QUOTE_LIST
tag_quote_list_old = TAG_QUOTE_LIST_OLD
tag_header_list = TAG_HEADER_LIST
PricePoint_tag_list = PRICEPOINT_TAG_LIST
CustomerInformation_tag_list = CUSTOMERINFORMATION_TAG_LIST
#added TSS_COMPONENTS_TAG_LIST +++++++++++++++++++++++++++++++++++++++++++++
Tss_components_tag_list = TSS_COMPONENTS_TAG_LIST

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
        ComRevDivCd_orgi2 = pd.DataFrame()
        return final_output, out_PricePoint_data

    def xmltodf(self, quote_xml, geomap):
        # Convert xml to dataframe
        try:
            root = etree.fromstring(quote_xml)
            TSSComponentincluded = False
            TssTagNotPresent = False
            RETURNCODE[0] = 0
            ERRORCODE[0] = ""
            ERRORDESCRIPTION[0] = ""

            QuoteInformation_find = root.find('QuoteInformation')
            
            print(QuoteInformation_find)
            
            RequestHeader_find = root.find('RequestHeader')
                                    
            component_find = QuoteInformation_find.getiterator('Component')
                        
            PricePoint_find = QuoteInformation_find.getiterator('PricePoint')

            CustomerInformation_find = root.find('CustomerInformation')
            
            if QuoteInformation_find.find('TSSComponentincluded') is None:
                TssTagNotPresent = True
            elif QuoteInformation_find.find('TSSComponentincluded').text == 'Y':
                TSSComponentincluded = True
            else:
                TSSComponentincluded = False
            
            out_data = self.getdata(component_find, tag_list)
            out_data = pd.DataFrame(out_data)
            out_data.columns = tag_list  
            """
            if TSSComponentincluded:
                out_data = out_data.copy()           
                out_data = pd.DataFrame(out_data)            
                del out_data['Level_1']
                del out_data['Level_2']
                del out_data['Level_3']
                del out_data['Level_4']
            """   
            #out_data.to_csv('./output_results/out_data.csv')
            
            out_PricePoint_data = self.getdata(PricePoint_find,
                                               PricePoint_tag_list)
            out_PricePoint_data = pd.DataFrame(out_PricePoint_data)
            out_PricePoint_data.columns = PricePoint_tag_list
            #out_PricePoint_data.to_csv('./output_results/out_PricePoint_data.csv') 
            
            if TssTagNotPresent:
                QuoteInformation_list = self.dataorganized(
                    QuoteInformation_find, tag_quote_list, out_data)
            elif TSSComponentincluded:
                QuoteInformation_list = self.dataorganized(
                    QuoteInformation_find, tag_quote_list + TAG_QUOTE_TSS_LIST, out_data)
            else:
                QuoteInformation_list = self.dataorganized(
                    QuoteInformation_find, TAG_QUOTE_LIST + ["TSSComponentincluded"], out_data)
                        
            merge_output = pd.concat((QuoteInformation_list,out_data), axis=1)
            #merge_output.to_csv('./output_results/merge_output1.csv')
                                                                                                 
            RequestHeader_list = self.dataorganized(
                RequestHeader_find, tag_header_list, merge_output)
            #RequestHeader_list.to_csv('./output_results/RequestHeader_list.csv') 
            
            merge_output_RequestHeader = pd.concat(
                (RequestHeader_list, merge_output), axis=1)
            
            CustomerInformation_list = self.dataorganized(
                CustomerInformation_find, CustomerInformation_tag_list,
                merge_output_RequestHeader)
            
            final_output11 = pd.concat(
                (CustomerInformation_list, merge_output_RequestHeader), axis=1)  
            #final_output11.to_csv('./output_results/final_output11.csv')

            if TSSComponentincluded:
            
            #if (TSSComponentincluded) and final_output11['TSSComponentincluded'][0]== 'Y':
            #if final_output11['TSSComponentincluded'][0]== 'Y':
                
                TssComponents_find = QuoteInformation_find.getiterator('TssComponents')
                                
                #added TSS_COMPONENTS_TAG_LIST +++++++++++++++++++++++++++++++++++++++++++++
                out_TSS_data = self.getdata(TssComponents_find,
                                            Tss_components_tag_list)
                #print (TSS_COMPONENTS_TAG_LIST)
                out_TSS_data = pd.DataFrame(out_TSS_data)
                
                out_TSS_data.columns = Tss_components_tag_list
                #out_TSS_data.to_csv('./output_results/xml_out_TSS_data.csv')
                
                out_TSS_data["ParentMapping_ComponentID"] = out_TSS_data.Componentid
                                                                
                final_output_TSS = pd.merge(final_output11, out_TSS_data, on='Componentid', how='left')
                                
                final_output_TSS['quoteidnew'] = final_output_TSS['QuoteID'] + '-' + final_output_TSS['Countrycode']
                del final_output_TSS['QuoteID']
                final_output_TSS['WinLoss'] = 1
                final_output_TSS['ComLowPofL'] = ''
                final_output_TSS['ComMedPofL'] = ''
                final_output_TSS['ComHighPofL'] = ''
                final_output_TSS['ComMedPrice'] = ''
                final_output_TSS['DealSize'] = ''
                final_output_TSS['LogDealSize'] = ''
                final_output_TSS['ComPctContrib'] = ''
            
                #final_output = pd.DataFrame(final_output, columns=COLUMNS)
                #final_output2 = pd.DataFrame(final_output_TSS)   
                final_output = pd.DataFrame(final_output_TSS).rename(columns=RENAME_COL)
                
                final_output['ComRevDivCd'] = final_output['ComRevDivCd'].map(str) 
                #final_output.to_csv('./output_results/TSSinc_out.csv')
        
                ComRevDivCd_orgi = '-'.join(final_output['ComRevDivCd'])
                
                final_output[['TSS_quantity','TSScomid','PTI0']] = final_output[['TSS_quantity','TSScomid','PTI0']].fillna(value=0)
                # Replace NaN values for TSS col with 0
                final_output[['basecharge','committedcharge','totalcharge','CMDAprice','Cost',
                              "coverage_hours_days",	"coverage_hours",	"coverage_days",
                              "sl_cntct",	"sl_fix_time",	"sl_onsite",	"sl_part_time"]] = final_output[['basecharge','committedcharge','totalcharge','CMDAprice','Cost',
                                                                                     "coverage_hours_days",	"coverage_hours",	"coverage_days","sl_cntct",	"sl_fix_time",	"sl_onsite",	"sl_part_time"]].fillna(value=0.0)
                #final_output.to_csv('./output_results/nann.csv')
                            
                int64list = INT64LIST_TSS 
                float64list = FLOAT64LIST_TSS
    
                for i in np.arange(len(int64list)):
                    final_output[int64list[i]] = np.array(
                        final_output[int64list[i]], dtype='int64')
                                    
                for i in np.arange(len(float64list)):
                    final_output[float64list[i]] = np.array(
                        final_output[float64list[i]], dtype='float64')             
                final_output.reset_index()   
                
                ComRevDivCd_orgi2 = final_output[['ComRevDivCd', 'Componentid','TSScomid']]

                #final_output = final_output2.copy().rename(columns=RENAME_COL)
                            
            #if TSSCompincluded tag not present in final_output     
            #elif (final_output11['TSSComponentincluded'][0] == 'N') or TssTagNotPresent:
            else:
            #elif TssTagNotPresent:    
                final_output_NTSS = final_output11
                                   
                final_output_NTSS['quoteidnew'] = final_output_NTSS['QuoteID'] + '-' + final_output_NTSS['Countrycode']
                del final_output_NTSS['QuoteID']
                
                final_output_NTSS['WinLoss'] = 1
                final_output_NTSS['ComLowPofL'] = ''
                final_output_NTSS['ComMedPofL'] = ''
                final_output_NTSS['ComHighPofL'] = ''
                final_output_NTSS['ComMedPrice'] = ''
                final_output_NTSS['DealSize'] = ''
                final_output_NTSS['LogDealSize'] = ''
                final_output_NTSS['ComPctContrib'] = ''
                
                #final_output = pd.DataFrame(final_output, columns=COLUMNS)
                #final_output3 = pd.DataFrame(final_output_NTSS)   
                #final_output = final_output3.copy().rename(columns=RENAME_COL)
                final_output = pd.DataFrame(final_output_NTSS).rename(columns=RENAME_COL)
                #final_output.to_csv('./output_results/non_TSS.csv')
                
                final_output['ComRevDivCd'] = final_output['ComRevDivCd'].map(str) 
                #final_output.to_csv('./output_results/TSSinc_out.csv')
        
                ComRevDivCd_orgi = '-'.join(final_output['ComRevDivCd'])

                int64list = INT64LIST
                float64list = FLOAT64LIST
    
                for i in np.arange(len(int64list)):
                    final_output[int64list[i]] = np.array(
                        final_output[int64list[i]], dtype='int64')
                                    
                for i in np.arange(len(float64list)):
                    final_output[float64list[i]] = np.array(
                        final_output[float64list[i]], dtype='float64')
    
                final_output.reset_index()    
                ComRevDivCd_orgi2 = final_output[['ComRevDivCd', 'Componentid']]

                #final_output.columns = output_columns
                #print ("['TSSComponentincluded'][0] == 'N')")
                
            #final_output.to_csv('./output_results/TSSinc.csv')
            final_output['ComBrand'] = final_output['ComBrand'].map(str)                                             
            final_output['ComBrand'] = final_output['ComBrand'].str.replace('&amp;amp;', '&')
            final_output['ComBrand'] = final_output['ComBrand'].str.replace('&amp;', '&')
            #final_output.to_csv('./output_results/post.csv')
        except:
            RETURNCODE[0] = 1
            ERRORCODE[0] = 1
            ERRORDESCRIPTION[0] = "Invalid XML Format"

            final_output, out_PricePoint_data, ComRevDivCd_orgi2 = self.df_error()
                                                                                                            
        ### Hotfix - ComRevDivCd (temporal, need to remove when having permanent solution)
        ### get the original ComRevDivCd and hard-coded to dummy XYZ
        #final_output['ComRevDivCd'] = str(final_output['ComRevDivCd'])
        # final_output['ComRevDivCd'] = 'XYZ'
              
        # 10/03/2017, Jerry Yang
        # Hard-coded ComRevDivCd to Storage if in ['2D','2K','2W','U5','Y4','72'],
        # else hard-coded ComRevDivCd to Power
             
        ComRevDivCd_orgi2 = pd.DataFrame(ComRevDivCd_orgi2)
        ComRevDivCd_orgi2.drop_duplicates(keep='first', inplace=True)

        #final_output.to_csv('./output_results/post_out.csv')
        #ComRevDivCd_orgi2.to_csv('./output_results/post_out_comp.csv')
                
        #Hotfix fix is only implimented for EMEA
        geomap_df = geomap
        #if (final_output['Countrycode'].all() in [x for x in geomap_df['Countrycode']]):
        Country_Code = final_output['Countrycode'].all()
        
        geomap_df = geomap_df.fillna('NA')  #Since Python was considering NA country code as NAN/blank. Hence, replacing NAN/blank with NA.
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
