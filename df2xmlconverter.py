"""Created on Sun Dec 03 22:25:40 2017

@author: vinita

"""

import numpy as np
import pandas as pd
from xml.dom.minidom import Document
from xml.dom.minidom import parseString
from config import *
from geodispatcher import GeoDispatcher
import re

RETURNCODE = [0]
ERRORCODE = [""]
ERRORDESCRIPTION = [""]

class Df2XMLConverter:
    """Convert pandas.df to XML for http reponse

    :param: df: output pandas.df from Reporter
    :return xml for http reponse
    
    """
    def __init__(self):
        pass
        #self._quote_df = quote_df
        #self._total_deal_df = total_deal_df
        #self._price_point_df = price_point_df
        #self._filename = filename

    def to_xml(self, df, name):
        def row_to_xml(row):
            if name == 'Component':
                xml = ['<Component>']
            elif name == 'PricePoint':
                xml = ['<PricePoint>']
            else:            
                xml = ['<TssComponents>']
                
            for i, col_name in enumerate(row.index):
                xml.append('<{0}>{1}</{0}>'.format(col_name, row.iloc[i]))
            if name == 'Component':
                xml.append('</Component>')
            elif name == 'PricePoint':
                xml.append('</PricePoint>')
            else:
                xml.append('</TssComponents>')
            return ''.join(xml)

        res = ''.join(df.apply(row_to_xml, axis=1))
        return res   
    # Convert dataframe to xml
    def df_to_xml(self,quote_df,total_deal_df,price_point_df,filename):
        
        """ Rename clientseg=E col & Indirect """
        quote_df.rename(columns={'ClientSeg=E':'ClientSeg_E', 
                                 'Indirect(1/0)' : 'Indirect_1_0',
                                 'QuotePrice' : 'QuotedPrice'}, inplace=True)
        
        #This section is to handle special characters 
        if ('DomBuyerGrpName' in quote_df ):
            quote_df['DomBuyerGrpName'] = quote_df['DomBuyerGrpName'].apply(lambda x : re.sub (r'([^a-zA-Z0-9.\-\: ]+?)',' ', str(x) ) )
        doc = Document()
        base = doc.createElement('iPATQuoteXML')
        doc.appendChild(base)
        #print (quote_df['Componentid'])
        #print ("DDDDDDDDDDDDDDDDDDDDDD")

        #quote_df.to_csv('./output_results/mka_dfxml.csv')
        quote_df['CustomerNumber'] = quote_df['CustomerNumber'].astype(str)
        Id_Country =quote_df.loc[0,'QuoteID']
        Id = Id_Country.split('-')
        #print ('hello vinita')
        #print (Id)
        #print(Id_Country)
        #print (total_deal_df)
        #print (price_point_df)

        xmlheader = ['<ResponseHeader>']
        xmlheader.append('<QuoteID>{0}</QuoteID>'.format(Id[0]))
        xmlheader.append('<Countrycode>{0}</Countrycode>'.format(Id[1]))
        xmlheader.append('<ModelID>{0}</ModelID>'.format(quote_df.loc[0,'ModelID']))
        xmlheader.append('<ReturnCode>{0}</ReturnCode>'.format(RETURNCODE[0]))
        xmlheader.append('<RequestingApplicationID>{0}</RequestingApplicationID>'.format(quote_df.loc[0,'RequestingApplicationID']))
        xmlheader.append('<Version>{0}</Version>'.format(quote_df.loc[0,'Version']))
        xmlheader.append('<ErrorCode>{0}</ErrorCode>'.format(ERRORCODE[0]))
        xmlheader.append('<ErrorDescription>{0}</ErrorDescription>'.format(ERRORDESCRIPTION[0]))
        xmlheader.append('</ResponseHeader>')
        xmlheadernew = ''.join(xmlheader)
        ResponseHeader = parseString(xmlheadernew).childNodes[0]
        base.appendChild(ResponseHeader)
    
        xml0 = []
        xml0.append('<Countrycode>{0}</Countrycode>'.format(Id[1]))
        xml0.append('<Channelidcode>{0}</Channelidcode>'.format(quote_df.loc[0,'ChannelID']))
        xml0.append('<Year>{0}</Year>'.format(quote_df.loc[0,'Year']))
        xml0.append('<Month>{0}</Month>'.format(quote_df.loc[0,'Month']))
        xml0.append('<EndOfQtr>{0}</EndOfQtr>'.format(quote_df.loc[0,'EndOfQtr']))
        xml0.append('<Indirect>{0}</Indirect>'.format(quote_df.loc[0,'Indirect_1_0']))
        if 'TSSComponentincluded' in quote_df:
            if quote_df['TSSComponentincluded'][0] == 'Y':            
                xml0.append('<TSSComponentincluded>{0}</TSSComponentincluded>'.format(quote_df.loc[0,'TSSComponentincluded']))
                xml0.append('<TSSContstartdate>{0}</TSSContstartdate>'.format(quote_df.loc[0,'TSSContstartdate']))
                xml0.append('<TSSContenddate>{0}</TSSContenddate>'.format(quote_df.loc[0,'TSSContenddate']))
                xml0.append('<TSSContduration>{0}</TSSContduration>'.format(quote_df.loc[0,'TSSContduration']))
                xml0.append('<TSSPricerefdate>{0}</TSSPricerefdate>'.format(quote_df.loc[0,'TSSPricerefdate']))
                xml0.append('<TSSPricoptdescription>{0}</TSSPricoptdescription>'.format(quote_df.loc[0,'TSSPricoptdescription']))
                xml0.append('<TSSFrameOffering>{0}</TSSFrameOffering>'.format(quote_df.loc[0,'TSSFrameOffering']))
                xml0.append('<ChangePeriodStopDate>{0}</ChangePeriodStopDate>'.format(quote_df.loc[0,'ChangePeriodStopDate']))
        xml0out = ''.join(xml0)
    
        xml1 = ['<TotalGeneralQuoteData>']
        xml1.append('<DealListPrice>{0}</DealListPrice>'.format(total_deal_df[1]))
        xml1.append('<DealSize>{0}</DealSize>'.format(total_deal_df[2]))
        xml1.append('<DealTMC>{0}</DealTMC>'.format(total_deal_df[3]))
        xml1.append('<DealPredictedQuotePrice>{0}</DealPredictedQuotePrice>'.format(total_deal_df[4]))
        xml1.append('</TotalGeneralQuoteData>')
        xml1out = ''.join(xml1)
    
        xml2 = []
        xml2.append('<Deal_Adj_LowPrice>{0}</Deal_Adj_LowPrice>'.format(total_deal_df[6]))
        xml2.append('<Deal_Adj_MedPrice>{0}</Deal_Adj_MedPrice>'.format(total_deal_df[7]))
        xml2.append('<Deal_Adj_HighPrice>{0}</Deal_Adj_HighPrice>'.format(total_deal_df[8]))
        xml2.append('<COP_LowPrice>0</COP_LowPrice>')
        xml2.append('<COP_MedPrice>0</COP_MedPrice>')
        xml2.append('<COP_HighPrice>0</COP_HighPrice>')
        xml2out = ''.join(xml2)
    
        xml3 = ['<TotalDealPricePoint>']
        xml3.append('<PricePointId>101</PricePointId>')
        xml3.append('<PricePointName>LineItemSumQuotedPrice</PricePointName>')
        xml3.append('<Price>{0}</Price>'.format(total_deal_df[10]))
        xml3.append('<WinProbPercent>{0}</WinProbPercent>'.format(total_deal_df[11]))
        xml3.append('<GP>{0}</GP>'.format(total_deal_df[12]))
        xml3.append('<ExpectedGP>{0}</ExpectedGP>'.format(total_deal_df[13]))
        xml3.append('</TotalDealPricePoint>')
        xml3out = ''.join(xml3)
    
        xml4 = ['<TotalDealPricePoint>']
        xml4.append('<PricePointId>102</PricePointId>')
        xml4.append('<PricePointName>LineItemSumOptimalPrice</PricePointName>')
        xml4.append('<Price>{0}</Price>'.format(total_deal_df[15]))
        xml4.append('<WinProbPercent>{0}</WinProbPercent>'.format(total_deal_df[16]))
        xml4.append('<GP>{0}</GP>'.format(total_deal_df[17]))
        xml4.append('<ExpectedGP>{0}</ExpectedGP>'.format(total_deal_df[18]))
        xml4.append('<IntervalLow>{0}</IntervalLow>'.format(total_deal_df[19]))
        xml4.append('<IntervalHigh>{0}</IntervalHigh>'.format(total_deal_df[20]))
        xml4.append('</TotalDealPricePoint>')
        xml4out = ''.join(xml4)
    
        xml5 = ['<TotalDealPricePoint>']
        xml5.append('<PricePointId>103</PricePointId>')
        xml5.append('<PricePointName>BottomLineQuotedPrice</PricePointName>')
        xml5.append('<Price>{0}</Price>'.format(total_deal_df[22]))
        xml5.append('<WinProbPercent>{0}</WinProbPercent>'.format(total_deal_df[23]))
        xml5.append('<GP>{0}</GP>'.format(total_deal_df[24]))
        xml5.append('<ExpectedGP>{0}</ExpectedGP>'.format(total_deal_df[25]))
        xml5.append('</TotalDealPricePoint>')
        xml5out = ''.join(xml5)
    
        xml6 = ['<TotalDealPricePoint>']
        xml6.append('<PricePointId>104</PricePointId>')
        xml6.append('<PricePointName>BottomLineOptimalPrice</PricePointName>')
        xml6.append('<Price>{0}</Price>'.format(total_deal_df[27]))
        xml6.append('<WinProbPercent>{0}</WinProbPercent>'.format(total_deal_df[28]))
        xml6.append('<GP>{0}</GP>'.format(total_deal_df[29]))
        xml6.append('<ExpectedGP>{0}</ExpectedGP>'.format(total_deal_df[30]))
        xml6.append('<IntervalLow>{0}</IntervalLow>'.format(total_deal_df[31]))
        xml6.append('<IntervalHigh>{0}</IntervalHigh>'.format(total_deal_df[32]))
        xml6.append('</TotalDealPricePoint>')
        xml6out = ''.join(xml6)
    
        xml7 = ['<CustomerInformation>']
        xml7.append('<customerNumber>{0}</customerNumber>'.format(quote_df.loc[0,'CustomerNumber']))
        xml7.append('<clientSegCode>{0}</clientSegCode>'.format(quote_df.loc[0,'ClientSegCd']))
        xml7.append('<clientSeg>{0}</clientSeg>'.format(quote_df.loc[0,'ClientSeg_E']))
        xml7.append('</CustomerInformation>')
        xml7out = ''.join(xml7)
        
        delcolumns = ['RequestingApplicationID',"Version",'ModelID','QuoteID','ChannelID','Year','Month','EndOfQtr','ClientSegCd','CustomerNumber']
        #delcolumns = ['RequestingApplicationID',"Version",'ModelID','QuoteID','ChannelID','Year','Month','EndOfQtr','Indirect_1_0','ClientSeg_E','ClientSegCd','CustomerNumber']
        #delcolumns = ['QuoteID','ChannelID','Year','Month','EndOfQtr','Indirect_1_0','ClientSeg_E','ClientSegCd']
        #for i in np.arange(len(delcolumns)):
        #    del quote_df[delcolumns[i]]
        ## Removing for loop above to delete cols writing code below using inbuilt pandas
        quote_df = quote_df.drop(quote_df.loc[:,delcolumns].head(0).columns, axis=1)
        
        #print (quote_df.columns)

        #Correction made to address the issue of special characters in ComFamily column.
        #if (quote_df["ComBrand"].any() != 0):
        #    quote_df['ComFamily'] = quote_df['ComFamily'].str.replace(
        #        '&amp;amp;', '&')
        #
        #    quote_df['ComFamily'] = quote_df['ComFamily'].str.replace(
        #            '&amp;', '&')
        #    quote_df['ComFamily'] = quote_df['ComFamily'].str.replace(
        #            '&', '')

        mask = quote_df.applymap(lambda x: x == 'None' or x is None)
        cols = quote_df.columns[(mask).any()]
        for col in quote_df[cols]:
            quote_df.loc[mask[col], col] = ''
        quote_df = quote_df.replace(np.nan, '', regex=True)    
        #if (quote_df['GEO_CODE'][0] == 'EMEA') & (not GeoDispatcher().is_EMEA_old(quote_df)):
        if 'TSSComponentincluded' in quote_df and quote_df['TSSComponentincluded'][0] == 'Y':
            #if quote_df['TSSComponentincluded'][0] == 'Y':            
            tss_component_df = quote_df.filter(np.unique(TSS_COMPONENTS_TAG_LIST + TSS_RETURN_FIELDS), axis=1)
            tss_component_df = tss_component_df[tss_component_df.ParentMapping_ComponentID.notnull()]
            tss_component_df = tss_component_df[tss_component_df.ParentMapping_ComponentID != '']
            tss_component_df['servlvldesc'] = tss_component_df['servlvldesc'].str.replace("&amp; ", '')
            tss_component_df['servlvldesc'] = tss_component_df['servlvldesc'].str.replace("&", '')
            del tss_component_df['ParentMapping_ComponentID']
            tss_component_df = tss_component_df.replace(np.nan, '', regex=True)
            tss_component_df = tss_component_df.rename(columns=lambda n: n.replace('TSS_', ''))
            #tss_component_df.to_csv('./output_results/tss_df.csv')
            quote_df.to_csv('./output_results/df2xml_pre.csv')
            quote_df = quote_df.drop(quote_df.loc[:,DELETE_LIST].head(0).columns, axis=1) 
            quote_df = quote_df.drop_duplicates(subset=['Componentid'],keep='first')
            del quote_df['ParentMapping_ComponentID']
            #quote_df.to_csv('./output_results/df2xml.csv')
            tss_component_df.to_csv('./output_results/df2xml_tss.csv')
            series1 = parseString('<QuoteInformation>' + xml0out  + 
                              self.to_xml(quote_df,'Component') +
                              self.to_xml(tss_component_df,'TssComponents')+
                              self.to_xml(price_point_df,'PricePoint') + 
                              '<TotalDealStatistics>' + 
                              xml2out  + 
                              xml1out  +
                              xml3out  + 
                              xml4out  + 
                              xml5out  + 
                              xml6out  + 
                              '</TotalDealStatistics>' +
                              '</QuoteInformation>').childNodes[0]
                
        else:
            #if 'TSSComponentincluded' in quote_df:
                #if quote_df['TSSComponentincluded'][0] == 'N':
            #quote_df = quote_df.drop(quote_df.loc[:,DELETE_LIST].head(0).columns, axis=1)
            series1 = parseString('<QuoteInformation>' + xml0out  + 
                              self.to_xml(quote_df,'Component') +
                              self.to_xml(price_point_df,'PricePoint') + 
                              '<TotalDealStatistics>' + 
                              xml2out  + 
                              xml1out  +
                              xml3out  + 
                              xml4out  + 
                              xml5out  + 
                              xml6out  + 
                              '</TotalDealStatistics>' +
                              '</QuoteInformation>').childNodes[0]
                        
        #component_df = quote_df.filter(REPORTER_COLUMNS, axis=1)
        #quote_df.to_csv("./output_results/quote_df_xml_check.csv")
        
        base.appendChild(series1)
    
        series2 = parseString(xml7out).childNodes[0]
        base.appendChild(series2)
    
        return doc.toprettyxml(encoding="UTF-8")
