"""Created on Sun Dec 03 22:25:40 2017

@author: vinita

"""

import numpy as np
import pandas as pd
from xml.dom.minidom import Document
from xml.dom.minidom import parseString

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
            else:
                xml = ['<PricePoint>']
            for i, col_name in enumerate(row.index):
                xml.append('<{0}>{1}</{0}>'.format(col_name, row.iloc[i]))
            if name == 'Component':
                xml.append('</Component>')
            else:
                xml.append('</PricePoint>')
            return ''.join(xml)

        res = ''.join(df.apply(row_to_xml, axis=1))
        return res   
    # Convert dataframe to xml
    def df_to_xml(self,quote_df,total_deal_df,price_point_df,filename):
        
        doc = Document()
        base = doc.createElement('iPATQuoteXML')
        doc.appendChild(base)
        #print (quote_df['Componentid'])
        #print ("DDDDDDDDDDDDDDDDDDDDDD")

        Id_Country =quote_df.iloc[0,0]
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
    
        delcolumns = ['RequestingApplicationID',"Version",'ModelID','QuoteID','ChannelID','Year','Month','EndOfQtr','Indirect_1_0','ClientSeg_E','ClientSegCd','CustomerNumber']
        #delcolumns = ['QuoteID','ChannelID','Year','Month','EndOfQtr','Indirect_1_0','ClientSeg_E','ClientSegCd']
        for i in np.arange(len(delcolumns)):
            del quote_df[delcolumns[i]]
        #print (quote_df.columns)

        #Correction made to address the issue of special characters in ComFamily column.
        if (quote_df["ComBrand"].any() != 0):
            quote_df['ComFamily'] = quote_df['ComFamily'].str.replace(
                '&amp;amp;', '&')
		    
            quote_df['ComFamily'] = quote_df['ComFamily'].str.replace(
                    '&amp;', '&')
            quote_df['ComFamily'] = quote_df['ComFamily'].str.replace(
                    '&', '')
            
        series1 = parseString('<QuoteInformation>' + xml0out  + self.to_xml(quote_df,'Component') + self.to_xml(price_point_df,'PricePoint') + '<TotalDealStatistics>' + xml2out  + xml1out  + xml3out  + xml4out  + xml5out  + xml6out + '</TotalDealStatistics>' +'</QuoteInformation>').childNodes[0]
        base.appendChild(series1)
    
        series2 = parseString(xml7out).childNodes[0]
        base.appendChild(series2)
    
        return doc.toprettyxml(encoding="UTF-8")
