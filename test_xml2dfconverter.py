# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:09:58 2018

@author: vinita
"""

import xml.etree.ElementTree as etree
import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from xml2dfconverter import XML2dfConverter
import os

cwd = os.getcwd()
path = os.path.join(os.path.dirname(__file__),'RequestXml_4154655-US.xml.001.lob')
pathcsv = os.path.join(os.path.dirname(__file__),'final_output.csv')
geomap = os.path.join(os.path.dirname(__file__),'Country_SubReg_Geo_Mapping.csv')

class TestXml2dfConverter(unittest.TestCase):
   """Convert XML to pandas dataframe
   :param xml: xml from http request
   :retur a new pandas.df
   
   """
   
   def test_xmltodf(self):
    # Convert XML to df
        quote_xml = etree.parse(path)
        classobj = XML2dfConverter()
        ComRevDivCd_orgi1,final_output_df,out_PricePoint_data = classobj.xmltodf(quote_xml,geomap)
        quote_df = pd.read_csv(pathcsv)
        quote_df.reset_index(drop = True, inplace = True)  
        final_output_df.reset_index(drop = True, inplace = True)  
        self.assertItemsEqual(final_output_df,quote_df)
                
if __name__ == '__main__':
    unittest.main()