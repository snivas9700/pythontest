"""Created on Sun Dec 03 22:25:40 2017

@author: vinita
"""

import unittest
from df2xmlconverter import Df2XMLConverter
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os
from contextlib import contextmanager

path = os.path.join(os.path.dirname(__file__),'Repoter_output.csv')
path_xml = os.path.join(os.path.dirname(__file__),'xmlfile.xml')

class TestDf2XmlConverter(unittest.TestCase):
    """Convert pandas.df to XML for http reponse

    :param: df: output pandas.df from Reporter
    :return xml for http reponse
    
    """  

    def test_df_to_xml(self):
        df = pd.read_csv(path)
        xml_file_object = open(path_xml,'r')
        xml = xml_file_object.read()
        classobj = Df2XMLConverter(df,'quote_df_out_response.xml')
        xml_expected_str = classobj.df_to_xml(df,'quote_df_out_response.xml')
        self.assertEqual(xml, xml_expected_str)
        xml_file_object.close()
        
if __name__ == '__main__':
    unittest.main()
