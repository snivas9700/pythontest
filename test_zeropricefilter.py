"""Created on Sun Dec 03 22:25:40 2017

@author: vinita
"""

import unittest
from zeropricefilter import ZeroPriceFilter
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os

path = os.path.join(os.path.dirname(__file__),'df.csv')
path_csv_zero = os.path.join(os.path.dirname(__file__),'zero.csv')
path_csv_normal = os.path.join(os.path.dirname(__file__),'normal.csv')

class TestZeroPriceFilter(unittest.TestCase):
    """Filter out zero list prices from pandas.df.

    :param df: output pandas.df from ProdConsolidator
    :return (a new pandas.df for zero prices, a new pandas.df for the rest (normal))

    """
    def test_zero_price(self):
        df = pd.read_csv(path)
        classobj = ZeroPriceFilter(df)
        df_zero_price,df_normal = classobj.zero_price(df)
        df_zero_price.reset_index(drop = True, inplace = True)  
        df_normal.reset_index(drop = True, inplace = True)  
        zero = pd.read_csv(path_csv_zero)
        normal = pd.read_csv(path_csv_normal)
        assert_frame_equal(df_zero_price,zero)
        assert_frame_equal(df_normal,normal)
    
if __name__ == '__main__':
    unittest.main()
