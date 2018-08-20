# -*- coding: utf-8 -*-
"""
Created on Mon Dec 4 12:04:06 2017
@author: SaiNaveen
Read fixed-discounted products table to pandas.df.
"""

import pandas as pd

class FixedProdTableReader:
    """Read fixed-discounted products table to pandas.df.
    """

    def __init__(self):
        pass


    def read_NA(self, fixed_discounts_path):
        """Returning a fixed discount data frame for NA
        :param table_file: path of the fixed-discounted products table
        :return a new pandas.df
        """

        discount_rules_na = pd.read_csv(fixed_discounts_path, low_memory=False)
        return discount_rules_na

    def read_EMEA(self, fixed_discounts_path):
        """Returning a fixed discount data frame for EMEA
        :param table_file: path of the fixed-discounted products table
        :return a new pandas.df
        """

        discount_rules_emea = pd.read_csv(fixed_discounts_path, low_memory=False)
        return discount_rules_emea
    
    def read_JP(self, fixed_discounts_path):
        """Returning a fixed discount data frame for JP
        :param table_file: path of the fixed-discounted products table
        :return a new pandas.df
        """
    
        discount_rules_jp = pd.read_csv(fixed_discounts_path, low_memory=False)
        return discount_rules_jp
