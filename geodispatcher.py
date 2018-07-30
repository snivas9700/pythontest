# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:04:06 2017
@author: Munish
"""
import pandas as pd
from copy import deepcopy

class GeoDispatcher:
    """
    class to  find the geography of the quotes, identify whether a quote
    is a BP or a direct quote. 
    
    dipatch: finds the geography of the quotes. Receives a DF as input and 
    adds a new column geo to the InputDF and returns the DF to main 
    function/class.
    
    is_BP : identifies whether a quote is BP quote. Receives a DF as input and 
    adds a new column BP_QUOTE to the InputDF and returns the DF to main 
    function/class.
    
    id_direct : identifies whether a quote is a direct quote. Receives a DF as 
    input and adds a new column BP_QUOTE to the InputDF and returns the DF to 
    main  function/class.
    """
    def __init__(self):
        pass
    
    def dispatch(self, input_df):
        """
        Main Function to find the geo based on ISO country
        :param df: output pandas.df from ZeroPriceFilter
        :return a new pandas.df
        """
        if not input_df.empty:
            if 'Countrycode' in input_df.columns:
                #output_df = copy.deepcopy(input_df)
                input_df['GEO_CODE'] = input_df['Countrycode'].apply(
                    lambda x: 'NA' if (x == 'CA' or x == 'US') else 'JP' if (x =='JP') else 'EMEA') #added Japan country code
                output_df = deepcopy(input_df)
                return output_df
            else:
                input_df['Countrycode'] = ""
                input_df['GEO_CODE'] = ""
                return deepcopy(input_df)
        else:
            input_df['Countrycode'] = ""
            input_df['GEO_CODE'] = ""
            return deepcopy(input_df)
        
    def is_BP(self, input_df):
        """Test if quote (pandas.df) is BP.
        :param df: quote
        :retur True if BP quote, False otherwise.
        """
        if input_df['QuoteType'][0] == 'B':
            return True
        else:
            return False

    def is_direct(self, input_df):
        """Test if quote (pandas.df) is direct.
        :param df: quote
        :retur True if direct quote, False otherwise.
        """
        if input_df["Indirect"][0] == 0:
            return True
        else:
            return False
