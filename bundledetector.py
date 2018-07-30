"""
Created on Wed Nov 22 13:23:31 2017

@author: Chaitra
"""

import pandas as pd
import os
import re
import copy
from copy import deepcopy
import numpy as np

class BundleDetector:
    """
    Split dataframe to two sets: bundled products and non-bundled products.
    """

    def __init__(self):
        pass

    def detect(self, bundle_df, quote_df):
        """
        :param df: output pandas.df from DataFieldsFilter
        :param df: output from BundleTableReader
        :retur (a new pandas.df for bundled, a new pandas.df for non-bundled)
        """
        # Create empty BundleFlag column in Quote_df
        quote_df['Bundle'] = ''
        
        if quote_df.empty:
            quote_df['ComMTM'] = quote_df.ComMTM
        else:
            quote_df['ComMTM'] = quote_df.ComMTM.str.strip()
            
        bundle_df['ComMTM'] = bundle_df.ComMTM.str.strip()

        # Reset index for Quote_df and bundle_table
        quote_df.reset_index(drop=True, inplace=True)
        
        # Merge BundleFlag from bundle_table to quote_df based on ComMTM ,COM_CATEGORY columns match
        for i in range(len(quote_df)):
            for j in range(len(bundle_df)):
                if (quote_df.loc[i, 'ComMTM'] == bundle_df.loc[j, 'ComMTM'] + 'a'
                    ) and (quote_df.loc[i, 'ComRevCat'] ==
                           bundle_df.loc[j, 'ComRevCat'][0]):
                    quote_df.loc[i, 'Bundle'] = bundle_df.loc[j,'Bundle']
                elif (quote_df.loc[i, 'ComMTM'][0:4] ==
                      bundle_df.loc[j, 'ComMTM']) and (
                          quote_df.loc[i, 'ComRevCat'] ==
                          bundle_df.loc[j, 'ComRevCat'][0]):
                    quote_df.loc[i, 'Bundle'] = bundle_df.loc[j,'Bundle']
                    
                quote_df.loc[quote_df.Bundle=='','Bundle'] = 'N'    
                
        # Split Quote_df to bundle_table, nonbundle_table based on BundleFlag
                
        bundle_table1  = quote_df[quote_df['Bundle']!= 'N']
        nonbundle_table1  = quote_df[quote_df['Bundle']== 'N']
        
        bundle_table1=pd.DataFrame(bundle_table1)
        nonbundle_table1=pd.DataFrame(nonbundle_table1)
        
        if bundle_table1.empty:
            bundle_table1 = bundle_table1.head(1)
        else:
            bundle_table1 = bundle_table1
            
        if nonbundle_table1.empty:
            nonbundle_table1 = nonbundle_table1.head(1)
        else:
            nonbundle_table1 = nonbundle_table1
        
        bundle_table1.reset_index(drop=True,inplace=True)
        nonbundle_table1.reset_index(drop=True,inplace=True)
                
        bundle_table = deepcopy(bundle_table1)        
        nonbundle_table = deepcopy(nonbundle_table1)
        
        #bundle_table.to_csv(data_path + 'bundle_table.csv', index=False)
        #nonbundle_table.to_csv(data_path + 'nonbundle_table.csv', index=False)
        
        return bundle_table , nonbundle_table
        
