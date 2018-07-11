# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:42:42 2017
@author: Ravindranath
"""

import numpy as np
import pandas as pd
import copy
from copy import deepcopy


class ProdConsolidator:
    """
    Consolidate same products to one row (1xN) in quote.
    """

    def __init__(self):
        pass

    def consolidate(self, df):
        """
        :param df: output pandas.df from XML2dfConverter
        :retur a new pandas.df
        """

        if df.empty:
            df['x'] = 0
            df = df.head(-1)
            del df['x']

        else:
            df['X'] = range(len(df))
            df1 = df[['X', 'Componentid']]
            df['Componentid'] = 1
            
            #df['Converted_to_1XN'] = 0
            #df.reset_index()
            df['concat0'] = pd.Series(df.fillna('').values.tolist())
    
            for i in range(len(df)):
                df.loc[i, 'concat'] = ''.join(
                    str(elt) for elt in df.loc[i, 'concat0'])
    
            del df['concat0']
    
            grp0 = pd.DataFrame(df['concat'])
            grp0['count'] = len(grp0)
    
            for i in range(len(grp0)):
                grp0.loc[i, 'concat'] = grp0.loc[i, 'concat'].strip()
                grp0.loc[i, 'concat'] = grp0.loc[i, 'concat'].replace(" ", "")
    
            for i in range(len(df)):
                df.loc[i, 'concat'] = df.loc[i, 'concat'].strip()
                df.loc[i, 'concat'] = df.loc[i, 'concat'].replace(" ", "")
    
            grp1 = grp0.groupby('concat').sum()
            grp2 = pd.merge(df, grp1, left_on='concat', right_index=True)
            df = grp2
            #df.to_csv('C:/Users/IBM_ADMIN/Desktop/1XN/df.csv')
    
            for i in range(len(df)):
                if (df.loc[i, 'Quantity'] == 1) & (df.loc[i, 'count'] > 1):
                    df.loc[i, 'Quantity'] = df.loc[i, 'count']
                    #df.loc[i, 'Converted_to_1XN'] = 1
    
            del df['count']
            df = df.drop_duplicates(subset=['concat'], keep='first')
            del df['concat']            
            del df['Componentid']
            
            df1 = df1.drop_duplicates(subset=['X'], keep='first')            
        
            df = pd.merge(df, df1, on='X', how='left')
            del df['X']            

            df = deepcopy(df)

        return df


#df_in = pd.read_csv('C:/Users/IBM_ADMIN/Desktop/1XN/quote_df_request_prod.csv')

#df_out = ProdConsolidator().consolidate(df_in)

#df_out.to_csv('C:/Users/IBM_ADMIN/Desktop/1XN/quote_df_modified.csv')

