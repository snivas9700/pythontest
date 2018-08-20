# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 04:51:20 2018

@author: harish
"""
import pandas as pd
import copy
        
class HWValidator():
    """
    Split dataframe to two sets: fixed-discounted/validator products and non-
    fixed-discounted/validator products
    """
    def __init__(self):
        pass

    def detect_and_split(self, discounts_df , quote_df, geomap):
        """
        Split dataframe to two sets: fixed-discounted/validator products and non-
        fixed-discounted/validator products
        :param discounts_df: output pandas.df from HWReader
        :param quote_df: output pandas.df from DataFieldsFilter
        :param geomap: output pandas.df from GeomapTableReader
        :return (a new pandas.df for fixed/validator, a new pandas.df for non-fixed)
        """
        if quote_df.empty:
            fixed_quotes1 = copy.deepcopy(quote_df)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1

        if discounts_df.empty:
            fixed_quote_columns = quote_df.columns.tolist()
            fixed_quote_columns.append("max_price")
            fixed_quote_columns.append("min_price")
            fixed_quotes1 = pd.DataFrame(columns=fixed_quote_columns)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1
        
        ## Run & store the Maximum value condition and if empty populate with -999
        exec_var = {}
        fixed_quotes = pd.DataFrame()
        discounts_df_copy = copy.deepcopy(discounts_df)
        quote_df_copy = copy.deepcopy(quote_df)
        df = quote_df_copy
        exec_var['df'] = quote_df_copy
        exec_var['geomap'] = geomap
         
        discounts_df_copy.MaxCondition = discounts_df_copy.MaxCondition.fillna(-999)
        
        for discount in discounts_df_copy.itertuples():
            # Evaluates the expression by using df and geomap.
            # Ex. df.loc[(df.ComMT == '6911a') & (df.CountryCode.isin(geomap.loc[(geomap.Geo == 'JP'), 'CountryCode']))]
            exec("%s=%s" % ("item", discount.Condition), exec_var)
            exec("%s=%s" % ("max_price", discount.MaxCondition), exec_var)
            exec_var["item"]['max_price'] = exec_var["max_price"]
            fixed_quotes = fixed_quotes.append(exec_var["item"])
        
        #Store the maximum values
        max_price_stored = fixed_quotes.max_price
        
        ## Run & store the Minimum value condition and if empty populate with -999
        exec_var = {}
        fixed_quotes = pd.DataFrame()
        discounts_df_copy = copy.deepcopy(discounts_df)
        quote_df_copy = copy.deepcopy(quote_df)
        df = quote_df_copy
        exec_var['df'] = quote_df_copy
        exec_var['geomap'] = geomap
         
        discounts_df_copy.MinCondition = discounts_df_copy.MinCondition.fillna(-999)
        
        for discount in discounts_df_copy.itertuples():
            # Evaluates the expression by using df and geomap.
            # Ex. df.loc[(df.ComMT == '6911a') & (df.CountryCode.isin(geomap.loc[(geomap.Geo == 'JP'), 'CountryCode']))]
            exec("%s=%s" % ("item", discount.Condition), exec_var)
            exec("%s" % (discount.MinCondition), exec_var)
            fixed_quotes = fixed_quotes.append(exec_var["item"])
        
        #Assign back the max_value
        fixed_quotes['max_price'] = max_price_stored

        quote_df_values = set(df.index.values)
        fixed_quote_values = set(fixed_quotes.index.values)
        non_fixed_quotes = df.iloc[list(quote_df_values - fixed_quote_values), :]

        #  Data Frame needs to reset index since some records are removed
        non_fixed_quotes.reset_index(drop=True, inplace=True)
        fixed_quotes1 = copy.deepcopy(fixed_quotes)
        non_fixed_quotes1 = copy.deepcopy(non_fixed_quotes)

        return fixed_quotes1, non_fixed_quotes1
