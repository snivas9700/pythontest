"""
Created on Thu Dec 04 12:04:06 2017
@author: SaiNaveen
"""

import pandas as pd
#from copy import deepcopy
import copy

class FixedProdDetector:
    """
    Split dataframe to two sets: fixed-discounted products and non-
    fixed-discounted products
    """

    def __init__(self):
        pass


    def split_EMEA(self, discounts_df, quote_df, geomap):
        """
        :param discounts_df: output pandas.df from FixedProdTableReader
        :param quote_df: output pandas.df from DataFiledsFilter
        :param geomap: output pandas.df from GeomapTableReader
        :retur (a new pandas.df for fixed, a new pandas.df for non-fixed)
        """
        if quote_df.empty:
            fixed_quotes1 = copy.deepcopy(quote_df)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1

        if discounts_df.empty:
            fixed_quote_columns = quote_df.columns.tolist()
            fixed_quote_columns.append("DiscountedPrice")
            fixed_quotes1 = pd.DataFrame(columns=fixed_quote_columns)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1

        exec_var = {}
        fixed_quotes = pd.DataFrame()
        discounts_df_copy = copy.deepcopy(discounts_df)
        quote_df_copy = copy.deepcopy(quote_df)
        df = quote_df_copy
        exec_var['df'] = quote_df_copy
        exec_var['geomap'] = geomap



        for discount in discounts_df_copy.itertuples():
            # Evaluates the expression by using df and geomap.
            # Ex. df.loc[(df.ComMT == '6911a') & (df.CountryCode.isin(geomap.loc[(geomap.Geo == 'EMEA'), 'CountryCode']))]
            exec("%s=%s" % ("item", discount.Condition), exec_var)
            exec("%s=%s" % ("discounted_price", discount.DiscountedPrice), exec_var)
            exec_var["item"]['DiscountedPrice'] = exec_var["discounted_price"]
            fixed_quotes = fixed_quotes.append(exec_var["item"])

        #fixed_quotes = fixed_quotes.drop_duplicates(subset=['ComMT', 'ComMTM'], keep='last')
        #fixed_quotes = fixed_quotes.drop_duplicates(subset=['Componentid'], keep='last')

        quote_df_values = set(df.index.values)
        fixed_quote_values = set(fixed_quotes.index.values)
        non_fixed_quotes = df.iloc[list(quote_df_values - fixed_quote_values), :]

        #  Data Frame needs to reset index since some records are removed
        non_fixed_quotes.reset_index(drop=True, inplace=True)
        fixed_quotes1 = copy.deepcopy(fixed_quotes)
        non_fixed_quotes1 = copy.deepcopy(non_fixed_quotes)


        return fixed_quotes1, non_fixed_quotes1

    def split_NA(self, discounts_df, quote_df, geomap):
        """
        :param discounts_df: output pandas.df from FixedProdTableReader
        :param quote_df: output pandas.df from DataFiledsFilter
        :param geomap: output pandas.df from GeomapTableReader
        :retur (a new pandas.df for fixed, a new pandas.df for non-fixed)
        """

        if quote_df.empty:
            fixed_quotes1 = copy.deepcopy(quote_df)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1

        if discounts_df.empty:
            fixed_quote_columns = quote_df.columns.tolist()
            fixed_quote_columns.append("DiscountedPrice")
            fixed_quotes1 = pd.DataFrame(columns=fixed_quote_columns)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1

        exec_var = {}
        fixed_quotes = pd.DataFrame()
        discounts_df_copy = copy.deepcopy(discounts_df)
        quote_df_copy = copy.deepcopy(quote_df)
        df = quote_df_copy
        exec_var['df'] = quote_df_copy
        exec_var['geomap'] = geomap
        #print(exec_var['geomap'])
        #print ('IBM************')


        for discount in discounts_df_copy.itertuples():
            # Evaluates the expression by using df and geomap.
            # Ex. df.loc[(df.ComMT == '6911a') & (df.CountryCode.isin(geomap.loc[(geomap.Geo == 'EMEA'), 'CountryCode']))]
            exec("%s=%s" % ("item", discount.Condition), exec_var)
            exec("%s=%s" % ("discounted_price", discount.DiscountedPrice), exec_var)
            exec_var["item"]['DiscountedPrice'] = exec_var["discounted_price"]
            fixed_quotes = fixed_quotes.append(exec_var["item"])

        #fixed_quotes = fixed_quotes.drop_duplicates(subset=['ComMT', 'ComMTM'], keep='last')
        fixed_quotes = fixed_quotes.drop_duplicates(subset=['Componentid'], keep='last')
        #fixed_quotes.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/fixed_quotesprod.csv')

        quote_df_values = set(df.index.values)
        fixed_quote_values = set(fixed_quotes.index.values)
        non_fixed_quotes = df.iloc[list(quote_df_values - fixed_quote_values), :]

        #  Data Frame needs to reset index since some records are removed
        non_fixed_quotes.reset_index(drop=True, inplace=True)
        fixed_quotes1 = copy.deepcopy(fixed_quotes)
        non_fixed_quotes1 = copy.deepcopy(non_fixed_quotes)


        return fixed_quotes1, non_fixed_quotes1
    
    def split_JP(self, discounts_df, quote_df, geomap):
        """
        Split dataframe to two sets: fixed-discounted products and non-
        fixed-discounted products for Japan
        :param discounts_df: output pandas.df from FixedProdTableReader
        :param quote_df: output pandas.df from DataFiledsFilter
        :param geomap: output pandas.df from GeomapTableReader
        :return (a new pandas.df for fixed, a new pandas.df for non-fixed)
        """
        if quote_df.empty:
            fixed_quotes1 = copy.deepcopy(quote_df)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1

        if discounts_df.empty:
            fixed_quote_columns = quote_df.columns.tolist()
            fixed_quote_columns.append("DiscountedPrice")
            fixed_quotes1 = pd.DataFrame(columns=fixed_quote_columns)
            non_fixed_quotes1 = copy.deepcopy(quote_df)

            return fixed_quotes1, non_fixed_quotes1

        exec_var = {}
        fixed_quotes = pd.DataFrame()
        discounts_df_copy = copy.deepcopy(discounts_df)
        quote_df_copy = copy.deepcopy(quote_df)
        df = quote_df_copy
        exec_var['df'] = quote_df_copy
        exec_var['geomap'] = geomap

        for discount in discounts_df_copy.itertuples():
            # Evaluates the expression by using df and geomap.
            # Ex. df.loc[(df.ComMT == '6911a') & (df.CountryCode.isin(geomap.loc[(geomap.Geo == 'JP'), 'CountryCode']))]
            exec("%s=%s" % ("item", discount.Condition), exec_var)
            exec("%s=%s" % ("discounted_price", discount.DiscountedPrice), exec_var)
            exec_var["item"]['DiscountedPrice'] = exec_var["discounted_price"]
            fixed_quotes = fixed_quotes.append(exec_var["item"])

        quote_df_values = set(df.index.values)
        fixed_quote_values = set(fixed_quotes.index.values)
        non_fixed_quotes = df.iloc[list(quote_df_values - fixed_quote_values), :]

        #  Data Frame needs to reset index since some records are removed
        non_fixed_quotes.reset_index(drop=True, inplace=True)
        fixed_quotes1 = copy.deepcopy(fixed_quotes)
        non_fixed_quotes1 = copy.deepcopy(non_fixed_quotes)

        return fixed_quotes1, non_fixed_quotes1
