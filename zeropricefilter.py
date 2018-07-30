"""Created on Sun Dec 03 22:25:40 2017

@author: vinita
"""

import pandas as pd
from copy import deepcopy
import numpy as np
from config import *

class ZeroPriceFilter:
    """
    Filter out zero list prices from pandas.df.

    :param df: output pandas.df from ProdConsolidator
    :return (a new pandas.df for zero prices, a new pandas.df for the rest (normal))

    """

    def __init__(self):
        pass
        #self._df = df

    def zero_price(self, df):
        if df.empty:
            df = df.head(1)
            df_zero_price1 = df
            df1 = df
            df_zero_price1 = pd.DataFrame(df_zero_price1)
            df1 = pd.DataFrame(df1)
        else:
            df_zero_price_0 = df[df["ComListPrice"] == 0]
            df1 = df[df["ComListPrice"] != 0]
            df1 = pd.DataFrame(df1, columns = NON_ZERO_PRICE)
            df_zero_price_0 = pd.DataFrame(df_zero_price_0)
    
            df_zero_price_0['GEO_CODE'] = np.nan
            df_zero_price_0['TreeNode'] = 0
            df_zero_price_0['ComLowPofL'] = 0
            df_zero_price_0['ComMedPofL'] = 0
            df_zero_price_0['ComHighPofL'] = 0
            df_zero_price_0['ComTMCPofL'] = 0
            df_zero_price_0['AdjComLowPofL'] = 0
            df_zero_price_0['AdjComMedPofL'] = 0
            df_zero_price_0['AdjComHighPofL'] = 0
            df_zero_price_0['AdjComLowPrice'] = 0
            df_zero_price_0['AdjComMedPrice'] = 0
            df_zero_price_0['AdjComHighPrice'] = 0
            df_zero_price_0['OptimalPricePofL'] = 0
            df_zero_price_0['OptimalPrice'] = 0
            df_zero_price_0['OptimalPriceWinProb'] = 0
            df_zero_price_0['OptimalPriceGP'] = 0
            df_zero_price_0['OptimalPriceExpectedGP'] = 0
            df_zero_price_0['OptimalPriceIntervalLow'] = 0
            df_zero_price_0['OptimalPriceIntervalHigh'] = 0
            df_zero_price_0['DealBotLineSpreadOptimalPrice'] = 0
            df_zero_price_0['QuotePricePofL'] = 0
            df_zero_price_0['QuotePrice'] = 0
            df_zero_price_0['QuotePriceWinProb'] = 0
            df_zero_price_0['QuotePriceGP'] = 0
            df_zero_price_0['QuotePriceExpectedGP'] = 0
            df_zero_price_0['PredictedQuotePricePofL'] = 0
            df_zero_price_0['PredictedQuotePrice'] = 0
            df_zero_price_0['COPComLowPrice'] = 0
            df_zero_price_0['COPComMedPrice'] = 0
            df_zero_price_0['COPComHighPrice'] = 0
            df_zero_price_0['COPComLowPofL'] = 0
            df_zero_price_0['COPComMedPofL'] = 0
            df_zero_price_0['COPComHighPofL'] = 0
            df_zero_price_0['COPOptimalPrice'] = 0
            df_zero_price_0['COPOptimalPricePofL'] = 0
            df_zero_price_0['COPOptimalPriceWinProb'] = 0
            df_zero_price_0['COPOptimalPriceGP'] = 0
            df_zero_price_0['COPOptimalPriceExpectedGP'] = 0
            df_zero_price_0['COPOptimalPriceIntervalLow'] = 0
            df_zero_price_0['COPOptimalPriceIntervalHigh'] = 0
            df_zero_price_0['COPQuotePriceWinProb'] = 0
            df_zero_price_0['COPQuotePriceGP'] = 0
            df_zero_price_0['COPQuotePriceExpectedGP'] = 0
    
            df_zero_price_0.reset_index(drop=True, inplace=True)
            df1.reset_index(drop=True, inplace=True)
                
            df_zero_price1 = pd.DataFrame(df_zero_price_0, columns = ZERO_PRICE)
        df_zero_price = deepcopy(df_zero_price1)
        df = deepcopy(df1)
        return df_zero_price, df
