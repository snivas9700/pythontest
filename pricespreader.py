# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:47:10 2018

@author: Ravindranath
"""
import numpy as np
import pandas as pd
import copy
from copy import deepcopy
import BasePricingFunctions as BPF

class PriceSpreader:
    """ populate the total deal bottom line 
        optimal price to the individual line items.  The ensures that in addition
        to each line item's optimal price, there will also be a field where
        the bottom line optimal price is spread to the line items.
    """

    def __init__(self):
          pass
    
    def spread_botline_optimal_price(self, COP_l=0, COP_m=0, COP_h=0, *quote_df):
        
        quote_df_all = pd.concat(quote_df).reset_index(drop=True)
        
        if quote_df_all.empty:
            quote_df_all['DealBotLineSpreadOptimalPrice'] = 0
            
            quote_df_all = quote_df_all.head(-1)
            quote_df_all1 = deepcopy(quote_df_all)
            return quote_df_all1
            
        else:                       
            total_deal_stats = pd.Series('', index=['  General Total Quote Data'])
            total_deal_stats['DealListPrice'] = quote_df_all['ComListPrice'].sum()
            #quote_df_all.to_csv('C:/Users/IBM_ADMIN/Desktop/My folder/NA Region/Github Classes/quote_df_all.csv')
            total_deal_stats['DealSize'] = quote_df_all['ComMedPrice'].sum()
            total_deal_stats['DealTMC'] = quote_df_all['ComTMC'].sum()
            ##print '*PredictedQuotePrice: ',quote_df_all['PredictedQuotePrice']
            total_deal_stats['DealPredictedQuotePrice'] = quote_df_all['PredictedQuotePrice'].sum() 
            #  this section contains Price Range Data (Line Item Sum)
            total_deal_stats['  Price Range Data (Line Item Sum)'] = ''
            total_deal_stats['DealAdjLowPrice'] = quote_df_all['AdjComLowPrice'].sum() 
            total_deal_stats['DealAdjMedPrice'] = quote_df_all['AdjComMedPrice'].sum() 
            total_deal_stats['DealAdjHighPrice'] = quote_df_all['AdjComHighPrice'].sum() 
            #  this section contains Quoted Price Data (Line Item Sum)
            total_deal_stats['  Quoted Price Data (Line Item Sum)'] = ''
            total_deal_stats['DealQuotePrice'] = quote_df_all['ComQuotePrice'].sum()
            total_deal_stats['DealQuotePriceWinProb'] = ''
            total_deal_stats['DealQuotePriceGP'] = total_deal_stats['DealQuotePrice'] - total_deal_stats['DealTMC']
            total_deal_stats['DealQuotePriceExpectedGP'] = quote_df_all['QuotePriceExpectedGP'].sum()
            total_deal_stats['DealQuotePriceWinProb'] = total_deal_stats['DealQuotePriceExpectedGP'] / total_deal_stats['DealQuotePriceGP']
            #  this section contains optimal price data
            total_deal_stats['  Optimal Price Data (Line Item Sum)'] = ''
            total_deal_stats['DealOptimalPrice'] = quote_df_all['OptimalPrice'].sum() 
            total_deal_stats['DealOptimalPriceWinProb'] = ''
            total_deal_stats['DealOptimalPriceGP'] = quote_df_all['OptimalPriceGP'].sum()
            total_deal_stats['DealOptimalPriceExpectedGP'] = quote_df_all['OptimalPriceExpectedGP'].sum()
            total_deal_stats['DealOptimalPriceWinProb'] = 0 #    
            if total_deal_stats['DealOptimalPriceGP'] != 0:#
                total_deal_stats['DealOptimalPriceWinProb'] = total_deal_stats['DealOptimalPriceExpectedGP'] / total_deal_stats['DealOptimalPriceGP']#
            total_deal_stats['DealOptimalPriceIntervalLow'] = quote_df_all['OptimalPriceIntervalLow'].sum() 
            total_deal_stats['DealOptimalPriceIntervalHigh'] = quote_df_all['OptimalPriceIntervalHigh'].sum() 
            #  this section contains Quoted Price Data (Bottom-Line)
            total_deal_stats['  Quoted Price Data (Bottom-Line)'] = ''
            total_deal_stats['DealBotLineQuotePrice'] = total_deal_stats['DealQuotePrice']
            total_deal_stats['DealBotLineQuotePriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealBotLineQuotePrice'], total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'])
            total_deal_stats['DealBotLineQuotePriceGP'] = total_deal_stats['DealBotLineQuotePrice'] - total_deal_stats['DealTMC']
            total_deal_stats['DealBotLineQuotePriceExpectedGP'] = total_deal_stats['DealBotLineQuotePriceGP'] * total_deal_stats['DealBotLineQuotePriceWinProb']
            #  this section contains Optimal Price Data (Bottom-Line)
            total_deal_stats['  Optimal Price Data (Bottom-Line)'] = ''
            total_deal_stats['DealBotLineOptimalPrice'] = BPF.OptPrice(total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'], total_deal_stats['DealTMC'], 0, 0, 0)
            total_deal_stats['DealBotLineOptimalPriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealBotLineOptimalPrice'], total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'])
            total_deal_stats['DealBotLineOptimalPriceGP'] = total_deal_stats['DealBotLineOptimalPrice'] - total_deal_stats['DealTMC']
            total_deal_stats['DealBotLineOptimalPriceExpectedGP'] = total_deal_stats['DealBotLineOptimalPriceGP'] * total_deal_stats['DealBotLineOptimalPriceWinProb']
            total_deal_stats['DealBotLineOptimalPriceIntervalLow'], total_deal_stats['DealBotLineOptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(total_deal_stats['DealBotLineOptimalPrice'], total_deal_stats['DealAdjLowPrice'], total_deal_stats['DealAdjMedPrice'], total_deal_stats['DealAdjHighPrice'], total_deal_stats['DealTMC'])
            
            #this section is executed only if customized optimal pricing (COP) is needed
            if (COP_h > COP_m) and (COP_m > COP_l):
                #  this section contains COP Price Range Data (Line Item Sum)
                total_deal_stats['  COP Price Range Data (Line Item Sum)'] = ''
                total_deal_stats['DealCOPLowPrice'] = quote_df_all['COPComLowPrice'].sum() 
                total_deal_stats['DealCOPMedPrice'] = quote_df_all['COPComMedPrice'].sum() 
                total_deal_stats['DealCOPHighPrice'] = quote_df_all['COPComHighPrice'].sum() 
                #  this section contains COP Quote Price Data (Line Item Sum)
                total_deal_stats['  COP Quote Price Data (Line Item Sum)'] = ''
                total_deal_stats['DealCOPQuotePrice'] = quote_df_all['ComQuotePrice'].sum()
                total_deal_stats['DealCOPQuotePriceWinProb'] = ''
                total_deal_stats['DealCOPQuotePriceGP'] = quote_df_all['COPQuotePriceGP'].sum() 
                total_deal_stats['DealCOPQuotePriceExpectedGP'] = quote_df_all['COPQuotePriceExpectedGP'].sum() 
                total_deal_stats['DealCOPQuotePriceWinProb'] = total_deal_stats['DealCOPQuotePriceExpectedGP'] / total_deal_stats['DealCOPQuotePriceGP']
                #  this section contains COP Optimal Price Data (Line Item Sum)
                total_deal_stats['  COP Optimal Price Data (Line Item Sum)'] = ''
                total_deal_stats['DealCOPOptimalPrice'] = quote_df_all['COPOptimalPrice'].sum() 
                total_deal_stats['DealCOPOptimalPriceWinProb'] = ''
                total_deal_stats['DealCOPOptimalPriceGP'] = quote_df_all['COPOptimalPriceGP'].sum()
                total_deal_stats['DealCOPOptimalPriceExpectedGP'] = quote_df_all['COPOptimalPriceExpectedGP'].sum()
                total_deal_stats['DealCOPOptimalPriceWinProb'] = total_deal_stats['DealCOPOptimalPriceExpectedGP'] / total_deal_stats['DealCOPOptimalPriceGP']
                total_deal_stats['DealCOPOptimalPriceIntervalLow'] = quote_df_all['COPOptimalPriceIntervalLow'].sum() 
                total_deal_stats['DealCOPOptimalPriceIntervalHigh'] = quote_df_all['COPOptimalPriceIntervalHigh'].sum() 
                #  this section contains quoted price data within the Customized Optimal Price (COP) estimates
                total_deal_stats['  COP Quote Price Data (Bottom-Line)'] = ''
                total_deal_stats['DealCOPBotLineQuotePrice'] = total_deal_stats['DealQuotePrice']
                total_deal_stats['DealCOPBotLineQuotePriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealCOPQuotePrice'], total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'])
                total_deal_stats['DealCOPBotLineQuotePriceGP'] = total_deal_stats['DealCOPBotLineQuotePrice'] - total_deal_stats['DealTMC']
                total_deal_stats['DealCOPBotLineQuotePriceExpectedGP'] = total_deal_stats['DealCOPBotLineQuotePriceGP'] *total_deal_stats['DealCOPBotLineQuotePriceWinProb']
                #  this section contains COP Optimal Price Data (Bottom-Line)
                total_deal_stats['  COP Optimal Price Data (Bottom-Line)'] = ''
                total_deal_stats['DealCOPBotLineOptimalPrice'] = BPF.OptPrice(total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'], total_deal_stats['DealTMC'], 0, 0, 0) 
                total_deal_stats['DealCOPBotLineOptimalPriceWinProb'] = BPF.ProbOfWin(total_deal_stats['DealCOPBotLineOptimalPrice'], total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'])
                total_deal_stats['DealCOPBotLineOptimalPriceGP'] = total_deal_stats['DealCOPBotLineOptimalPrice'] - total_deal_stats['DealTMC']
                total_deal_stats['DealCOPBotLineOptimalPriceExpectedGP'] = total_deal_stats['DealCOPBotLineOptimalPriceGP'] * total_deal_stats['DealCOPBotLineOptimalPriceWinProb']
                total_deal_stats['DealCOPBotLineOptimalPriceIntervalLow'], total_deal_stats['DealCOPBotLineOptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(total_deal_stats['DealCOPBotLineOptimalPrice'], total_deal_stats['DealCOPLowPrice'], total_deal_stats['DealCOPMedPrice'], total_deal_stats['DealCOPHighPrice'], total_deal_stats['DealTMC'])
    
            #spread the bottom line optimal price to the line items    
            quote_df_all = self.spread_optimal_price(quote_df_all, total_deal_stats)
            quote_df_all1 = copy.deepcopy(quote_df_all)
            return quote_df_all1
        
    def spread_optimal_price(self, quote_df_all, total_deal_stats):
        """quote_df_all = quote dataframe, total_deal_stats = total deal statistics dataframe"""
    
    # This section extracts the total deal bottom line optimal price
        total_deal_stats = total_deal_stats.to_dict()
        optimal_price = total_deal_stats['DealBotLineOptimalPrice']
            
        # This section creates a spread_price dataframe of component price price points & includes a total row
        spread_price = quote_df_all.loc[:, ('AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'ComListPrice')]  # vasu 7/26/17  Defect # 1534724
    
        spread_price.insert(0, 'ZeroPrice', 0)
        spread_price['10XPrice'] = spread_price['ComListPrice'] * 10
        spread_price.loc['Total'] = spread_price.sum().values
    
        # This section creates an adj_spread_price dataframe that removes "mountains"
        # (i.e for price points to left of optimal price: take lowest price of current column up to the optimal price column
        #      for price points to right of optimal price: take highest price of current column down to the optimal price column)
        adj_spread_price = pd.DataFrame()
        spread_columns = spread_price.columns
        adj_spread_price['ZeroPrice'] = spread_price.loc[:, spread_columns].min(axis=1)
        adj_spread_price['AdjComLowPrice'] = spread_price.loc[:, spread_columns[1:3]].min(axis=1)
        adj_spread_price['AdjComMedPrice'] = spread_price['AdjComMedPrice']   # vasu 7/26/17  Defect # 1534724
        adj_spread_price['AdjComHighPrice'] = spread_price.loc[:, spread_columns[2:4]].max(axis=1)
        adj_spread_price['ComListPrice'] = spread_price.loc[:, spread_columns[2:5]].max(axis=1)
        adj_spread_price['10XPrice'] = spread_price['10XPrice']
        adj_spread_price = adj_spread_price[:-1]
        adj_spread_price.loc['Total'] = adj_spread_price.sum().values
    
       # This section selects the lower column of the two columns used to perform the linear interpolation between    
        adj_points = [ 1 if ((adj_spread_price.loc['Total']['ZeroPrice'] < optimal_price) &(adj_spread_price.loc['Total']['AdjComLowPrice'] >= optimal_price)) else 0,
                       1 if ((adj_spread_price.loc['Total']['AdjComLowPrice'] < optimal_price) &(adj_spread_price.loc['Total']['AdjComMedPrice'] >= optimal_price)) else 0,
                       1 if ((adj_spread_price.loc['Total']['AdjComMedPrice'] < optimal_price) &(adj_spread_price.loc['Total']['AdjComHighPrice'] >= optimal_price)) else 0,
                       1 if ((adj_spread_price.loc['Total']['AdjComHighPrice'] < optimal_price) &(adj_spread_price.loc['Total']['ComListPrice'] >= optimal_price)) else 0,
                       1 if ((adj_spread_price.loc['Total']['ComListPrice'] < optimal_price) &(adj_spread_price.loc['Total']['10XPrice'] >= optimal_price)) else 0,
                       1 if (adj_spread_price.loc['Total']['10XPrice'] < optimal_price) else 0 ]  # vasu 7/26/17  Defect # 1534724
        weight_df = pd.DataFrame(adj_points, index=['ZeroPrice', 'AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'ComListPrice', '10XPrice'])   # vasu 7/26/17  Defect # 1534724
    
        # This section spreads the bottom line optimal price to the line items
        spread_mechanism = pd.DataFrame()
        spread_mechanism['lower_price'] = adj_spread_price.iloc[:, 0:5].dot(weight_df.iloc[0:5, :])[0]
        weight_df.index = ['AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'ComListPrice', '10XPrice', 'ZeroPrice']  # vasu 7/26/17  Defect # 1534724
        spread_mechanism['higher_price'] = adj_spread_price.iloc[:, 1:6].dot(weight_df.iloc[0:5, :])[0]
        total_lower = spread_mechanism.loc['Total']['lower_price']
        total_higher = spread_mechanism.loc['Total']['higher_price']
        spread_value = ((optimal_price - total_lower) / (total_higher - total_lower)) if (total_higher - total_lower) != 0 else 0
        spread_mechanism['spread_price'] = spread_mechanism['lower_price'] + (spread_mechanism['higher_price'] - spread_mechanism['lower_price'])*spread_value
    
        # This section loads the spread optimal prices to the quote_df_all dataframe
        quote_df_all['DealBotLineSpreadOptimalPrice'] = spread_mechanism['spread_price']
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than DealBotLineSpreadOptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if (quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] > quote_df_all.loc[i, 'ComListPrice']):
                quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] = quote_df_all.loc[i, 'ComListPrice']
        #quote_df_all.to_csv(data_path + 'spread_optimal_price.csv', index=False)
                
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than DealBotLineSpreadOptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if ((quote_df_all.loc[i, 'ComTMC'] < quote_df_all.loc[i, 'ComListPrice']) & (quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] < quote_df_all.loc[i, 'ComTMC'])):
                quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'] = quote_df_all.loc[i, 'ComTMC']
        
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than OptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if (quote_df_all.loc[i, 'OptimalPrice'] > quote_df_all.loc[i, 'ComListPrice']):
                quote_df_all.loc[i, 'OptimalPrice'] = quote_df_all.loc[i, 'ComListPrice']
        #quote_df_all.to_csv(data_path + 'spread_optimal_price.csv', index=False)
                
        for i in range(len(quote_df_all)): #since ComListPrice was coming greater than OptimalPrice. So, to correct that capping done on 26th Mar 2018.
            if ((quote_df_all.loc[i, 'ComTMC'] < quote_df_all.loc[i, 'ComListPrice']) & (quote_df_all.loc[i, 'OptimalPrice'] < quote_df_all.loc[i, 'ComTMC'])):
                quote_df_all.loc[i, 'OptimalPrice'] = quote_df_all.loc[i, 'ComTMC']
        #Optimal price is less ComTMC and ComTMC is less then ListPrice then cap optimal price with ComTMC and recalucate the intervals
        #for i in range(len(quote_df_all)):
         #   quote_df_all.loc[i, 'OptimalPriceIntervalLow'], quote_df_all.loc[i, 'OptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'OptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
        
        for i in range(len(quote_df_all)):
            quote_df_all.loc[i, 'L1'], quote_df_all.loc[i, 'H1'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'OptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
            quote_df_all.loc[i, 'L2'], quote_df_all.loc[i, 'H2'] = BPF.OptPriceConfIntervl(quote_df_all.loc[i, 'DealBotLineSpreadOptimalPrice'], quote_df_all.loc[i,'AdjComLowPrice'], quote_df_all.loc[i,'AdjComMedPrice'], quote_df_all.loc[i,'AdjComHighPrice'], quote_df_all.loc[i, 'ComTMC'])
            quote_df_all.loc[i,'OptimalPriceIntervalLow'] = min(quote_df_all.loc[i, 'L1'],quote_df_all.loc[i, 'L2'])
            quote_df_all.loc[i,'OptimalPriceIntervalHigh'] = max(quote_df_all.loc[i, 'H1'],quote_df_all.loc[i, 'H2'])
            
        del quote_df_all['L1']
        del quote_df_all['L2']
        del quote_df_all['H1']
        del quote_df_all['H2']
        
        
        
        #print (quote_df_all['Year'])
        #print ("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
        return quote_df_all
        
    
    
    
    