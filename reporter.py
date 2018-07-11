"""Created on Sun Dec 21 22:25:40 2017

@author: vinita
"""

import pandas as pd
from copy import deepcopy
import BasePricingFunctions as BPF
import numpy as np
from config import *


class Reporter:
    """Integrate pandas.dfs from all sources and prepare optimal price report.

    :param df: output pandas.df from FixedProdHandler/BundleHandler/NormalHandler
    :retur a new pandas.df
    """

    def __init__(self):
        pass

    def dataclean(self, quote_df_out, total_deal_stats, PricePoint_data):
        # Clean the data
        quote_df_out_columns = []
        quote_df_out_columns = quote_df_out.columns
        quote_df_out_columns = list(quote_df_out_columns)

        for i in np.arange(len(quote_df_out_columns)):
            if quote_df_out_columns[i] == 'Indirect(1/0)':
                quote_df_out_columns[i] = 'Indirect_1_0'
            if quote_df_out_columns[i] == 'ClientSeg=E':
                quote_df_out_columns[i] = 'ClientSeg_E'
            if quote_df_out_columns[i] == 'QuotePrice':
                quote_df_out_columns[i] = 'QuotedPrice'

        quote_df_out.columns = quote_df_out_columns
        if (quote_df_out["ComBrand"].any() != 0):
            quote_df_out['ComBrand'] = quote_df_out['ComBrand'].str.replace(
                '&', '&amp;')
        #quote_df_out['Componentid'] = quote_df_out.index + 1

        low_price = total_deal_stats['DealAdjLowPrice']
        median_price = total_deal_stats['DealAdjMedPrice']
        high_price = total_deal_stats['DealAdjHighPrice']
        PricePoint_data.loc[PricePoint_data.PricePointName == 'Optimal Price',
                            'Price'] = total_deal_stats[
                                'DealBotLineOptimalPrice']
        for i in np.arange(len(PricePoint_data)):
            input_price = float(PricePoint_data.loc[i, 'Price'])
            if low_price == 0:
                win_probability = 0
            else:
                win_probability = BPF.ProbOfWin(input_price, low_price,
                                                median_price, high_price)
            PricePoint_data.loc[i, 'WinProbPercent'] = win_probability
        return quote_df_out, total_deal_stats, PricePoint_data

    def calculate_total_deal_stats(self, quote_df, COP_h=0, COP_m=0, COP_l=0):
        #this section calculates the total deal values
        total_deal_stats = pd.Series('', index=['  General Total Quote Data'])
        total_deal_stats['DealListPrice'] = quote_df['ComListPrice'].sum()
        total_deal_stats['DealSize'] = quote_df['ComMedPrice'].sum()
        total_deal_stats['DealTMC'] = quote_df['ComTMC'].sum()
        #print '*PredictedQuotePrice: ',quote_df['PredictedQuotePrice']

        total_deal_stats['DealPredictedQuotePrice'] = quote_df[
            'PredictedQuotePrice'].sum()
        #  this section contains Price Range Data (Line Item Sum)
        total_deal_stats['  Price Range Data (Line Item Sum)'] = ''
        total_deal_stats['DealAdjLowPrice'] = quote_df['AdjComLowPrice'].sum()
        total_deal_stats['DealAdjMedPrice'] = quote_df['AdjComMedPrice'].sum()
        total_deal_stats['DealAdjHighPrice'] = quote_df[
            'AdjComHighPrice'].sum()
        #  this section contains Quoted Price Data (Line Item Sum)
        total_deal_stats['  Quoted Price Data (Line Item Sum)'] = ''
        total_deal_stats['DealQuotePrice'] = quote_df['ComQuotePrice'].sum()
        total_deal_stats['DealQuotePriceWinProb'] = ''
        total_deal_stats[
            'DealQuotePriceGP'] = total_deal_stats['DealQuotePrice'] - total_deal_stats['DealTMC']
        total_deal_stats['DealQuotePriceExpectedGP'] = quote_df[
            'QuotePriceExpectedGP'].sum()
        if total_deal_stats['DealQuotePriceGP'] == 0 :
            total_deal_stats['DealQuotePriceWinProb'] = 0
        else :
            total_deal_stats['DealQuotePriceWinProb'] = total_deal_stats[
                    'DealQuotePriceExpectedGP'] / total_deal_stats['DealQuotePriceGP']
        #  this section contains optimal price data
        total_deal_stats['  Optimal Price Data (Line Item Sum)'] = ''
        total_deal_stats['DealOptimalPrice'] = quote_df['OptimalPrice'].sum()
        total_deal_stats['DealOptimalPriceWinProb'] = ''
        total_deal_stats['DealOptimalPriceGP'] = quote_df[
            'OptimalPriceGP'].sum()
        total_deal_stats['DealOptimalPriceExpectedGP'] = quote_df[
            'OptimalPriceExpectedGP'].sum()
        total_deal_stats['DealOptimalPriceWinProb'] = 0  #
        if total_deal_stats['DealOptimalPriceGP'] != 0:  #
            total_deal_stats['DealOptimalPriceWinProb'] = total_deal_stats[
                'DealOptimalPriceExpectedGP'] / total_deal_stats[
                    'DealOptimalPriceGP']  #
        else :
            total_deal_stats['DealOptimalPriceWinProb'] = 0
        total_deal_stats['DealOptimalPriceIntervalLow'] = quote_df[
            'OptimalPriceIntervalLow'].sum()
        total_deal_stats['DealOptimalPriceIntervalHigh'] = quote_df[
            'OptimalPriceIntervalHigh'].sum()
        #  this section contains Quoted Price Data (Bottom-Line)
        total_deal_stats['  Quoted Price Data (Bottom-Line)'] = ''
        total_deal_stats['DealBotLineQuotePrice'] = total_deal_stats[
            'DealQuotePrice']
        if total_deal_stats['DealAdjLowPrice'] == 0:
            total_deal_stats['DealBotLineQuotePriceWinProb'] = 0
        else:
            total_deal_stats['DealBotLineQuotePriceWinProb'] = BPF.ProbOfWin(
                total_deal_stats['DealBotLineQuotePrice'],
                total_deal_stats['DealAdjLowPrice'],
                total_deal_stats['DealAdjMedPrice'],
                total_deal_stats['DealAdjHighPrice'])
        total_deal_stats[
            'DealBotLineQuotePriceGP'] = total_deal_stats['DealBotLineQuotePrice'] - total_deal_stats['DealTMC']
        total_deal_stats['DealBotLineQuotePriceExpectedGP'] = total_deal_stats[
            'DealBotLineQuotePriceGP'] * total_deal_stats[
                'DealBotLineQuotePriceWinProb']
        #  this section contains Optimal Price Data (Bottom-Line)
        total_deal_stats['  Optimal Price Data (Bottom-Line)'] = ''
        if total_deal_stats['DealAdjLowPrice'] == 0:
            total_deal_stats['DealBotLineOptimalPrice'] = 0
        else:
            total_deal_stats['DealBotLineOptimalPrice'] = BPF.OptPrice(
                total_deal_stats['DealAdjLowPrice'],
                total_deal_stats['DealAdjMedPrice'],
                total_deal_stats['DealAdjHighPrice'], total_deal_stats['DealTMC'],
                0, 0, 0)
        if total_deal_stats['DealAdjLowPrice'] == 0:
            total_deal_stats['DealBotLineOptimalPriceWinProb'] = 0
        else:
            total_deal_stats['DealBotLineOptimalPriceWinProb'] = BPF.ProbOfWin(
                total_deal_stats['DealBotLineOptimalPrice'],
                total_deal_stats['DealAdjLowPrice'],
                total_deal_stats['DealAdjMedPrice'],
                total_deal_stats['DealAdjHighPrice'])
        total_deal_stats[
            'DealBotLineOptimalPriceGP'] = total_deal_stats['DealBotLineOptimalPrice'] - total_deal_stats['DealTMC']
        total_deal_stats[
            'DealBotLineOptimalPriceExpectedGP'] = total_deal_stats[
                'DealBotLineOptimalPriceGP'] * total_deal_stats[
                    'DealBotLineOptimalPriceWinProb']
        if total_deal_stats['DealAdjLowPrice'] == 0:
            total_deal_stats['DealBotLineOptimalPriceIntervalLow'], total_deal_stats['DealBotLineOptimalPriceIntervalHigh'] = 0,0
        else:
            total_deal_stats['DealBotLineOptimalPriceIntervalLow'], total_deal_stats[
                'DealBotLineOptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(
                    total_deal_stats['DealBotLineOptimalPrice'],
                    total_deal_stats['DealAdjLowPrice'],
                    total_deal_stats['DealAdjMedPrice'],
                    total_deal_stats['DealAdjHighPrice'],
                    total_deal_stats['DealTMC'])

        #this section is executed only if customized optimal pricing (COP) is needed
        if (COP_h > COP_m) and (COP_m > COP_l):
            #  this section contains COP Price Range Data (Line Item Sum)
            total_deal_stats['  COP Price Range Data (Line Item Sum)'] = ''
            total_deal_stats['DealCOPLowPrice'] = quote_df[
                'COPComLowPrice'].sum()
            total_deal_stats['DealCOPMedPrice'] = quote_df[
                'COPComMedPrice'].sum()
            total_deal_stats['DealCOPHighPrice'] = quote_df[
                'COPComHighPrice'].sum()
            #  this section contains COP Quote Price Data (Line Item Sum)
            total_deal_stats['  COP Quote Price Data (Line Item Sum)'] = ''
            total_deal_stats['DealCOPQuotePrice'] = quote_df[
                'ComQuotePrice'].sum()
            total_deal_stats['DealCOPQuotePriceWinProb'] = ''
            total_deal_stats['DealCOPQuotePriceGP'] = quote_df[
                'COPQuotePriceGP'].sum()
            total_deal_stats['DealCOPQuotePriceExpectedGP'] = quote_df[
                'COPQuotePriceExpectedGP'].sum()
            if total_deal_stats['DealCOPQuotePriceGP'] == 0:
                total_deal_stats['DealCOPQuotePriceWinProb'] = 0
            else:
                total_deal_stats['DealCOPQuotePriceWinProb'] = total_deal_stats[
                        'DealCOPQuotePriceExpectedGP'] / total_deal_stats[
                                'DealCOPQuotePriceGP']
            #  this section contains COP Optimal Price Data (Line Item Sum)
            total_deal_stats['  COP Optimal Price Data (Line Item Sum)'] = ''
            total_deal_stats['DealCOPOptimalPrice'] = quote_df[
                'COPOptimalPrice'].sum()
            total_deal_stats['DealCOPOptimalPriceWinProb'] = ''
            total_deal_stats['DealCOPOptimalPriceGP'] = quote_df[
                'COPOptimalPriceGP'].sum()
            total_deal_stats['DealCOPOptimalPriceExpectedGP'] = quote_df[
                'COPOptimalPriceExpectedGP'].sum()
            if total_deal_stats['DealCOPOptimalPriceGP'] == 0:
                total_deal_stats['DealCOPOptimalPriceWinProb'] = 0
            else:
                total_deal_stats['DealCOPOptimalPriceWinProb'] = total_deal_stats[
                    'DealCOPOptimalPriceExpectedGP'] / total_deal_stats[
                        'DealCOPOptimalPriceGP']
            total_deal_stats['DealCOPOptimalPriceIntervalLow'] = quote_df[
                'COPOptimalPriceIntervalLow'].sum()
            total_deal_stats['DealCOPOptimalPriceIntervalHigh'] = quote_df[
                'COPOptimalPriceIntervalHigh'].sum()
            #  this section contains quoted price data within the Customized Optimal Price (COP) estimates
            total_deal_stats['  COP Quote Price Data (Bottom-Line)'] = ''
            total_deal_stats['DealCOPBotLineQuotePrice'] = total_deal_stats[
                'DealQuotePrice']
            if total_deal_stats['DealCOPLowPrice'] == 0:
                total_deal_stats['DealCOPBotLineQuotePriceWinProb'] = 0
            else:
                total_deal_stats[
                    'DealCOPBotLineQuotePriceWinProb'] = BPF.ProbOfWin(
                        total_deal_stats['DealCOPQuotePrice'],
                        total_deal_stats['DealCOPLowPrice'],
                        total_deal_stats['DealCOPMedPrice'],
                        total_deal_stats['DealCOPHighPrice'])
            total_deal_stats[
                'DealCOPBotLineQuotePriceGP'] = total_deal_stats['DealCOPBotLineQuotePrice'] - total_deal_stats['DealTMC']
            total_deal_stats[
                'DealCOPBotLineQuotePriceExpectedGP'] = total_deal_stats[
                    'DealCOPBotLineQuotePriceGP'] * total_deal_stats[
                        'DealCOPBotLineQuotePriceWinProb']
            #  this section contains COP Optimal Price Data (Bottom-Line)
            total_deal_stats['  COP Optimal Price Data (Bottom-Line)'] = ''
            if total_deal_stats['DealCOPLowPrice'] == 0:
                total_deal_stats['DealCOPBotLineOptimalPrice'] = 0
            else:
                total_deal_stats['DealCOPBotLineOptimalPrice'] = BPF.OptPrice(
                    total_deal_stats['DealCOPLowPrice'],
                    total_deal_stats['DealCOPMedPrice'],
                    total_deal_stats['DealCOPHighPrice'],
                    total_deal_stats['DealTMC'], 0, 0, 0)
            if total_deal_stats['DealCOPLowPrice'] == 0:
                total_deal_stats['DealCOPBotLineOptimalPriceWinProb'] = 0
            else:
                total_deal_stats[
                    'DealCOPBotLineOptimalPriceWinProb'] = BPF.ProbOfWin(
                        total_deal_stats['DealCOPBotLineOptimalPrice'],
                        total_deal_stats['DealCOPLowPrice'],
                        total_deal_stats['DealCOPMedPrice'],
                        total_deal_stats['DealCOPHighPrice'])
            total_deal_stats[
                'DealCOPBotLineOptimalPriceGP'] = total_deal_stats['DealCOPBotLineOptimalPrice'] - total_deal_stats['DealTMC']
            total_deal_stats[
                'DealCOPBotLineOptimalPriceExpectedGP'] = total_deal_stats[
                    'DealCOPBotLineOptimalPriceGP'] * total_deal_stats[
                        'DealCOPBotLineOptimalPriceWinProb']
            if total_deal_stats['DealCOPLowPrice'] == 0:
                total_deal_stats['DealCOPBotLineOptimalPriceIntervalLow'], total_deal_stats['DealCOPBotLineOptimalPriceIntervalHigh'] = 0,0
            else:
                total_deal_stats[
                    'DealCOPBotLineOptimalPriceIntervalLow'], total_deal_stats[
                        'DealCOPBotLineOptimalPriceIntervalHigh'] = BPF.OptPriceConfIntervl(
                            total_deal_stats['DealCOPBotLineOptimalPrice'],
                            total_deal_stats['DealCOPLowPrice'],
                            total_deal_stats['DealCOPMedPrice'],
                            total_deal_stats['DealCOPHighPrice'],
                            total_deal_stats['DealTMC'])

        return total_deal_stats
    #quote_df_out, total_deal_stats, PricePoint_data = Reporter(
        #    ).report(out_PricePoint_data, df_zero_price, ComRevDivCd_orgi,
         #            ComRevDivCd_orgi2, quote_df_all, final_output_fixed)

    def report(self, price_df, zero_price_df, ComRevDivCd_orgi,ComRevDivCd_orgi2, *quote_dfs):

        PricePoint_data1 = price_df
        ComRevDivCd_orgi = ComRevDivCd_orgi
        zero_price_df = zero_price_df
        ComRevDivCd_orgi2 = ComRevDivCd_orgi2
        
        quote_df_out_0 = pd.concat(quote_dfs).reset_index(drop=True)
        
        """For Fixed products, hotfix code added to convert Float varaibles to interger datatype
        ('Year','Month','EndOfQtr','Indirect(1/0)','ClientSeg=E','UpgMES','Quantity','WinLoss','Componentid')"""
        
        int64list = INT64LIST
        
        for i in np.arange(len(int64list)):
                quote_df_out_0[int64list[i]] = np.array(
                    quote_df_out_0[int64list[i]], dtype='int64')
             
        if (quote_df_out_0.empty) :
            quote_df_out_0 = zero_price_df
            quote_df_out_0['ComMedPrice'] = 0
            quote_df_out_0['DealSize'] = 0

            quote_df_out_0['LogDealSize'] = 0
            quote_df_out_0['ComPctContrib'] = 0
        else:
            if zero_price_df.empty:
                pass
            else:
                quote_df_out_0 = pd.concat([quote_df_out_0,
                                       zero_price_df]).reset_index(drop=True)
            
            quote_df_out_0['ComMedPrice'] = (1.0 *(quote_df_out_0['ComMedPrice'])).astype(
                    float)
            quote_df_out_0['DealSize'] = quote_df_out_0['ComMedPrice'].sum()

            quote_df_out_0['LogDealSize'] = np.log10(
                    quote_df_out_0['DealSize'].astype(float))
            quote_df_out_0['ComPctContrib'] = quote_df_out_0[
                    'ComMedPrice'] / quote_df_out_0['DealSize']
            
        total_deal_stats_0 = self.calculate_total_deal_stats(
            quote_df_out_0, 0, 0, 0)
        
        quote_df, total_deal_stats1, PricePoint_data1 = self.dataclean(
            quote_df_out_0, total_deal_stats_0, price_df)

        quote_df_out1 = pd.DataFrame(quote_df, columns = REPORTER_COLUMNS )        
        
        quote_df_out1 = quote_df_out1.reset_index(drop=True)
        quote_df_out1.sort_values('Componentid', inplace=True)
        quote_df_out1['ComRevDivCd'] = pd.DataFrame(
            ComRevDivCd_orgi.split('-'))
        del quote_df_out1['ComRevDivCd']
        quote_df_out1 = pd.merge(quote_df_out1,ComRevDivCd_orgi2, on='Componentid')

        quote_df_out = deepcopy(quote_df_out1)
        total_deal_stats = deepcopy(total_deal_stats1)
        PricePoint_data = deepcopy(PricePoint_data1)

        return quote_df_out, total_deal_stats, PricePoint_data
