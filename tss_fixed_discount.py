# -*- coding: utf-8 -*-
import pandas as pd
from copy import deepcopy
import numpy as np

class TssFixedDiscounts():
    """
    class to  apply fixed discounts to SWMA tss quotes, 
    
    apply: applies discount to tss quotes.
    Per TSS business requirements, our web services/online model need to apply the fixed discount rule for TSS components identified as SWMA extension components. The rule is as follows:
        1. for COPRA HWMA Optimal at PTI between 0 and 5%, SWMA extensions can be priced at 65% discount;
        2. for COPRA HWMA Optimal at PTI between 5 and 10%, SWMA extensions can be priced at 60% discount;
        3. for COPRA HWMA Optimal at PTI between 10 and 15%, SWMA extensions can be priced at 55% discount.
 
        In order to apply this rule, web services/online model need to conduct three steps:
        a- Idenfity SWMA extension components:
         
            If "servoffcode" != {"HW-MAINTENANCE","WSU","TMS","HDR"},
            then the TSS component is identified as SWMA extension component.
         
        b- Calculate HWMA PTI % associated with the specific SWMA extension components: 
         
            Identify the specific HWMA component that has the same component id as the SWMA extension components,
            HWMA PTI % = COPRA_NET_GP % - 30%
            (COPRA_NET_GP% = 1- Cost/COPRA_NET_PRICE)
         
        c- Given the calculated HWMA PTI%, apply fixed discount rule for SWMA extension components
   
    """
    def __init__(self):
        pass
    
    def calc_discount(self,input_df):
        swma_discount =  input_df['totalcharge']/(1-input_df['CMDAprice']) * (1-0.60)

        return swma_discount

    def calc_hwma_pti(self,input_df):
        copra_net_gp_pct = 1 - (input_df['Cost']/input_df['TSS_DealBotLineSpreadOptimalPrice'])
        #copra_net_gp = input_df['copra_net_gp_pct'] * input_df['OptimalPrice']
        hwma_pti = copra_net_gp_pct - (0.3)
        return hwma_pti
    
    def apply(self, input_df):

        # Changed "MSS" to "HDR" to accommodate representation in ePricer fields (abh)
        mask = input_df['servoffcode'].isin(["HW-MAINTENANCE","WSU","TMS","HDR",""])

        #input_df['TssComponentType'] = np.where(mask, 'HWMA', 'SWMA')

        # Anything that is not HWMA, WSU, TMS, or HDR falls into SWMA category
        input_df.loc[~mask, 'TssComponentType'] = 'SWMA'
        input_df.loc[mask, 'TssComponentType'] = input_df.loc[mask, 'servoffcode']

        # Identify HW-MAINTENANCE as HWMA for ease of use
        hwma_check = (input_df['servoffcode'] == 'HW-MAINTENANCE')
        input_df.loc[hwma_check, 'TssComponentType'] = 'HWMA'

        #replacing non TSS rows to blank
        input_df['TssComponentType'] = input_df.apply(
                            lambda x: '' if x['TSScomid']==0 else x['TssComponentType'] ,axis=1)
    
        input_df['hwma_pti'] = input_df.apply(
                                    lambda x: self.calc_hwma_pti(x) if x['TssComponentType']=='HWMA' else 0 ,axis=1)

        #replacing non TSS rows to blank
        input_df['hwma_pti'] = input_df.apply(
                            lambda x: '' if x['TSScomid']==0 else x['hwma_pti'] ,axis=1)
        
        input_df['pti'] = input_df.groupby(['Componentid']).hwma_pti.transform('sum')
        
        # Updating the Bottom Line price and the optimal price with the SWMA discount
        # Also removed 80% of OptimalPrice calculation (was only necessary in interim model)
        input_df['TSS_DealBotLineSpreadOptimalPrice'] = input_df.apply(
                    lambda x: self.calc_discount(x) if x['TssComponentType']=='SWMA'
                    else x['TSS_DealBotLineSpreadOptimalPrice'],axis=1)

        input_df['TSS_OptimalPrice'] = input_df.apply(
                    lambda x: self.calc_discount(x) if x['TssComponentType']=='SWMA' else x['TSS_OptimalPrice'],axis=1)
        
        input_df.loc[input_df.TssComponentType=='SWMA','TSS_OptimalPricePofL'] = \
                input_df.TSS_OptimalPrice/(input_df.totalcharge/(1-input_df.CMDAprice))
        
        #replacing non TSS rows to blank
        input_df['TSS_DealBotLineSpreadOptimalPrice'] = input_df.apply(
                    lambda x: '' if x['TSScomid']==0 else x['TSS_DealBotLineSpreadOptimalPrice'] ,axis=1)

        input_df['TSS_OptimalPrice'] = input_df.apply(
                    lambda x: '' if x['TSScomid'] == 0 else x['TSS_OptimalPrice'], axis=1)

        # The AdjComLowPrice, Med, and High prices are set to same value as the discounted_price as the discounts are fixed.
        input_df.loc[~mask, 'TSS_AdjComLowPrice'] = input_df.loc[~mask, 'TSS_OptimalPrice']
        input_df.loc[~mask, 'TSS_AdjComMedPrice'] = input_df.loc[~mask, 'TSS_OptimalPrice'] + 1
                # make sure AdjComLowPrice is less than AdjComMedPrice
        input_df.loc[~mask, 'TSS_AdjComHighPrice'] = input_df.loc[~mask, 'TSS_OptimalPrice'] + 2
                # make sure AdjComHighPrice is greater than AdjComMedPrice

        # optimal_price is same as the discounted_price
        input_df.loc[~mask, 'TSS_OptimalPrice'] = input_df.loc[~mask, 'TSS_AdjComMedPrice']  # make sure OptimalPrice equals to AdjComMedPrice
        input_df.loc[~mask, 'TSS_AdjComLowPofL'] = input_df.loc[~mask, 'TSS_AdjComLowPrice'] / (input_df.loc[~mask, 'totalcharge']/(1-input_df['CMDAprice']))
        input_df.loc[~mask, 'TSS_AdjComMedPofL'] = input_df.loc[~mask, 'TSS_AdjComMedPrice'] / (input_df.loc[~mask, 'totalcharge']/(1-input_df['CMDAprice']))
        input_df.loc[~mask, 'TSS_AdjComHighPofL'] = input_df.loc[~mask, 'TSS_AdjComHighPrice'] / (input_df.loc[~mask, 'totalcharge']/(1-input_df['CMDAprice']))
        input_df.loc[~mask, 'TSS_OptimalPricePofL'] = input_df.loc[~mask, 'TSS_OptimalPrice'] / (input_df.loc[~mask, 'totalcharge']/(1-input_df['CMDAprice']))

        # setting the win probability as 0.5 (50-50) since this is any way a fixed price and not one computed by engine
        input_df.loc[~mask, 'TSS_OptimalPriceWinProb'] = 0.5
        input_df.loc[~mask, 'TSS_OptimalPriceGP'] = input_df.loc[~mask, 'TSS_OptimalPrice'] - input_df.loc[~mask, 'Cost']
        input_df.loc[~mask, 'TSS_OptimalPriceExpectedGP'] = input_df.loc[~mask, 'TSS_OptimalPriceGP'] * 0.5

        # setting intervalLow and intervalHigh also as same as discounted_price because this is a fixed discount anyways
        input_df.loc[~mask, 'TSS_OptimalPriceIntervalLow'] = input_df.loc[~mask, 'TSS_OptimalPrice']
        input_df.loc[~mask, 'TSS_OptimalPriceIntervalHigh'] = input_df.loc[~mask, 'TSS_OptimalPrice']
        input_df.loc[~mask, 'TSS_DealBotLineSpreadOptimalPrice'] = input_df.loc[~mask, 'TSS_OptimalPrice']

        return deepcopy(input_df)

