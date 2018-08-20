import pandas as pd

from JP.online.BasePricingFunctions import ProbOfWin


# Per CIO reqs, this section builds the quote_df object found in PricingEngine.py (from old Model Factory code)
#  lines 188-242 (quote_df) and lines 329-433 (total_deal_stats)
def transform_output(prep_comp):
    # prep quote-level output

    # NOTE: original Model Factory was all in terms of list price. New approach bases everything off value_score
    #       these fields are all left as "PofL", i.e. Percent of List, for convenience (not having to bother CIO),
    #       but they are actually Price of Value Score; PofV

    # maps quote_df col -> comp col
    quote_map = {
        'AdjComHighPofL': 'adj_pred_H',
        'AdjComHighPrice': 'adj_price_H',
        'OptimalPricePofL': 'OP_VS',
        'AdjComLowPofL': 'adj_pred_L',
        'AdjComLowPrice': 'adj_price_L',
        'AdjComMedPofL': 'adj_pred_M',
        'AdjComMedPrice': 'adj_price_M',
        'bot_spread_OPAdj': 'bot_spread_OPAdj',
        'CCMScustomerNumber': 'ccmscustomernumber',
        'ChannelID': 'channelid',
        'ci_high': 'ci_high',
        'ci_low': 'ci_low',
        'ClientSeg=E': 'client_e',
        'ClientSegCd': 'clientsegcd',
        'ComTMC': 'tmc',
        'ComBrand': 'combrand',
        'ComCategory': 'comcategory',
        'ComCostPofL': 'comcostpofl',
        'ComDelgPriceL4': 'comdelgpricel4',
        'ComDelgPriceL4PofL': 'comdelgpricel4pofl',
        'ComFamily': 'comfamily',
        'ComGroup': 'comgroup',
        'ComHighPofL': 'comhighpofl',
        'ComListPrice': 'list_price',
        'ComLogListPrice': 'comloglistprice',
        'ComLowPofL': 'comlowpofl',
        'ComMedPofL': 'commedpofl',
        'ComMedPrice': 'commedprice',
        'ComMT': 'commt',
        'ComMTM': 'commtm',
        'ComMTMDesc': 'commtmdesc',
        'ComMTMDescLocal': 'commtmdesclocal',
        'ComPctContrib': 'compctcontrib',
        'Componentid': 'componentid',
        'ComQuotePricePofL': 'comquotepricepofl',
        'ComRevCat': 'comrevcat',
        'ComRevDivCd': 'comrevdivcd',
        'ComSubCategory': 'comsubcategory',
        'ComTMCPofL': 'ComTMCPofL',
        'COPComHighPofL': 'COPComHighPofL',
        'COPComHighPrice': 'COPComHighPrice',
        'COPComLowPofL': 'COPComLowPofL',
        'COPComLowPrice': 'COPComLowPrice',
        'COPComMedPofL': 'COPComMedPofL',
        'COPComMedPrice': 'COPComMedPrice',
        'COPOptimalPrice': 'COPOptimalPrice',
        'COPOptimalPriceExpectedGP': 'COPOptimalPriceExpectedGP',
        'COPOptimalPriceGP': 'COPOptimalPriceGP',
        'COPOptimalPriceIntervalHigh': 'COPOptimalPriceIntervalHigh',
        'COPOptimalPriceIntervalLow': 'COPOptimalPriceIntervalLow',
        'COPOptimalPricePofL': 'COPOptimalPricePofL',
        'COPOptimalPriceWinProb': 'COPOptimalPriceWinProb',
        'COPQuotePriceExpectedGP': 'COPQuotePriceExpectedGP',
        'COPQuotePriceGP': 'COPQuotePriceGP',
        'COPQuotePriceWinProb': 'COPQuotePriceWinProb',
        'Countrycode': 'countrycode',
        'CustomerIndustryName': 'customerindustryname',
        'CustomerNumber': 'customernumber',
        'CustomerSecName': 'customersecname',
        'DealBotLineSpreadOptimalPrice': 'bot_spread_OP',
        'DealSize': 'dealsize',
        'discount': 'discount',
        'DomBuyerGrpID': 'dombuyergrpid',
        'DomBuyerGrpName': 'dombuyergrpname',
        'EndOfQtr': 'endofqtr',
        'FeatureQuantity': 'featurequantity',
        'HWPlatformid': 'hwplatformid',
        'Indirect(1/0)': 'indirect',
        'LogDealSize': 'logdealsize',
        'LogValueScore': 'LogValueScore',
        'Level_0': 'lvl0',
        'Level_1': 'lvl1',
        'Level_2': 'lvl2',
        'Level_3': 'lvl3',
        'Level_4': 'lvl4',
        'ModelID': 'modelid',
        'Month': 'month',
        'norm_op': 'norm_op',
        'OptimalPrice': 'price_opt',
        'OptimalPriceExpectedGP': 'opt_EGP',
        'OptimalPriceGP': 'opt_GP',
        'OptimalPriceIntervalHigh': 'opt_ci_high',
        'OptimalPriceIntervalLow': 'opt_ci_low',
        'OptimalPriceWinProb': 'wp_opt',
        'Quantity': 'quantity',
        'QuotePrice': 'quoted_price',
        'QuoteID': 'quoteid',
        'quoteidnew': 'quoteidnew',
        'QuotePriceExpectedGP': 'QuotePriceExpectedGP',
        'QuotePriceGP': 'QuotePriceGP',
        'QuoteType': 'quotetype',
        'RequestingApplicationID': 'requestingapplicationid',
        'sbc_incl': 'sbc_incl',
        'ComSpclBidCode1': 'sbc1',
        'ComSpclBidCode2': 'sbc2',
        'ComSpclBidCode3': 'sbc3',
        'ComSpclBidCode4': 'sbc4',
        'ComSpclBidCode5': 'sbc5',
        'ComSpclBidCode6': 'sbc6',
        'ufc': 'ufc',
        'ufc_incl': 'ufc_incl',
        'UpgMES': 'upgmes',
        'ValueScore': 'ValueScore',
        'Version': 'version',
        'WinLoss': 'winloss',
        'Year': 'year'
    }

    quote_missing = ['QuotePricePofL', 'QuotePrice', 'QuotePriceWinProb', 'QuotePriceGP', 'QuotePriceExpectedGP',
                     'COPComLowPrice', 'COPComMedPrice', 'COPComHighPrice', 'COPComLowPofL', 'COPComMedPofL',
                     'COPComHighPofL', 'COPOptimalPrice', 'COPOptimalPricePofL', 'COPOptimalPriceWinProb',
                     'COPOptimalPriceGP', 'COPOptimalPriceExpectedGP', 'COPOptimalPriceIntervalLow',
                     'COPOptimalPriceIntervalHigh', 'COPQuotePriceWinProb', 'COPQuotePriceGP',
                     'COPQuotePriceExpectedGP']

    quote_df = prep_comp.copy().rename(columns={v: k for k, v in quote_map.items()})

    for col in quote_missing:
        quote_df[col] = None

    for i in range(len(quote_df)):
        quote_df.loc[i, 'PredictedQuotePricePofL'] = quote_df.loc[i, 'OptimalPricePofL']
        quote_df.loc[i, 'PredictedQuotePrice'] = quote_df.loc[i, 'OptimalPrice']
        quote_df.loc[i, 'ComQuotePrice'] = quote_df.loc[i, 'ComQuotePricePofL'] * quote_df.loc[i, 'ComListPrice']
        quote_df.loc[i, 'QuotePricePofL'] = quote_df.loc[i, 'ComQuotePricePofL']
        L = quote_df.loc[i, 'AdjComLowPofL']
        M = quote_df.loc[i, 'AdjComMedPofL']
        H = quote_df.loc[i, 'AdjComHighPofL']
        quote_df.loc[i, 'QuotePrice'] = quote_df.loc[i, 'ComQuotePrice']
        quote_df.loc[i, 'QuotePriceWinProb'] = ProbOfWin(quote_df.loc[i, 'QuotePricePofL'], L, M, H)
        quote_df.loc[i, 'QuotePriceGP'] = quote_df.loc[i, 'QuotePrice'] - quote_df.loc[i, 'ComTMC']
        quote_df.loc[i, 'QuotePriceExpectedGP'] = quote_df.loc[i, 'QuotePriceGP'] * quote_df.loc[i, 'QuotePriceWinProb']

        quote_df.loc[i, 'ComLowPrice'] = quote_df.loc[i, 'AdjComLowPrice']
        quote_df.loc[i, 'ComMedPrice'] = quote_df.loc[i, 'AdjComMedPrice']
        quote_df.loc[i, 'ComHighPrice'] = quote_df.loc[i, 'AdjComHighPrice']

        quote_df.loc[i, 'ComLowPofL'] = quote_df.loc[i, 'AdjComLowPofL']
        quote_df.loc[i, 'ComMedPofL'] = quote_df.loc[i, 'AdjComMedPofL']
        quote_df.loc[i, 'ComHighPofL'] = quote_df.loc[i, 'AdjComHighPofL']

    return quote_df


def spread_optimal_price_ravi(quote_df, total_deal_stats):
    """quote_df = quote dataframe, total_deal_stats = total deal statistics dataframe"""
    '''
        The purpose of this function is to populate the total deal bottom line
        optimal price to the individual line items.  The ensures that in addition
        to each line item's optimal price, there will also be a field where
        the bottom line optimal price is spread to the line items.  

        The method of this spread is based on linear interpolation between key
        price points.  The price points are:

        Zero Price <= Low Price <= Median Price <= High Price <= List Price <= 10xList Price   # vasu 7/26/17  Defect # 1534724

        These component price points are totaled and then the DealBotLineOptimalPrice
        is compared to this list of total price points.  The logic finds the two 
        adjacent price points that surround the bottom line price.  Linear
        interpolation finds the relative position between the two price point and
        then each line item spread price is calculated by finding the same
        relitive position between its corresponding line item price points.

        Created:  10 Apr 2017 by SaiNeveen Sare
        Updated:  01 Jun 2017 by Glenn Melzer
        Updated:  26 July 2017 by vasu for Defect # 1534724

        INPUTS:
          quote_df = the dataframe of quote components (with the same column 
              format as the historical data input dataframe).
          total_deal_stats = the statistics of the total deal

        OUTPUTS:
          quote_df = the original quote_df with the DealBotLineSpreadOptimalPrice
              column populated
    '''

    # This section extracts the total deal bottom line optimal price
    # total_deal_stats = total_deal_stats.to_dict()
    total_deal_stats = total_deal_stats
    print(total_deal_stats)
    optimal_price = total_deal_stats['bot_OP'][0]
    print(optimal_price)

    # This section creates a spread_price dataframe of component price price points & includes a total row
    # spread_price = quote_df.loc[:, ('AdjComLowPrice', 'AdjComMedPrice', 'AdjComHighPrice', 'ComListPrice')]  # vasu 7/26/17  Defect # 1534724
    spread_price = quote_df.loc[:, ['adj_price_L', 'adj_price_M', 'adj_price_H', 'list_price']]

    spread_price.insert(0, 'ZeroPrice', 0)
    spread_price['10XPrice'] = spread_price['list_price'] * 10
    spread_price.loc['Total'] = spread_price.sum().values

    # This section creates an adj_spread_price dataframe that removes "mountains"
    # (i.e for price points to left of optimal price: take lowest price of current column up to the optimal price column
    #      for price points to right of optimal price: take highest price of current column down to the optimal price column)
    adj_spread_price = pd.DataFrame()
    spread_columns = spread_price.columns
    adj_spread_price['ZeroPrice'] = spread_price.loc[:, spread_columns].min(axis=1)
    adj_spread_price['adj_price_L'] = spread_price.loc[:, spread_columns[1:3]].min(axis=1)
    adj_spread_price['adj_price_M'] = spread_price['adj_price_M']  # vasu 7/26/17  Defect # 1534724
    adj_spread_price['adj_price_H'] = spread_price.loc[:, spread_columns[2:4]].max(axis=1)
    adj_spread_price['list_price'] = spread_price.loc[:, spread_columns[2:5]].max(axis=1)
    adj_spread_price['10XPrice'] = spread_price['10XPrice']
    adj_spread_price = adj_spread_price[:-1]
    adj_spread_price.loc['Total'] = adj_spread_price.sum().values

    # This section selects the lower column of the two columns used to perform the linear interpolation between
    adj_points = [1 if ((adj_spread_price.loc['Total']['ZeroPrice'] < optimal_price) & (
    adj_spread_price.loc['Total']['adj_price_L'] >= optimal_price)) else 0,
                  1 if ((adj_spread_price.loc['Total']['adj_price_L'] < optimal_price) & (
                  adj_spread_price.loc['Total']['adj_price_M'] >= optimal_price)) else 0,
                  1 if ((adj_spread_price.loc['Total']['adj_price_M'] < optimal_price) & (
                  adj_spread_price.loc['Total']['adj_price_H'] >= optimal_price)) else 0,
                  1 if ((adj_spread_price.loc['Total']['adj_price_H'] < optimal_price) & (
                  adj_spread_price.loc['Total']['list_price'] >= optimal_price)) else 0,
                  1 if ((adj_spread_price.loc['Total']['list_price'] < optimal_price) & (
                  adj_spread_price.loc['Total']['10XPrice'] >= optimal_price)) else 0,
                  1 if (
                  adj_spread_price.loc['Total']['10XPrice'] < optimal_price) else 0]  # vasu 7/26/17  Defect # 1534724
    weight_df = pd.DataFrame(adj_points, index=['ZeroPrice', 'adj_price_L', 'adj_price_M', 'adj_price_H', 'list_price',
                                                '10XPrice'])  # vasu 7/26/17  Defect # 1534724

    # This section spreads the bottom line optimal price to the line items
    spread_mechanism = pd.DataFrame()
    spread_mechanism['lower_price'] = adj_spread_price.iloc[:, 0:5].dot(weight_df.iloc[0:5, :])[0]
    weight_df.index = ['adj_price_L', 'adj_price_M', 'adj_price_H', 'list_price', '10XPrice',
                       'ZeroPrice']  # vasu 7/26/17  Defect # 1534724
    spread_mechanism['higher_price'] = adj_spread_price.iloc[:, 1:6].dot(weight_df.iloc[0:5, :])[0]
    total_lower = spread_mechanism.loc['Total']['lower_price']
    total_higher = spread_mechanism.loc['Total']['higher_price']
    spread_value = ((optimal_price - total_lower) / (total_higher - total_lower)) if (
                                                                                     total_higher - total_lower) != 0 else 0
    spread_mechanism['spread_price'] = spread_mechanism['lower_price'] + (spread_mechanism['higher_price'] -
                                                                          spread_mechanism[
                                                                              'lower_price']) * spread_value

    # This section loads the spread optimal prices to the quote_df dataframe
    quote_df['bot_spread_OP'] = spread_mechanism['spread_price']
    # quote_df.to_csv(data_path + 'spread_optimal_price.csv', index=False)

    print(quote_df['bot_spread_OP'])
    print('World++++++++++++++++++')

    ########SUPPLEMENT########

    df = quote_df[['componentid', 'tmc', 'list_price', 'bot_spread_OP']]

    df = df[df['list_price'] > 0]

    df['num'] = range(len(df))

    df.index = range(len(df))

    df = calculate_initial_adjusted_spread_prices_greater_than_tmc(df)

    print('Initial Adjustment of df...')
    print(df['bot_spread_OPAdj'])
    print(df['bot_spread_OPAdj'].sum())

    for i in range(len(df)):
        df.loc[i, 'Diff'] = df.loc[i, 'list_price'] - df.loc[i, 'tmc'] - df.loc[i, 'Delta'] * df.loc[
            i, 'bot_spread_OPAdj']

    print('Here diffs are...')
    print(df['Diff'])

    if is_there_negative_difference(df) is True:
        print('Hello Suresh!')
        df = increment_lowest_decrement_highest(df)

    print('After Algorithm df is...')
    print(df['bot_spread_OPAdj'])
    print(df['bot_spread_OPAdj'].sum())

    print('Diffs after Algorithm...')
    print(df['Diff'])

    for i in range(len(df)):
        # df.loc[i, 'bot_spread_OPAdj'] = df.loc[i, 'tmc'] + df.loc[i, 'Delta']*df.loc[i, 'bot_spread_OP']
        df.loc[i, 'bot_spread_OPAdj'] = df.loc[i, 'tmc'] + df.loc[i, 'Delta'] * df.loc[i, 'bot_spread_OPAdj']

    print('After bot_spread_OP based df is ...')
    print(df['bot_spread_OPAdj'])
    print(df['bot_spread_OPAdj'].sum())

    # df.to_csv('C:ModelFactoryGithub ClassesVasuoutput7_2_0.csv')

    for iter in range(25):
        df.index = range(len(df))
        df = spread_back_the_error(df)
        df.index = range(len(df))

        # df.to_csv('C:ModelFactoryGithub ClassesVasuoutput7_2_2.csv')

        del df['bot_spread_OPDiff']

    # df.index = df['num']

    df.sort_index(inplace=True)

    # df.to_csv('C:ModelFactoryGithub ClassesVasuoutput3_inter.csv')

    print('After SpreadBack based...')
    print(df['bot_spread_OPAdj'])
    print(df['bot_spread_OPAdj'].sum())
    print('DealBotLineOptimalPrice is...')
    print(optimal_price)

    # df.to_csv('C:ModelFactoryGithub ClassesVasuoutput7_2.csv')

    df = df[['componentid', 'bot_spread_OPAdj']]

    # quote_df_out = pd.read_csv('C:/ModelFactory/Github Classes/Vasu/quote_df_out_6march.csv')

    quote_df_out_final = pd.merge(quote_df, df, how='inner', sort=True)

    return quote_df_out_final


def is_there_negative_difference(df):
    df['P'] = 1.0
    P1 = 1.0
    for i in range(len(df)):
        if (df.loc[i, 'Diff'] < 0):
            df.loc[i, 'P'] = 0
        P1 = P1 * df.loc[i, 'P']

    if P1 == 0:
        return True
    else:
        return False


def is_there_negative_bot_spread_OPAdj(df):
    df['P'] = 1.0
    P1 = 1.0
    for i in range(len(df)):
        if (df.loc[i, 'bot_spread_OPAdj'] < 0):
            df.loc[i, 'P'] = 0
        P1 = P1 * df.loc[i, 'P']

    if P1 == 0:
        return True
    else:
        return False


def increment_lowest_decrement_highest(df):
    highest = df['Diff'].max()
    lowest = df['Diff'].min()

    h = df.index.get_loc(df["Diff"].argmax())
    l = df.index.get_loc(df["Diff"].argmin())

    rates = [0.1, 0.05, 0.005, 0.0005]

    if highest * lowest < 0:
        for rate in rates:
            n = int(1 / rate)
            increment_decrement_algorithm_successful = False

            for i in range(n):
                df.loc[l, 'Delta'] = df.loc[l, 'Delta'] * (1 - rate)
                df.loc[h, 'Delta'] = df.loc[h, 'Delta'] * (
                1 + rate * (df.loc[l, 'bot_spread_OPAdj'] / df.loc[h, 'bot_spread_OPAdj']))

                if (df.loc[l, 'Delta'] < 0):
                    break

                df.loc[h, 'Diff'] = df.loc[h, 'list_price'] - df.loc[h, 'tmc'] - df.loc[h, 'Delta'] * df.loc[
                    h, 'bot_spread_OPAdj']
                df.loc[l, 'Diff'] = df.loc[l, 'list_price'] - df.loc[l, 'tmc'] - df.loc[l, 'Delta'] * df.loc[
                    l, 'bot_spread_OPAdj']

                # df.loc[h, 'bot_spread_OPAdj'] = df.loc[h, 'tmc'] + df.loc[h, 'Delta']*df.loc[h, 'bot_spread_OPAdj']
                # df.loc[l, 'bot_spread_OPAdj'] = df.loc[l, 'tmc'] + df.loc[l, 'Delta']*df.loc[l, 'bot_spread_OPAdj']

                if (df.loc[h, 'Diff'] > 0 and df.loc[l, 'Diff'] > 0):
                    increment_decrement_algorithm_successful = True
                    print('Increment Decrement Step Successful!')
                    break
            if increment_decrement_algorithm_successful is True:
                break
    if is_there_negative_difference(df) is True:
        try:
            # print 'Going into inner loop...'
            df = increment_lowest_decrement_highest(df)
            return df
        except:
            print('Recursion unsuccessful...')
            return df
    else:
        return df


        # This is not used


def calculate_initial_adjusted_spread_prices_greater_than_tmc(df):
    df.index = range(len(df))
    B = df['bot_spread_OP'].sum()
    T = df['tmc'].sum()
    for i in range(len(df)):
        # df.loc[i, 'bot_spread_OPAdj'] = B/len(df)
        df.loc[i, 'Delta'] = (B - T) / (len(df) * df.loc[i, 'bot_spread_OP'])
        df.loc[i, 'bot_spread_OPAdj'] = df.loc[i, 'tmc'] + df.loc[i, 'Delta'] * df.loc[i, 'bot_spread_OP']
    return df


def calculate_initial_adjusted_spread_prices_less_than_list_price(df):
    df.index = range(len(df))
    L = df['list_price'].sum()
    B = df['bot_spread_OP'].sum()
    for i in range(len(df)):
        # df.loc[i, 'bot_spread_OPAdj'] = B/len(df)
        df.loc[i, 'Delta'] = (L - B) / (len(df) * df.loc[i, 'bot_spread_OP'])
        df.loc[i, 'bot_spread_OPAdj'] = df.loc[i, 'list_price'] - df.loc[i, 'Delta'] * df.loc[i, 'bot_spread_OP']
    return df


def spread_back_the_error(df):
    Error = (df['bot_spread_OPAdj'].sum() - df['bot_spread_OP'].sum()) * 0.5
    if Error < 0:
        print('Error is -ve!')
        print('Error is ', Error)
        for i in range(len(df)):
            df.loc[i, 'bot_spread_OPDiff'] = df.loc[i, 'bot_spread_OPAdj'] - df.loc[i, 'bot_spread_OP']

        df1 = df[df['bot_spread_OPDiff'] < 0]
        df2 = df[df['bot_spread_OPDiff'] > 0]

        df1 = df1.sort_values(by=['bot_spread_OPDiff'], ascending=True)

        df1.index = range(len(df1))
        for l in range(len(df1)):
            df1.loc[l, 'Diff'] = (df1.loc[l, 'list_price'] - df1.loc[l, 'bot_spread_OPAdj']) * (
            df1.loc[l, 'bot_spread_OPAdj'] - df1.loc[l, 'tmc'])

        for j in range(len(df1) - 1):
            print(len(df1))
            no_negative_difference = False
            # df1.index = range(len(df1))
            # df11.sort_values(by=['bot_spread_OPDiff'], ascending= False)
            df11 = df1.head(j + 1)
            df1 = df1.sort_values(by=['bot_spread_OPDiff'], ascending=False)
            print('Differences are...')
            print(df1['bot_spread_OPDiff'])
            df12 = df1.head(len(df1) - j - 1)

            df11 = df11.sort_values(by=['bot_spread_OPDiff'], ascending=True)
            # df11.sort_index(inplace=True, ascending=True)
            '''
            if j < len(df1)-1:
                df12 = df1.head(j+1)
            else:
                df12 = pd.DataFrame()
            '''
            print('Diffs are...')
            print(df11['bot_spread_OPDiff'])

            print('j is...', j)
            print('HelloWorld!...', len(df11))
            print('HelloRavi!...', len(df12))

            df11.index = range(len(df11))
            for k in range(len(df11)):
                temp = df11.loc[k, 'bot_spread_OPAdj'] - (1 / float(len(df11))) * (Error)
                df11.loc[k, 'bot_spread_OPAdj'] = temp

                df11.loc[k, 'Diff'] = (df11.loc[k, 'list_price'] - df11.loc[k, 'bot_spread_OPAdj']) * (
                df11.loc[k, 'bot_spread_OPAdj'] - df11.loc[k, 'tmc'])

            print('Is there negative difference?')
            print(is_there_negative_difference(df11))

            if not is_there_negative_difference(df11):
                df1 = pd.concat([df11, df12])
                print('Hello India...')
                print(no_negative_difference)
                no_negative_difference = True
                print(no_negative_difference)
                break

        Error_new = df['bot_spread_OPAdj'].sum() - df['bot_spread_OP'].sum()
        print('New Error is...', Error_new)
        print(no_negative_difference)
        if no_negative_difference is True:
            print('Hello Bangalore...')
            print(no_negative_difference)
            # df1 = pd.concat([df11, df12])
            df = pd.concat([df1, df2])
            print('Search Successful!')
            return df
        else:
            print('Hello Noida...')
            print(no_negative_difference)
            # df1 = pd.concat([df11, df12])
            df = pd.concat([df1, df2])
            print('Search Unsuccessful!')
            return df

    elif Error > 0:
        print('Error is +ve!')
        print('Error is ', Error)
        for i in range(len(df)):
            df.loc[i, 'bot_spread_OPDiff'] = df.loc[i, 'bot_spread_OPAdj'] - df.loc[i, 'bot_spread_OP']

        df1 = df[df['bot_spread_OPDiff'] > 0]
        df2 = df[df['bot_spread_OPDiff'] < 0]

        df1 = df1.sort_values(by=['bot_spread_OPDiff'], ascending=False)

        df1.index = range(len(df1))
        for l in range(len(df1)):
            df1.loc[l, 'Diff'] = (df1.loc[l, 'list_price'] - df1.loc[l, 'bot_spread_OPAdj']) * (
            df1.loc[l, 'bot_spread_OPAdj'] - df1.loc[l, 'tmc'])

        for j in range(len(df1) - 1):
            print(len(df1))
            no_negative_difference = False
            # df1.index = range(len(df1))
            # df11.sort_values(by=['bot_spread_OPDiff'], ascending= False)
            df11 = df1.head(j + 1)
            df1 = df1.sort_values(by=['bot_spread_OPDiff'], ascending=True)
            print('Differences are...')
            print(df1['bot_spread_OPDiff'])
            df12 = df1.head(len(df1) - j - 1)

            df11 = df11.sort_values(by=['bot_spread_OPDiff'], ascending=False)
            # df11.sort_index(inplace=True, ascending=True)
            '''
            if j < len(df1)-1:
                df12 = df1.head(j+1)
            else:
                df12 = pd.DataFrame()
            '''
            print('Diffs are...')
            print(df11['bot_spread_OPDiff'])

            print('j is...', j)
            print('HelloWorld!...', len(df11))
            print('HelloRavi!...', len(df12))

            df11.index = range(len(df11))
            for k in range(len(df11)):
                temp = df11.loc[k, 'bot_spread_OPAdj'] - (1 / float(len(df11))) * (Error)
                df11.loc[k, 'bot_spread_OPAdj'] = temp

                df11.loc[k, 'Diff'] = (df11.loc[k, 'list_price'] - df11.loc[k, 'bot_spread_OPAdj']) * (
                df11.loc[k, 'bot_spread_OPAdj'] - df11.loc[k, 'tmc'])

            print('Is there negative difference?')
            print(is_there_negative_difference(df11))

            if not is_there_negative_difference(df11):
                df1 = pd.concat([df11, df12])
                print('Hello India...')
                print(no_negative_difference)
                no_negative_difference = True
                print(no_negative_difference)
                break

        Error_new = df['bot_spread_OPAdj'].sum() - df['bot_spread_OP'].sum()
        print('New Error is...', Error_new)
        print(no_negative_difference)
        if no_negative_difference is True:
            print('Hello Bangalore...')
            print(no_negative_difference)
            # df1 = pd.concat([df11, df12])
            df = pd.concat([df1, df2])
            print('Search Successful!')
            return df
        else:
            print('Hello Noida...')
            print(no_negative_difference)
            # df1 = pd.concat([df11, df12])
            df = pd.concat([df1, df2])
            print('Search Unsuccessful!')
            return df
