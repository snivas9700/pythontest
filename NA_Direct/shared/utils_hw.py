from pandas import DataFrame, Series, merge
from datetime import datetime as dt
from numpy import log, exp

from shared.modeling_hw import calc_mf_opt
from shared.modeling_utils import calc_quant_op, calc_win_prob, opt_price_ci, price_adj


def build_comp_out(comp):
    comp_out = comp.copy()

    # prep "quote_df" per PricingEngine.py (lines 246-390), but using local field names
    comp_out['price_M'] = comp_out['pred_M'] * comp_out['value_score']
    comp_out['adj_price_L'] = comp_out['adj_pred_L'] * comp_out['value_score']
    comp_out['adj_price_M'] = comp_out['adj_pred_M'] * comp_out['value_score']
    comp_out['adj_price_H'] = comp_out['adj_pred_H'] * comp_out['value_score']
    comp_out['OP_VS'] = comp_out['price_opt'] / comp_out['value_score']
    comp_out['opt_GP'] = comp_out['price_opt'] - comp_out['tmc']
    comp_out['opt_EGP'] = comp_out['opt_GP'] * comp_out['wp_opt']
    comp_out['opt_ci_low'] = comp_out['ci_low'] * comp_out['value_score']
    comp_out['opt_ci_high'] = comp_out['ci_high'] * comp_out['value_score']

    return comp_out


def build_tot_stats(comp_out):
    # this section contains general quote totals - PricingEngine.py lines 392-473
    # NOTE: modified to work with online (single quote) and offline (multi-quote) settings

    stats = DataFrame()
    grp = comp_out.groupby('quoteid')

    stats['list_price'] = grp['list_price'].sum()
    stats['deal_size'] = grp['price_M'].sum().round(decimals=2)
    stats['tmc'] = grp['tmc'].sum()
    stats['price_opt'] = grp['price_opt'].sum().round(decimals=0)
    # this section contains Price Range Data (Line Item Sum)
    stats['adj_price_L'] = grp['adj_price_L'].sum().round(decimals=0)
    stats['adj_price_M'] = grp['adj_price_M'].sum().round(decimals=0)
    stats['adj_price_H'] = grp['adj_price_H'].sum().round(decimals=0)
    stats['opt_GP'] = grp['opt_GP'].sum().round(decimals=2)
    stats['opt_EGP'] = grp['opt_EGP'].sum().round(decimals=2)
    stats['opt_wp'] = 0.

    stats.loc[stats['opt_GP'] > 0., 'opt_wp'] = stats['opt_EGP'] / stats['opt_GP']

    stats['opt_ci_low'] = grp['opt_ci_low'].sum().round(decimals=0)
    stats['opt_ci_high'] = grp['opt_ci_high'].sum().round(decimals=0)

    stats['bot_OP'] = stats.apply(lambda row: calc_quant_op(
        row['adj_price_L'], row['adj_price_M'], row['adj_price_H'], row['tmc'], 0, 0, 0), axis=1)

    stats['bot_OP_wp'] = stats.apply(lambda row: calc_win_prob(
        row['bot_OP'], row['adj_price_L'], row['adj_price_M'], row['adj_price_H']), axis=1)

    stats[['bot_ci_low', 'bot_ci_high']] = stats.apply(lambda row: Series(opt_price_ci(
        row['bot_OP'], row['adj_price_L'], row['adj_price_M'], row['adj_price_H'], row['tmc'])), axis=1)

    stats['bot_GP'] = stats['bot_OP'] - stats['tmc']
    stats['bot_EGP'] = stats['bot_GP'] * stats['bot_OP_wp']
    stats = stats.reset_index() # force the quoteid from index to column #added by BB on May 30th, 2018
    return stats


def build_output(comp, source):
    comp_out = build_comp_out(comp)

    stats = build_tot_stats(comp_out)

    c_out = spread_optimal_price(comp_out, stats)

    if source == 'online':
        stats = stats.iloc[0]  # convert to Series object, as expected by CIO

    return c_out, stats


# Per CIO reqs, this section builds the quote_df object found in PricingEngine.py (from old Model Factory code)
#  lines 188-242 (quote_df) and lines 329-433 (total_deal_stats)
def transform_output(prep_comp):
    # prep quote-level output

    # NOTE: original Model Factory was all in terms of list price. New approach bases everything off value_score
    #       these fields are all left as "PofL", i.e. Percent of List, for convenience (not having to bother CIO),
    #       but they are actually Price of Value Score; PofV

    # maps quote_df col -> comp col
    quote_map = {
        'ComListPrice': 'list_price'
        , 'DealSize': 'deal_size'
        , 'LogDealSize': 'ln_ds'
        , 'ValueScore': 'value_score'
        , 'LogValueScore': 'ln_vs'
        , 'ComTMC': 'tmc'
        , 'ComTMCPofL': 'cf'
        , 'ComLowPofL': 'pred_L'
        , 'ComMedPofL': 'pred_M'
        , 'ComHighPofL': 'pred_H'
        , 'ComMedPrice': 'price_M'
        , 'AdjComLowPofL': 'adj_pred_L'
        , 'AdjComMedPofL': 'adj_pred_M'
        , 'AdjComHighPofL': 'adj_pred_H'
        , 'AdjComLowPrice': 'adj_price_L'
        , 'AdjComMedPrice': 'adj_price_M'
        , 'AdjComHighPrice': 'adj_price_H'
        , 'OptimalPricePofL': 'OP_VS'
        , 'OptimalPrice': 'price_opt'
        , 'OptimalPriceWinProb': 'wp_opt'
        , 'OptimalPriceGP': 'opt_GP'
        , 'OptimalPriceExpectedGP': 'opt_EGP'
        , 'OptimalPriceIntervalLow': 'opt_ci_low'
        , 'OptimalPriceIntervalHigh': 'opt_ci_high'
        , 'DealBotLineSpreadOptimalPrice': 'bot_spread_OP'
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

    quote_df['PredictedQuotePricePofL'] = quote_df['OptimalPricePofL']
    quote_df['PredictedQuotePrice'] = quote_df['OptimalPrice']

    return quote_df


def spread_optimal_price(quote_df, tot_deal_stats):
    # This section creates a spread_price dataframe of component price price points & includes a total row
    spread_price = quote_df.loc[:, ['quoteid', 'componentid', 'adj_price_L', 'adj_price_M', 'adj_price_H', 'list_price']]

    # spread_price.insert(0, 'zero_price', 0)
    spread_price['zero_price'] = 0.
    spread_price['10x_price'] = spread_price['list_price'] * 10
    # spread_price.loc['total'] = spread_price.sum().values
    # spread_price['total'] = spread_price.groupby('quoteid').sum()

    # This section creates an adj_spread_price dataframe that removes "mountains"
    # (i.e for price points to left of optimal price:
    #           take lowest price of current column up to the optimal price column
    #      for price points to right of optimal price:
    #           take highest price of current column down to the optimal price column)
    adj_spread_price = DataFrame()
    spread_cols = ('zero_price', 'adj_price_L', 'adj_price_M', 'adj_price_H', 'list_price', '10x_price')

    # adjust price points - confirm they are in order
    adj_spread_price['zero_price'] = spread_price.loc[:, spread_cols].min(axis=1)
    adj_spread_price['adj_price_L'] = spread_price.loc[:, ['adj_price_L', 'adj_price_M']].min(axis=1)
    adj_spread_price['adj_price_M'] = spread_price['adj_price_M']
    adj_spread_price['adj_price_H'] = spread_price.loc[:, ['adj_price_M', 'adj_price_H']].max(axis=1)
    adj_spread_price['list_price'] = spread_price.loc[:, ['adj_price_M', 'adj_price_H', 'list_price']].max(axis=1)
    adj_spread_price['10x_price'] = spread_price['10x_price']

    # bring in quoteid so this can all be manipulated + re-joined in later
    adj_spread_price[['quoteid', 'componentid']] = spread_price[['quoteid', 'componentid']]

    tots = adj_spread_price.groupby('quoteid')[spread_cols].sum().reset_index()

    tots = merge(tots, tot_deal_stats.reset_index()[['bot_OP', 'quoteid']], on='quoteid', how='inner')

    # NOTE: bot_OP CANNOT be above 10x list_price
    N = tots.loc[tots['bot_OP'] > tots['10x_price']].shape[0]
    if N > 0:
        print(('Reducing OP on {} entries to keep them at 10x price.'.format(N)))
        tots.loc[tots['bot_OP'] > tots['10x_price'], 'bot_OP'] = tots.loc[tots['bot_OP'] > tots['10x_price'], '10x_price']

    mask_cols = list(zip(spread_cols[:-1], spread_cols[1:]))

    # build mask
    mask = tots.apply(lambda row: Series([((row[cols[0]] < row['bot_OP']) & (row['bot_OP'] <= row[cols[1]]))*1 for cols in mask_cols]), axis=1)
    mask['quoteid'] = tots['quoteid']

    # duplicate mask entry per quote across all components of the quote
    mask_ = merge(mask, spread_price[['quoteid', 'componentid']], on='quoteid', how='right').drop(['quoteid', 'componentid'], axis=1)

    # This section spreads the bottom line optimal price to the line items

    # pull out lower bounds to be filtered
    low_prices = adj_spread_price[list(spread_cols)[:-1]].values  # pull out matrix
    # multiply column-wise to filter out column of interest + sum
    adj_spread_price['spread_low'] = (low_prices * mask_.values).sum(axis=1)

    # pull out higher bounds to be filtered
    high_prices = adj_spread_price[list(spread_cols)[1:]].values
    adj_spread_price['spread_high'] = (high_prices * mask_.values).sum(axis=1)

    tot_spread = adj_spread_price.groupby('quoteid')[['spread_low', 'spread_high']].sum().reset_index()

    tots = merge(tots, tot_spread, on='quoteid')
    tots['alpha'] = ((tots['bot_OP'] - tots['spread_low'])/(tots['spread_high'] - tots['spread_low'])).fillna(0.)

    adj_spread_price = merge(adj_spread_price, tots[['quoteid', 'alpha']], on='quoteid', how='left')
    adj_spread_price['spread_price'] = adj_spread_price['spread_low'] + \
                   (adj_spread_price['spread_high'] - adj_spread_price['spread_low']) * adj_spread_price['alpha']

    # This section loads the spread optimal prices to the quote_df dataframe
    quote_df = quote_df.set_index(['quoteid', 'componentid'])
    quote_df['bot_spread_OP'] = adj_spread_price.set_index(['quoteid', 'componentid'])['spread_price']
    quote_df = quote_df.reset_index()
    # quote_df.to_csv(data_path + 'spread_optimal_price.csv', index=False)

    return quote_df


def spread_comp_quants(comp, quote, source, alpha):
    dat = comp.copy()
    q = quote.copy()

    adj_cols = ['adj_pred_L', 'adj_pred_M', 'adj_pred_H']

    # online flow assumes very small comp and quote objects
    if source == 'online':
        # recalculate adj_L/M/H as comp_adj_L * (quote_adj_L/sum(comp_adj_L))
        for col in adj_cols:
            dat[col] = dat[col] * ((q[col] * q['value_score']) / (dat[col] * dat['value_score']).sum())

        # re-adjust the redistributed L/M/H
        dat[adj_cols] = dat.apply(lambda row: Series(price_adj(*row[adj_cols].values)), axis=1)

    # offline flow assumes large comp and quote objects
    else:
        q_cols = [x + '_q' for x in adj_cols]  # quote-level price points
        t_cols = [x + '_t' for x in adj_cols]  # component total price points

        # sum price points by quote (across components)
        tots = dat.groupby('quoteid').apply(lambda df: (df[adj_cols].multiply(df['value_score'], axis=0).sum(axis=0))).reset_index()

        # join aggregate/quote-level data together
        q_ = merge(q[['quoteid', 'value_score'] + adj_cols], tots[['quoteid'] + adj_cols],
                   on='quoteid', how='inner', suffixes=('_q', '_t'))
        q_.rename(columns={'value_score': 'value_score_q'}, inplace=True)

        # bring in aggregate data
        # NOTE: This adds in extra columns with _q and _t suffixes. Will drop later
        dat = merge(dat, q_[['quoteid', 'value_score_q'] + q_cols + t_cols], on='quoteid', how='inner')
        for col in adj_cols:
            q_col = col + '_q'  # ID the quote-level value
            t_col = col + '_t'  # ID the sum component total
            # recalculate adj_L/M/H as comp_adj_L * (quote_adj_L/sum(comp_adj_L))
            dat[col] = dat[col] * ((dat[q_col] * dat['value_score_q'])/dat[t_col])  # redistribute the price point

        dat = dat.drop(q_cols + t_cols + ['value_score_q'], axis=1)  # removed added columns

        # re-adjust the redistributed L/M/H
        dat[adj_cols] = dat.apply(lambda row: Series(price_adj(*row[adj_cols].values)), axis=1)

    # recalculate OP
    dat = dat.apply(lambda row: calc_mf_opt(row, alpha), axis=1)

    return dat, q


def build_quarter_map(data):
    # builds dataframe of quote_id, quarter_designation
    dat = data[['quoteid', 'quote_date']].drop_duplicates(subset='quote_date')

    # training data SUBMIT_DATE has format YYYYMMDD
    dat['date'] = dat['quote_date'].apply(lambda x: dt.strptime(str(x), '%m/%d/%y') if isinstance(x, str) else dt.strptime(str(x), '%Y%m%d'))

    dat['quarter'] = dat['date'].apply(lambda x: str((x.month-1)//3+1) + 'Q' + str(x.year)[2:] )

    return dat[['quoteid', 'quarter']]


# TODO - optimize this
def apply_bounds(data_optprice):
    print('Applying bounds...')

    mask_h = (data_optprice['price_opt'] > 1.2 * data_optprice['Value'])
    mask_l = (data_optprice['price_opt'] < 0.8 * data_optprice['Value'])

    print(('{}% hit upper bound by VS'.format(mask_h.sum() / len(data_optprice.index))))
    print(('{}% hit lower bound by VS'.format(mask_l.sum() / len(data_optprice.index))))

    data_optprice.loc[mask_h, 'price_opt'] = 1.2 * data_optprice.loc[mask_h, 'Value']
    data_optprice.loc[mask_l, 'price_opt'] = 0.8 * data_optprice.loc[mask_l, 'Value']

    mask_h = (data_optprice['price_opt'] > data_optprice['ENTITLED_SW'])
    mask_l = (data_optprice['price_opt'] < 0.15 * data_optprice['ENTITLED_SW'])

    print(('{}% hit upper bound by EP'.format(mask_h.sum() / len(data_optprice.index))))
    print(('{}% hit lower bound by EP'.format(mask_l.sum() / len(data_optprice.index))))

    data_optprice.loc[mask_h, 'price_opt'] = data_optprice.loc[mask_h, 'ENTITLED_SW']
    data_optprice.loc[mask_l, 'price_opt'] = 0.15 * data_optprice.loc[mask_l, 'ENTITLED_SW']

    data_optprice['Discount_opt'] = 1 - data_optprice['price_opt'] / data_optprice['ENTITLED_SW']

    print('Done')
    return data_optprice


# use train_vs (component-level value score calc dataframe) to determine quote type (SSW/SaaS/mixed)
# and apply that type designation to the optimal price output
def add_quote_type(op, vs):
    type_dict = vs.groupby('WEB_QUOTE_NUM').apply(
        lambda df: 'mixed' if len(df['SSW_Saas'].unique()) > 1 else df['SSW_Saas'].unique()[0]).to_dict()

    op['type'] = op['WEB_QUOTE_NUM'].map(type_dict)
    return op


def post_process_op(optprice, vs, quarters):
    optprice['unbounded_price_opt'] = optprice['price_opt']

    optprice['deviance'] = -log(1 - optprice['wp_act'])
    optprice.loc[optprice['WIN'], 'deviance'] = -log(optprice.loc[optprice['WIN'], 'wp_act'])
    optprice['mean_deviance'] = 1 - exp(-optprice['deviance'].mean())

    optprice = merge(optprice, quarters, on='WEB_QUOTE_NUM', how='left')

    op = apply_bounds(optprice.copy())
    op = add_quote_type(op=op, vs=vs)  # to help the test suite

    return op
