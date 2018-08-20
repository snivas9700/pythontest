from EMEA_LME.model_def import config_wr_hw, config_wr_tsso
from pandas import DataFrame, Series, merge
from EMEA_LME.wr_predict import wr_train_hw, wr_train_tss, func
from numpy import sign, argmax
from datetime import datetime as dt
from EMEA_LME.utils import apply_bounds_hw, apply_bounds_tss


def apply_bounds_quote_hw(data):
    print('----------')
    print('Applying HW bounds...')
    print('----------')
    print('Bounds based on Value Score...')
    mask_h1 = (data['op_GP_pct_list_hw'] > 6.0 * data['y_pred'])
    mask_l1 = (data['op_GP_pct_list_hw'] < 0.5 * data['y_pred'])

    mask_h2 = (data['op_REV_pct_list_hw'] > 6.0 * data['y_pred'])
    mask_l2 = (data['op_REV_pct_list_hw'] < 0.5 * data['y_pred'])

    print('GP Optimal Price:')
    print('{}% hit upper bound by VS'.format(100*mask_h1.sum() / len(data.index)))
    print('{}% hit lower bound by VS'.format(100*mask_l1.sum() / len(data.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by VS'.format(100*mask_h2.sum() / len(data.index)))
    print('{}% hit lower bound by VS'.format(100*mask_l2.sum() / len(data.index)))

    data.loc[mask_h1, 'op_GP_pct_list_hw'] = 6.0 * data.loc[mask_h1, 'y_pred']
    data.loc[mask_l1, 'op_GP_pct_list_hw'] = 0.5 * data.loc[mask_l1, 'y_pred']

    data.loc[mask_h2, 'op_REV_pct_list_hw'] = 6.0 * data.loc[mask_h2, 'y_pred']
    data.loc[mask_l2, 'op_REV_pct_list_hw'] = 0.5 * data.loc[mask_l2, 'y_pred']

    # Second set of boundaries
    print('----------')
    print('Bounds based on List Price...')
    mask_h1 = (data['op_GP_pct_list_hw'] > 1.0)
    mask_l1 = (data['op_GP_pct_list_hw'] < 0.2)
    mask_h2 = (data['op_REV_pct_list_hw'] > 1.0)
    mask_l2 = (data['op_REV_pct_list_hw'] < 0.2)

    print('GP Optimal Price:')
    print('{}% hit upper bound by LP'.format(100*mask_h1.sum() / len(data.index)))
    print('{}% hit lower bound by LP'.format(100*mask_l1.sum() / len(data.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by LP'.format(100*mask_h2.sum() / len(data.index)))
    print('{}% hit lower bound by LP'.format(100*mask_l2.sum() / len(data.index)))

    data.loc[mask_h1, 'op_GP_pct_list_hw'] = 1.0
    data.loc[mask_l1, 'op_GP_pct_list_hw'] = 0.2
    data.loc[mask_h2, 'op_REV_pct_list_hw'] = 1.0
    data.loc[mask_l2, 'op_REV_pct_list_hw'] = 0.2

    # Third Set of Boundaries
    '''
    print('----------')
    print('Bounds based on Quantiles...')
    mask_h1 = (data['op_GP_pct_list_hw'] > data['pred_H1'] * data['y_pred'])
    mask_l1 = (data['op_GP_pct_list_hw'] < data['pred_L1'] * data['y_pred'])
    mask_h2 = (data['op_REV_pct_list_hw'] > data['pred_H1'] * data['y_pred'])
    mask_l2 = (data['op_REV_pct_list_hw'] < data['pred_L1'] * data['y_pred'])

    print('GP Optimal Price:')
    print('{}% hit upper bound by Quantile'.format(100*mask_h1.sum() / len(data.index)))
    print('{}% hit lower bound by Quantile'.format(100*mask_l1.sum() / len(data.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by Quantile'.format(100*mask_h2.sum() / len(data.index)))
    print('{}% hit lower bound by Quantile'.format(100*mask_l2.sum() / len(data.index)))

    data.loc[mask_h1, 'op_GP_pct_list_hw'] = data.loc[mask_h1, 'pred_H1'] * data.loc[mask_h1, 'y_pred']
    data.loc[mask_l1, 'op_GP_pct_list_hw'] = data.loc[mask_l1, 'pred_L1'] * data.loc[mask_l1, 'y_pred']
    data.loc[mask_h2, 'op_REV_pct_list_hw'] = data.loc[mask_h2, 'pred_H1'] * data.loc[mask_h2, 'y_pred']
    data.loc[mask_l2, 'op_REV_pct_list_hw'] = data.loc[mask_l2, 'pred_L1'] * data.loc[mask_l2, 'y_pred']
    '''

    data['optimal_price_GP_hw'] = data['op_GP_pct_list_hw'] * data['p_list_hw']
    data['optimal_price_REV_hw'] = data['op_REV_pct_list_hw'] * data['p_list_hw']

    print('Done with Applying Bounds to HW data.')
    return data


def apply_bounds_quote_tss(data):
    print('----------')
    print('Applying TSS bounds...')
    print('----------')
    print('Bounds based on Value Score...')
    mask_h1 = (data['op_GP_pct_list_hwma'] > 6.0 * data['y_pred'])
    mask_l1 = (data['op_GP_pct_list_hwma'] < 0.5 * data['y_pred'])

    mask_h2 = (data['op_PTI_pct_list_hwma'] > 6.0 * data['y_pred'])
    mask_l2 = (data['op_PTI_pct_list_hwma'] < 0.5 * data['y_pred'])

    mask_h3 = (data['op_REV_pct_list_hwma'] > 6.0 * data['y_pred'])
    mask_l3 = (data['op_REV_pct_list_hwma'] < 0.5 * data['y_pred'])

    print('GP Optimal Price:')
    print('{}% hit upper bound by VS'.format(100*mask_h1.sum() / len(data.index)))
    print('{}% hit lower bound by VS'.format(100*mask_l1.sum() / len(data.index)))

    print('PTI Optimal Price:')
    print('{}% hit upper bound by VS'.format(100*mask_h2.sum() / len(data.index)))
    print('{}% hit lower bound by VS'.format(100*mask_l2.sum() / len(data.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by VS'.format(100*mask_h3.sum() / len(data.index)))
    print('{}% hit lower bound by VS'.format(100*mask_l3.sum() / len(data.index)))

    data.loc[mask_h1, 'op_GP_pct_list_hwma'] = 6.0 * data.loc[mask_h1, 'y_pred']
    data.loc[mask_l1, 'op_GP_pct_list_hwma'] = 0.5 * data.loc[mask_l1, 'y_pred']

    data.loc[mask_h2, 'op_PTI_pct_list_hwma'] = 6.0 * data.loc[mask_h2, 'y_pred']
    data.loc[mask_l2, 'op_PTI_pct_list_hwma'] = 0.5 * data.loc[mask_l2, 'y_pred']

    data.loc[mask_h3, 'op_REV_pct_list_hwma'] = 6.0 * data.loc[mask_h3, 'y_pred']
    data.loc[mask_l3, 'op_REV_pct_list_hwma'] = 0.5 * data.loc[mask_l3, 'y_pred']

    # Second set of boundaries
    print('----------')
    print('Bounds based on List Price...')
    mask_h1 = (data['op_GP_pct_list_hwma'] > 1.0)
    mask_l1 = (data['op_GP_pct_list_hwma'] < 0.2)
    mask_h2 = (data['op_PTI_pct_list_hwma'] > 1.0)
    mask_l2 = (data['op_PTI_pct_list_hwma'] < 0.2)
    mask_h3 = (data['op_REV_pct_list_hwma'] > 1.0)
    mask_l3 = (data['op_REV_pct_list_hwma'] < 0.2)

    print('GP Optimal Price:')
    print('{}% hit upper bound by LP'.format(100*mask_h1.sum() / len(data.index)))
    print('{}% hit lower bound by LP'.format(100*mask_l1.sum() / len(data.index)))

    print('PTI Optimal Price:')
    print('{}% hit upper bound by LP'.format(100*mask_h2.sum() / len(data.index)))
    print('{}% hit lower bound by LP'.format(100*mask_l2.sum() / len(data.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by LP'.format(100*mask_h3.sum() / len(data.index)))
    print('{}% hit lower bound by LP'.format(100*mask_l3.sum() / len(data.index)))

    data.loc[mask_h1, 'op_GP_pct_list_hwma'] = 1.0
    data.loc[mask_l1, 'op_GP_pct_list_hwma'] = 0.2
    data.loc[mask_h2, 'op_PTI_pct_list_hwma'] = 1.0
    data.loc[mask_l2, 'op_PTI_pct_list_hwma'] = 0.2
    data.loc[mask_h3, 'op_REV_pct_list_hwma'] = 1.0
    data.loc[mask_l3, 'op_REV_pct_list_hwma'] = 0.2

    # Third Set of Boundaries
    '''
    print('----------')
    print('Bounds based on Quantiles...')
    mask_h1 = (data['op_GP_pct_list_hwma']  > data['pred_H']*data['y_pred'])
    mask_l1 = (data['op_GP_pct_list_hwma']  < data['pred_L']*data['y_pred'])
    mask_h2 = (data['op_PTI_pct_list_hwma'] > data['pred_H']*data['y_pred'])
    mask_l2 = (data['op_PTI_pct_list_hwma'] < data['pred_L']*data['y_pred'])
    mask_h3 = (data['op_REV_pct_list_hwma'] > data['pred_H']*data['y_pred'])
    mask_l3 = (data['op_REV_pct_list_hwma'] < data['pred_L']*data['y_pred'])

    print('GP Optimal Price:')
    print('{}% hit upper bound by Quantile'.format(100*mask_h1.sum() / len(data.index)))
    print('{}% hit lower bound by Quantile'.format(100*mask_l1.sum() / len(data.index)))

    print('PTI Optimal Price:')
    print('{}% hit upper bound by Quantile'.format(100*mask_h2.sum() / len(data.index)))
    print('{}% hit lower bound by Quantile'.format(100*mask_l2.sum() / len(data.index)))

    print('Revenue Optimal Price:')
    print('{}% hit upper bound by Quantile'.format(100*mask_h3.sum() / len(data.index)))
    print('{}% hit lower bound by Quantile'.format(100*mask_l3.sum() / len(data.index)))

    data.loc[mask_h1, 'op_GP_pct_list_hwma'] = data.loc[mask_h1,'pred_H']*data.loc[mask_h1,'y_pred']
    data.loc[mask_l1, 'op_GP_pct_list_hwma'] = data.loc[mask_l1,'pred_L']*data.loc[mask_l1,'y_pred']
    data.loc[mask_h2, 'op_PTI_pct_list_hwma'] = data.loc[mask_h2,'pred_H']*data.loc[mask_h2,'y_pred']
    data.loc[mask_l2, 'op_PTI_pct_list_hwma'] = data.loc[mask_l2,'pred_L']*data.loc[mask_l2,'y_pred']
    data.loc[mask_h3, 'op_REV_pct_list_hwma'] = data.loc[mask_h3,'pred_H']*data.loc[mask_h3,'y_pred']
    data.loc[mask_l3, 'op_REV_pct_list_hwma'] = data.loc[mask_l3,'pred_L']*data.loc[mask_l3,'y_pred']
    '''
    # Fourth set of boundaries
    print('----------')
    print('Bounds based on PTI5...')
    mask_l1 = (data['op_GP_pct_list_hwma'] < data['PTI_5price']/data['p_list_hwma'])
    mask_l2 = (data['op_PTI_pct_list_hwma'] < data['PTI_5price']/data['p_list_hwma'])
    mask_l3 = (data['op_REV_pct_list_hwma'] < data['PTI_5price']/data['p_list_hwma'])

    data.loc[mask_l1, 'op_GP_pct_list_hwma'] = data.loc[mask_l1, 'PTI_5price']/data.loc[mask_l1, 'p_list_hwma']
    data.loc[mask_l2, 'op_PTI_pct_list_hwma'] = data.loc[mask_l2, 'PTI_5price']/data.loc[mask_l2, 'p_list_hwma']
    data.loc[mask_l3, 'op_REV_pct_list_hwma'] = data.loc[mask_l3, 'PTI_5price']/data.loc[mask_l3, 'p_list_hwma']

    print('GP Optimal Price:')
    print('{}% hit lower bound by PTI5'.format(100*mask_l1.sum() / len(data.index)))

    print('PTI Optimal Price:')
    print('{}% hit lower bound by PTI5'.format(100*mask_l2.sum() / len(data.index)))

    print('Revenue Optimal Price:')
    print('{}% hit lower bound by PTI5'.format(100*mask_l3.sum() / len(data.index)))

    data['optimal_price_GP_tss'] = data['op_GP_pct_list_hwma'] * data['p_list_hwma']
    data['optimal_price_PTI_tss'] = data['op_PTI_pct_list_hwma'] * data['p_list_hwma']
    data['optimal_price_REV_tss'] = data['op_REV_pct_list_hwma'] * data['p_list_hwma']

    print('Done with Applying Bounds to tss.')
    return data


def hw_botline(data, config=config_wr_hw, uplift=1.0):
    # first aggregated hw data by quoteid
    quote = data.copy()
    cols = ['cost_hw', 'p_list_hw', 'p_bid_hw']
    pred_cols = [s for s in quote.columns if 'pred_' in s]
    quote.loc[:, 'y_pred'] = quote.loc[:, 'y_pred'] * quote.loc[:, 'p_list_hw']  # value score in dollar price
    for col in pred_cols:
        quote.loc[:, col] = quote.loc[:, col] * quote.loc[:, 'y_pred']  # pred_L1.. etc in dollar value

    # update the column list for aggregation
    cols = cols + pred_cols + ['y_pred','won']

    quote[cols] = quote[cols].astype(float)
    quote = DataFrame(quote.groupby('quoteid')[cols].sum()).reset_index()
    quote['won'] = sign(quote['won'])

    def f(x): # To find the leading brand
        return Series(dict(leading_brand=x['taxon_hw_level_4'][argmax(x['p_list_hw'])]))

    quote['leading_brand']=data.groupby('quoteid').apply(f).iloc[:,0].tolist()

    # update y_pred and all pred_* to be a fraction of the list price
    for col in pred_cols:
        quote.loc[:, col] = quote.loc[:, col] / quote.loc[:, 'y_pred']  # pred_L1.. etc in value score
    quote.loc[:, 'y_pred'] = quote.loc[:, 'y_pred'] / quote.loc[:, 'p_list_hw']  # value score in list price

    # then apply wr_train_hw
    quote = wr_train_hw(quote, config, uplift=uplift)

    quote_f = apply_bounds_quote_hw(quote.copy())

    return quote_f


# tss quote level optimal price
def tss_botline(data, config=config_wr_tsso, uplift=1.0):
    # first aggregated hw data by quoteid
    quote = data.copy()
    cols = ['cost', 'PTI_0price', 'PTI_5price', 'p_list_hwma', 'p_bid_hwma']
    pred_cols = [s for s in quote.columns if 'pred_' in s]
    quote.loc[:, 'y_pred'] = quote.loc[:, 'y_pred'] * quote.loc[:, 'p_list_hwma']  # value score in dollar price
    for col in pred_cols:
        quote.loc[:, col] = quote.loc[:, col] * quote.loc[:, 'y_pred']  # pred_L1.. etc in dollar value
    # update the column list for aggregation
    cols = cols + pred_cols + ['y_pred']
    # print(cols)
    quote[cols] = quote[cols].astype(float)
    quote = DataFrame(quote.groupby('contract_number')[cols].sum()).reset_index()

    def f(x): # To find the leading brand
        return Series(dict(leading_brand=x['taxon_hw_level_4'][argmax(x['p_list_hwma'])]))

    quote['leading_brand'] = data.groupby('contract_number').apply(f).iloc[:,0].tolist()

    # update y_pred and all pred_* to be a fraction of the list price
    for col in pred_cols:
        quote.loc[:, col] = quote.loc[:, col] / quote.loc[:, 'y_pred']  # pred_L1.. etc in value score
    quote.loc[:, 'y_pred'] = quote.loc[:, 'y_pred'] / quote.loc[:, 'p_list_hwma']  # value score in list price

    # then apply wr_train_hw
    quote = wr_train_tss(quote, config, uplift=uplift)

    quote = apply_bounds_quote_tss(quote)

    return quote


# hw bottom-line spreading function
def spread_optimal_price_hw(quote_df, botline_df):

    quote_df = quote_df.reset_index()

    # This section creates a spread_price dataframe of component price price points & includes a total row
    spread_price = quote_df.loc[:,
                   ['index', 'quoteid', 'componentid', 'adj_price_L', 'adj_price_M', 'adj_price_H', 'list_price']]

    spread_price['zero_price'] = 0.
    spread_price['10x_price'] = spread_price['list_price'] * 10

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
    adj_spread_price[['index', 'quoteid', 'componentid']] = spread_price[['index', 'quoteid', 'componentid']]

    tots = adj_spread_price.groupby('quoteid')[spread_cols].sum().reset_index()

    tots = merge(tots, botline_df.reset_index()[['bot_OP', 'quoteid']], on='quoteid', how='inner')

    # NOTE: bot_OP CANNOT be above 10x list_price
    N = tots.loc[tots['bot_OP'] > tots['10x_price']].shape[0]
    if N > 0:
        print(('Reducing OP on {} entries to keep them at 10x price.'.format(N)))
        tots.loc[tots['bot_OP'] > tots['10x_price'], 'bot_OP'] = tots.loc[
            tots['bot_OP'] > tots['10x_price'], '10x_price']

    mask_cols = list(zip(spread_cols[:-1], spread_cols[1:]))

    # build mask
    mask = tots.apply(lambda row: Series(
        [((row[cols[0]] < row['bot_OP']) & (row['bot_OP'] <= row[cols[1]])) * 1 for cols in mask_cols]), axis=1)
    mask['quoteid'] = tots['quoteid']

    # duplicate mask entry per quote across all components of the quote
    mask_ = merge(mask, spread_price[['index', 'quoteid', 'componentid']], on='quoteid', how='right').drop(
        ['quoteid', 'componentid'], axis=1)

    # This section spreads the bottom line optimal price to the line items
    # first reorder adj_spread_price by index
    adj_spread_price = adj_spread_price.sort_values(by=['index'])
    mask_ = mask_.sort_values(by=['index'])
    mask_ = mask_.drop(columns=['index'])

    # pull out lower bounds to be filtered
    low_prices = adj_spread_price[list(spread_cols)[:-1]].values  # pull out matrix
    # multiply column-wise to filter out column of interest + sum
    adj_spread_price['spread_low'] = (low_prices * mask_.values).sum(axis=1)

    # pull out higher bounds to be filtered
    high_prices = adj_spread_price[list(spread_cols)[1:]].values
    adj_spread_price['spread_high'] = (high_prices * mask_.values).sum(axis=1)

    tot_spread = adj_spread_price.groupby('quoteid')[['spread_low', 'spread_high']].sum().reset_index()

    tots = merge(tots, tot_spread, on='quoteid')
    tots['alpha'] = ((tots['bot_OP'] - tots['spread_low']) / (tots['spread_high'] - tots['spread_low'])).fillna(0.)

    adj_spread_price = merge(adj_spread_price, tots[['quoteid', 'alpha']], on='quoteid', how='left')
    adj_spread_price['spread_price'] = adj_spread_price['spread_low'] + \
                                       (adj_spread_price['spread_high'] - adj_spread_price['spread_low']) * \
                                       adj_spread_price['alpha']

    # This section loads the spread optimal prices to the quote_df dataframe
    quote_df = quote_df.set_index(['quoteid', 'componentid'])
    quote_df['bot_spread_OP_hw'] = adj_spread_price.set_index(['quoteid', 'componentid'])['spread_price']

    quote_df = quote_df.reset_index()

    quote_df = quote_df.drop(columns=['index'])

    quote_df['p_list_hw'] = 10.0 ** quote_df['p_list_hw_log']
    quote_df['Win_Rate_at_bot_spread_OP_hw'] = func([quote_df.P1, quote_df.P2, quote_df.P3],
                                                    quote_df.bot_spread_OP_hw / (quote_df.p_list_hw * quote_df.y_pred), 0)

    quote_df['optimal_price_GP_hw'] = quote_df['op_GP_pct_list_hw'] * quote_df['p_list_hw']
    quote_df['optimal_price_REV_hw'] = quote_df['op_REV_pct_list_hw'] * quote_df['p_list_hw']

    quote_df.drop(columns=['p_list_hw'], inplace=True)

    return quote_df


def spread_optimal_price_tss(quote_df, botline_df):

    # reset index but keep the index column for later merge
    quote_df = quote_df.reset_index()

    # This section creates a spread_price dataframe of component price price points & includes a total row
    spread_price = quote_df.loc[:,
                   ['index', 'contract_number', 'tsscomid', 'adj_price_L', 'adj_price_M', 'adj_price_H', 'list_price']]

    spread_price['zero_price'] = 0.
    spread_price['10x_price'] = spread_price['list_price'] * 10

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
    adj_spread_price[['index', 'contract_number', 'tsscomid']] = spread_price[['index', 'contract_number', 'tsscomid']]

    tots = adj_spread_price.groupby('contract_number')[spread_cols].sum().reset_index()

    tots = merge(tots, botline_df.reset_index()[['bot_OP', 'contract_number']], on='contract_number', how='inner')

    # NOTE: bot_OP CANNOT be above 10x list_price
    N = tots.loc[tots['bot_OP'] > tots['10x_price']].shape[0]
    if N > 0:
        print(('Reducing OP on {} entries to keep them at 10x price.'.format(N)))
        tots.loc[tots['bot_OP'] > tots['10x_price'], 'bot_OP'] = tots.loc[
            tots['bot_OP'] > tots['10x_price'], '10x_price']

    mask_cols = list(zip(spread_cols[:-1], spread_cols[1:]))

    # build mask
    mask = tots.apply(lambda row: Series(
        [((row[cols[0]] < row['bot_OP']) & (row['bot_OP'] <= row[cols[1]])) * 1 for cols in mask_cols]), axis=1)
    mask['contract_number'] = tots['contract_number']

    # duplicate mask entry per quote across all components of the quote
    mask_ = merge(mask, spread_price[['index', 'contract_number', 'tsscomid']], on='contract_number', how='right').drop(
        ['contract_number', 'tsscomid'], axis=1)

    # This section spreads the bottom line optimal price to the line items

    # pull out lower bounds to be filtered
    # first reorder adj_spread_price by index
    adj_spread_price = adj_spread_price.sort_values(by=['index'])
    mask_ = mask_.sort_values(by=['index'])
    mask_ = mask_.drop(columns=['index'])

    low_prices = adj_spread_price[list(spread_cols)[:-1]].values  # pull out matrix
    # multiply column-wise to filter out column of interest + sum
    adj_spread_price['spread_low'] = (low_prices * mask_.values).sum(axis=1)

    # pull out higher bounds to be filtered
    high_prices = adj_spread_price[list(spread_cols)[1:]].values
    adj_spread_price['spread_high'] = (high_prices * mask_.values).sum(axis=1)

    tot_spread = adj_spread_price.groupby('contract_number')[['spread_low', 'spread_high']].sum().reset_index()

    tots = merge(tots, tot_spread, on='contract_number')
    tots['alpha'] = ((tots['bot_OP'] - tots['spread_low']) / (tots['spread_high'] - tots['spread_low'])).fillna(0.)

    adj_spread_price = merge(adj_spread_price, tots[['contract_number', 'alpha']], on='contract_number', how='left')
    adj_spread_price['spread_price'] = adj_spread_price['spread_low'] + \
                                       (adj_spread_price['spread_high'] - adj_spread_price['spread_low']) * \
                                       adj_spread_price['alpha']

    # This section loads the spread optimal prices to the quote_df dataframe
    quote_df = quote_df.set_index(['contract_number', 'tsscomid'])
    quote_df['bot_spread_OP_tss'] = adj_spread_price.set_index(['contract_number', 'tsscomid'])['spread_price']
    quote_df = quote_df.reset_index()

    quote_df = quote_df.drop(columns=['index'])

    quote_df['p_list_hwma'] = 10 ** quote_df['p_list_hwma_log']
    quote_df['Win_Rate_at_bot_spread_OP_tss'] = func([quote_df.P1, quote_df.P2, quote_df.P3],
                                                     quote_df.bot_spread_OP_tss /(quote_df.p_list_hwma * quote_df.y_pred),
                                                     0)

    quote_df['optimal_price_GP_tss'] = quote_df['op_GP_pct_list_hwma'] * quote_df['p_list_hwma']
    quote_df['optimal_price_PTI_tss'] = quote_df['op_PTI_pct_list_hwma'] * quote_df['p_list_hwma']
    quote_df['optimal_price_REV_tss'] = quote_df['op_REV_pct_list_hwma'] * quote_df['p_list_hwma']
    quote_df.drop(columns=['p_list_hwma'], inplace=True)
    return quote_df


def spread_hw(com_hw, HW_UPLIFT=1.0, spread_opt=True):

    hw_opt_col = 'optimal_price_GP_hw'
    hw_opt_col1 = 'optimal_price_GP_unbounded_hw'

    if 'index' in com_hw.columns:
        com_hw = com_hw.drop(columns=['index'])
    com_hw.rename(columns={'quote_id': 'quoteid', 'component_id': 'componentid'}, inplace=True)

    q_hw = hw_botline(data=com_hw, config=config_wr_hw, uplift=HW_UPLIFT)

    com_hw_map = {'adj_price_L': 'pred_L',
                  'adj_price_M': 'pred_M',
                  'adj_price_H': 'pred_H',
                  'list_price': 'p_list_hw'}

    com_hw = com_hw.rename(columns={k: v for v, k in com_hw_map.items()})

    # if spreading by optimal price
    if spread_opt:
        com_hw['opt_VS'] = com_hw[hw_opt_col1] / (com_hw['y_pred'] * com_hw['list_price'])
        com_hw.rename(columns={'adj_price_M': 'adj_price_M_0'}, inplace=True)
        com_hw.rename(columns={'opt_VS': 'adj_price_M'}, inplace=True)

    # convert these pricing columns to dollar values:
    cols = [v for v, k in com_hw_map.items()][:-1]
    for col in cols:
        com_hw[col] = com_hw[col] * (com_hw['y_pred'] * com_hw['list_price'])
    # make a copy of the optimal column in q_hw to be the one for spreading

    q_hw['bot_OP'] = q_hw[hw_opt_col]

    com_hw = spread_optimal_price_hw(com_hw, q_hw)

    # convert the pricing columns back to be normalized by value score
    for col in cols:
        com_hw[col] = com_hw[col] / (com_hw['y_pred'] * com_hw['list_price'])

    # after spreading (if) by optimal price, rename the corresponding column names back
    if spread_opt:
        com_hw.rename(columns={'adj_price_M': 'opt_VS'}, inplace=True)
        com_hw.rename(columns={'adj_price_M_0': 'adj_price_M'}, inplace=True)

    # rename the columns back back
    com_hw = com_hw.rename(columns={v: k for v, k in com_hw_map.items()})

    return q_hw, com_hw


def spread_tss(com_tss, TSS_UPLIFT=1.0, spread_opt=True, tss_cost='GP'):

    if tss_cost == 'GP':
        tss_opt_col1 = 'optimal_price_GP_unbounded_tss'
        tss_opt_col = 'optimal_price_GP_tss'
    elif tss_cost == 'PTI':
        tss_opt_col1 = 'optimal_price_PTI_unbounded_tss'
        tss_opt_col = 'optimal_price_PTI_tss'

    if 'index' in com_tss.columns:
        com_tss = com_tss.drop(columns=['index'])

    if ~('tsscomid' in com_tss.columns):
        com_tss.reset_index(inplace=True)
        com_tss.rename(columns={'index': 'tsscomid'}, inplace=True)

    q_tss = tss_botline(com_tss, config=config_wr_tsso, uplift=TSS_UPLIFT)

    com_tss_map = {'adj_price_L': 'pred_L',
                   'adj_price_M': 'pred_M',
                   'adj_price_H': 'pred_H',
                   'list_price': 'p_list_hwma'}

    com_tss = com_tss.rename(columns={k: v for v, k in com_tss_map.items()})

    # if spreading by optimal price
    if spread_opt:
        com_tss['opt_VS'] = com_tss[tss_opt_col1] / (com_tss['y_pred'] * com_tss['list_price'])
        com_tss.rename(columns={'adj_price_M': 'adj_price_M_0'}, inplace=True)
        com_tss.rename(columns={'opt_VS': 'adj_price_M'}, inplace=True)

    # convert these pricing columns to dollar values:
    cols = [v for v, k in com_tss_map.items()][:-1]
    for col in cols:
        com_tss[col] = com_tss[col] * (com_tss['y_pred'] * com_tss['list_price'])

    # make a copy of the optimal column in q_hw to be the one for spreading

    q_tss['bot_OP'] = q_tss[tss_opt_col]

    # spread the deal bottom-line price to line components
    com_tss = spread_optimal_price_tss(com_tss, q_tss)

    # convert the pricing columns back to be normalized by value score
    for col in cols:
        com_tss[col] = com_tss[col] / (com_tss['y_pred'] * com_tss['list_price'])

    # after spreading (if) by optimal price, rename the corresponding column names back
    if spread_opt:
        com_tss.rename(columns={'adj_price_M': 'opt_VS'}, inplace=True)
        com_tss.rename(columns={'adj_price_M_0': 'adj_price_M'}, inplace=True)

    # rename the columns back back
    com_tss = com_tss.rename(columns={v: k for v, k in com_tss_map.items()})
    return q_tss, com_tss