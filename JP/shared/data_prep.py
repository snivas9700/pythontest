from pandas import merge
from numpy import log

SBC_COLS = ['sbc'+str(i) for i in range(1, 7)]


# only applies to component-level data for now
def prep_comp(data, sbc_map=None, source='train'):# sbc_map):#edited by Bonnie (removed sbc_map)
    dat = data.copy()

    # additional filters
    #dat = dat.loc[(dat['quoted_price'] > 1) & (dat['list_price'] > 1)].reset_index(drop=True)
    # additional filters
    if ( source=='train'):
        # additional filters
        dat = dat.loc[(dat['quoted_price'] > 1) & (dat['list_price'] > 1)].reset_index(drop=True)
    else:
        dat = dat.reset_index(drop=True)
        
    # build additional input features
    dat.loc[:, 'com_cost_price'] = dat['comcostpofl'] * dat['list_price']
    dat.loc[dat['com_cost_price'].eq(0), 'com_cost_price'] = 1.
    dat['comcostpofl'] = dat['com_cost_price'] / dat['list_price']

    dat['componentid'] = dat.groupby('quoteid').apply(lambda df: df.reset_index()).drop(
        ['index', 'quoteid'], axis=1).reset_index().rename(columns={'level_1': 'componentid'})['componentid']

    # dealsize
    ds = dat.groupby('quoteid')['com_cost_price'].sum().to_frame('deal_size').reset_index()
    dat = merge(dat, ds, on='quoteid', how='left').reset_index(drop=True)
    if any(dat['deal_size'].isnull()):
        missing_ds = dat.loc[dat['deal_size'].isnull(), 'quote_id'].unique()
        print(('The following quotes had null deal_size values: {}'.format(missing_ds)))
        dat = dat.loc[dat['deal_size'].notnull()].reset_index(drop=True)

    # natural log of dealsize
    dat.loc[:, 'ln_ds'] = dat['deal_size'].apply(log)
    # proportion each component contributes to overall dealsize
    dat.loc[:, 'com_contrib'] = dat['com_cost_price']/dat['deal_size']

    # define target variable(s)
    dat.loc[:, 'discount'] = 1. - dat['quoted_price']/dat['list_price']

    dat.loc[:, 'crmindustryname'] = dat.loc[:, 'crmindustryname'].astype(str).replace('.', '_').replace('&', ' ').fillna('NA')
    dat.loc[:, 'crmsectorname'] = dat.loc[:, 'crmsectorname'].astype(str).fillna('NA')

    dat['quant_intercept'] = 1 #added by BB on May 31th, 2018 (chen)    

    # new component-level fields
    # Unique Feature codes 1/0
#    dat.loc[:, 'ufc_incl'] = (data['ufc'].astype(str).fillna('').apply(lambda x: x.strip()) != '') * 1

    # ### SBC handling ###
    #dat.loc[:, SBC_COLS] = dat.loc[:, SBC_COLS].fillna('').astype(str)
    #dat.loc[:, 'n_sbc'] = dat.apply(lambda row: sum([row[col].strip() != '' for col in SBC_COLS]), axis=1)
    #dat.loc[:, 'sbc_incl'] = dat.loc[:, 'n_sbc'] > 0

    # TODO - clean this up
    # find entries that need to be updated, because they contain sbc values
    #sbc_filt = dat.loc[:, 'sbc_incl'].eq(1)
    #if sum(sbc_filt) > 0:
        # update will either be sbc value with highest count, per sbc_map, or number non-null sbc entries
        # only apply logic to subset of data that needs to be updated
        #tmp = dat.loc[sbc_filt, SBC_COLS]
        #sbc = dict(zip(sbc_map['sbc'].astype(int).astype(str), sbc_map['count']))

#        def find_max_sbc(row, sbc):
#            if any([x in list(sbc.keys()) for x in row.values]):
#                match_keys = [x for x in row.values if x in list(sbc.keys())]
#                matches = sorted(match_keys, key=lambda x: sbc[x], reverse=True)
#                return str(int(float(matches[0])))  # clean up string-cast value
#            else:
#                return '1'
#
#        tmp['max_sbc'] = tmp.apply(lambda row: find_max_sbc(row, sbc), axis=1)
#
#        dat.loc[sbc_filt, 'lvl1'] = dat.loc[sbc_filt, 'lvl1'] + '_' + tmp['max_sbc']  # append to lvl1_ prefix
#        dat.loc[~sbc_filt, 'lvl1'] = dat.loc[~sbc_filt, 'lvl1'] + '_0'

#    else:
#        # if no sbc values present, then lvl1 = lvl1_0, so default to that
#        dat.loc[:, 'lvl1'] = dat.loc[:, 'lvl1'] + '_0'

    return dat


def prep_quote(data, source='train'):

    keep_cols = ['quoteid', 'com_componentid', 'upgmes', 'value_score', 'countrycode', 'list_price',
                 'quoted_price', 'lvl3', 'indirect', 'tmc']
    if source == 'train':
        keep_cols += ['winloss', 'quote_date']

    dat = data.loc[:, keep_cols].copy()

    # TODO do we want to keep this filter?
    mask = dat['value_score'] > dat['list_price']
    dat.loc[mask, 'value_score'] = dat.loc[mask, 'list_price']

    # calculate aggregate values for quote-level features, i.e. deal size (quote), list price (quote), etc
    grp_sum_cols = ['quoteid', 'value_score', 'list_price', 'quoted_price', 'tmc']
    sum_grp = dat.loc[:, grp_sum_cols].groupby(['quoteid']).sum()
    sum_grp = sum_grp.reset_index()

    # NOTE: combrand -> lvl3
    leading_val_cols = ['quoteid', 'lvl3', 'value_score']
    brand_val = dat.loc[:, leading_val_cols].groupby(
        ['quoteid', 'lvl3'], sort=True).sum().reset_index()

    leading_brand_first = brand_val.sort_values(by='value_score', ascending=False).groupby(
        'quoteid', as_index=False).first()

    leading_brand_first = leading_brand_first.rename(
        columns={'lvl3': 'leading_brand', 'value_score': 'leading_brand_value'})

    quote_grp = merge(sum_grp, leading_brand_first, on=['quoteid'], how='left')

    quote_grp['leading_brand_pct'] = quote_grp['leading_brand_value'] / quote_grp['value_score']

    meta_cols = ['quoteid', 'upgmes', 'countrycode', 'indirect']
    if source == 'train':
        meta_cols += ['winloss', 'quote_date']

    tc_flag = dat[meta_cols].groupby('quoteid').max().reset_index()
    quote_grp = merge(quote_grp, tc_flag, on=['quoteid'], how='inner')

    if source == 'train':
        quote_grp['APE'] = abs(quote_grp['quoted_price'] - quote_grp['value_score']) / quote_grp['quoted_price']

    quote_grp['ln_ds'] = log(quote_grp['list_price'] + 1.)
    quote_grp['ln_vs'] = log(quote_grp['value_score'] + 1.)
    quote_grp['gap_ev'] = (quote_grp['list_price'] - quote_grp['value_score']) / (quote_grp['value_score'] + 1.)
    quote_grp['tc'] = 1 * (quote_grp['upgmes'] == 1)

    if source == 'train':
        quote_grp['discount_qt'] = 1. - (quote_grp['quoted_price'] / quote_grp['list_price'])
        quote_grp['q_v'] = quote_grp['quoted_price'] / (quote_grp['value_score'] + 1.)
    
    # add quant_intercept for quantile regression
    quote_grp.loc[:,'quant_intercept'] = 1 #added by BB on May 30th, 2018     
    # return fresh indices
    quote_grp.reset_index(drop=True, inplace=True)

    return quote_grp
