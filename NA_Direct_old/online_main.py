from numpy import log10
#from os.path import join as os_join
from os.path import join as os_join, dirname, abspath
from pandas import DataFrame

from online.modeling_hw import run_quants_online
from online.utils_hw import transform_output
from shared.data_prep import prep_comp, prep_quote
from shared.utils_hw import build_output, spread_comp_quants
from shared.io_utils import load_data

from modeling.segmentation_utils import find_seg_id
from modeling.modeling_main import apply_lme

# scale on L/M/H price points
UPLIFT = 1.1

# scale on lower bound
ALPHA = 0.0

# A couple data fields are out of sync between CIO and CAO.
# FIELD_MAP maps CIO field: CAO field
FIELD_MAP = {
    'featureid': 'ufc'
    , 'comspclbidcode1': 'sbc1'
    , 'comspclbidcode2': 'sbc2'
    , 'comspclbidcode3': 'sbc3'
    , 'comspclbidcode4': 'sbc4'
    , 'comspclbidcode5': 'sbc5'
    , 'comspclbidcode6': 'sbc6'
    #, 'level_0': 'lvl0'
    , 'level_1': 'lvl1'
    , 'level_2': 'lvl2'
    , 'level_3': 'lvl3'
    , 'level_4': 'lvl4'
    , 'indirect(1/0)': 'indirect'
    , 'clientseg=e': 'client_e'
    , 'comlistprice': 'list_price'
    , 'comquoteprice': 'quoted_price'
    , 'comtmc': 'tmc'
}


def process_quote(df, loaded_models):
    # initial set of transforms
    df_ = df.copy().fillna('')
    df_.columns = [x.lower() for x in df_.columns]
    df_ = df_.rename(columns=FIELD_MAP)

    # TEMP
    df_ = df_.rename(columns={'customerindustryname': 'crmindustryname', 'customersecname': 'crmsectorname'})

    # assume dataframe input just like the training data "df"
    #region = 'NA' if df_['countrycode'].unique()[0] in ['US', 'CA'] else 'EMEA'
    x = df_['countrycode'].unique()[0]
    region =  'NA' if (x == 'CA' or x == 'US') else 'JP' if (x =='JP') else 'EMEA'


    # load in applicable models
    lme = loaded_models[region][0]
    seg_defs = loaded_models[region][1]
    quant_models = loaded_models[region][2]

    # prep component-level data
    #dat_path = os_join('NA_Direct\data', region)  # assumes sbc map is held in data/ folder
    dat_path = os_join(os_join(dirname(abspath(__file__)), 'data'), region)  # assumes sbc map is held in data/ folder
    sbc_map = load_data(dat_path, 'SBC')
    comp = prep_comp(df_, sbc_map,source='api')

    # apply LME to get discount, then project back to value_score space
    comp['discount'] = comp.apply(lambda row: apply_lme(row, lme), axis=1)
    comp['value_score'] = (1. - comp['discount']) * comp['list_price']
    comp['ln_vs'] = comp['value_score'].apply(log10)
    comp.to_csv('C:\\Model_Factory_JP\\CSV\\comp74.csv')
    # prep quote-level data (aggregate)
    quote = prep_quote(comp, source='api')
    quote.to_csv('C:\\Model_Factory_JP\\CSV\\quote77.csv')

    if isinstance(quote, DataFrame):
        quote = quote.iloc[0].copy()  # shape the quote-level data object

    # segment_id maps to a set of models
    quote['segment_id'] = find_seg_id(quote, seg_defs)

    # run quote-level quantile regression
    quote = run_quants_online(quote, quant_models['qt'], seg_field='segment_id', uplift=UPLIFT, alpha=ALPHA)
    quote.to_csv('C:\\Model_Factory_JP\\CSV\\quote87.csv')
    # spread OP down to components
    #comp.to_csv('C:\\22 June 2018\Result\\comp.csv')
    comp = comp.apply(lambda row: run_quants_online(row, quant_models['pn'], seg_field='lvl1', uplift=UPLIFT, alpha=ALPHA), axis=1)
    comp.to_csv('C:\\Model_Factory_JP\\CSV\\comp.csv')
    comp_adj, _ = spread_comp_quants(comp, quote, 'online', alpha=ALPHA)
    comp_adj.to_csv('C:\\Model_Factory_JP\\CSV\\comp_adj.csv')

    comp_prep, tot_stats = build_output(comp_adj, 'online')
    comp_prep.to_csv('C:\\Model_Factory_JP\\CSV\\comp_prep.csv')

    quote_out = transform_output(comp_prep)
    quote_out.to_csv('C:\\Model_Factory_JP\\CSV\\quote_out.csv')

    return quote_out
