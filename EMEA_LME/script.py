# Hardware-TSS pricing
# Aaron Slowey, Gloria Zhang, IBM Chief Analytics Office, 2018

import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import importlib  # to reload modules after editing

# PANDAS & SETTINGS
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('max_colwidth', 20)
pd.set_option('precision', 3)
pd.set_option('display.memory_usage', True) # use df.info(memory_usage='deep')
# If a calculation would produce -/+ inf, replace that result with NaN
# pd.set_option('mode.use_inf_as_null', True)

# Specify module location(s)
sys_path_root = 'C:\\Users\\\AARONSLOWEY\\Documents\\IBM_CAO'
sys.path.append(sys_path_root + '\\Pricing\\TSS\\src\\pricing_tss_hw')
# sys.path.append(sys_path_root + '\\Analytics\\Python0\\MyModules')
# sys.path.append(sys_path_root + '\\Analytics\\crispdm\\src')
# import inspect_filter as inspect
import aa_visualizations
from aa_visualizations import AAvisualization

path_root = 'C:/Users/AARONSLOWEY/Documents/IBM_CAO/Pricing/TSS'
path_in = path_root + '/data/'
path_in2 = path_root + '/data/from_collaborator/'
path_out = path_root + '/output/'

# ----------------------------------------------------------------------------
# Load data
# df_2017 = pd.read_excel(path_in + 'HW Data 2017.xlsx')
# df_tica = pd.read_csv(path_in + 'TICA_Europe_12052017.csv')

# Replace spaces in field names with underscores
df.columns = df.columns.str.replace(' ', '_')

# List field names to trim data after it is loaded
select_cols = ['auto_renewal_flg', 'bill_amt_usd', 'chnl',
               'comp_maint_period', 'comp_status', 'contr_nbr',
               'contr_nbr_chis', 'contr_start_date', 'contr_stop_date',
               'contr_typ', 'country', 'cust_name', 'date_inst',
               'date_warr_end', 'flag_attach_hwsale', 'flag_posthwsale_ma',
               'flag_only_wsu',
               'geo_name',
               'gross_amt_usd', 'mach_type', 'mmmc_mach_usd', 'prepay_ind',
               'product_description',
               'platform', 'serial5', 'service_lvl', 'subbrand', 'tss_type']

# Reformat the field names
# select_cols_upper = [x.upper() for x in select_cols]

# Read the data
dfx = pd.read_csv(path_in + 'tss_inv_hwma_joined.csv', encoding='latin-1')  #, error_bad_lines=False, quoting=csv.QUOTE_NONE)
dfx = pd.read_excel(path_in2 +
                    'INV_EP_CONTRACT_TYPE_v3_20171115_20112012installs.xlsx')

df = pd.read_excel(path_in + 'tss_inv_hwma_joined.xlsx')  #, usecols=select_cols_upper)

# Display basic information about the data and list the field names
df.info()
list(df)

# Display a more in-depth summary of the data
# rows, columns, area, dimensionality = inspect.dimensionality(df, indices)

# Change field name format to lowercase & reorder alphabetically
df.columns = map(str.lower, df)
df.sort_index(axis=1, inplace=True)

# Sample the data set
df2 = df[select_cols]

# ----------------------------------------------------------------------------
# Define the offering hierarchy with which to aggregate prices
indices = [['chnl', 'platform', 'contr_nbr'],
           []]  # Optionally include alternate hierarchies

# Index the data
df2.set_index(indices[0], inplace=True)

# Compute discount for each instance
df2.loc[:, 'discount_total'] = df2.mmmc_mach_usd / df2.bill_amt_usd - 1
df2.loc[:, 'discount_standard'] = df2.mmmc_mach_usd / df2.gross_amt_usd - 1

# View aggregate bid prices
df2.groupby(level=indices[0])['bill_amt_usd'].agg({np.sum, np.size})
df2.groupby(level=indices[0][:-1])['bill_amt_usd'].agg({np.sum, np.size})

# Plot the distribution of prices for each group
cdata = df2.groupby(level=indices[0][:-1])['bill_amt_usd'].sum() / 1000

plt.hist(cdata, log=True, bins=20, alpha=0.5)  # layout=(2, 2)
plt.title('Price distribution:\nIBM Systems (Power or Storage)-TSS bundles')
plt.xlabel('Bid price ($K)')
plt.ylabel('Count')


# ----------------------------------------------------------------------------
'''
Model approach
1. Component-level value score prediction
2. Bundle-level optimal price 
3. Split revenue among HW, SW, MA (maintenance; i.e., TSS)

Regressions
CMDA
1) HW price ~ bundle win rate
2) HW+TSS price ~ bundled win rate
Segments based on HW data, as no detailed TSS offering is available for lost sales opportunities

HW data
1) Component HW price in a bundle ~ list price, cost, client, product,
    channel, upgrade, deal size etc.
2) Predict bundled vs. non-bundled HW price

CHIS
1) Component MA price in a bundle ~ list price, contract length,
    contract size, warranty period, product, client, channel
2) TSS price, attached vs. non-attached
3) Stream revenue ~ 1st contract comp. value, TSS offering type,
    warranty years, client, contract size

Combined HW and CHIS data
1) won data only; how to correct today's high price issue?
'''
