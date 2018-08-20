'''''
Current equation in LME-TSS model
# define the config of each run
config = {'model':
              {'formula_fe': 'p_bid_log ~ lncomlp + lnconlp '
                            # 'p_bid_log ~ lncomlp + lnconlp '
                             '+ coverage_days*coverage_hours'
                             '+ comp_duration_days + response_time_lvl1 +'
                             ' response_time_lvl2'
                             '+ comp_sched_ind + sale_hwma_attached'
                             '+ chnl + committed + tss_type'
               , 'formula_re': ' (lncomlp||hw_hier_level4/hw_hier_level3/hw_hier_level2/mtm_id)'
                               # ' + (comp_list_value||response_time_lvl1/response_time_lvl2)'
                               # ' + (comp_list_value||hw_hier_level4/hw_hier_level3/hw_hier_level2/mtm_id)'

               }
          }
'''''

# ------Code reuse from main.py------
# ------line 342 343
# Gloria: the intention is to aggregate all the TSS component list prices. Same logic applies to incoming quote.
#         However incoming quote don't have contract_number.
#         All component under <TssComponents> tag, and mark as <servoffcode> HW-MAINTENANCE, WSU, TMS or HDR
#         should be aggregated together to represent p_list_all_hwma.
#
#         Today, in the main.py code, we haven't include the code of cleansing and merging TMS and HDR data.
#         After we add that part, it's possible this column name will change.
hwma['p_list_all_hwma'] = hwma.groupby('contract_number'). \
        p_list.transform(lambda x: x.sum())

# ------line 620
# Gloria: this is HW section. This should be consistent with what we are developing for COPRA HW.
#         However, the data column are renamed in main.py. I think it might be critical that
#         we either in line with column naming in current COPRA HW engine, or the naming in the main.py code.
# @Kyle can you pls add your comment here?

# Derive contract values from component values
# Multiply price & cost by unit quantity, as one instance can include
# more than one unit, and then append the group total to each instance
df_hw['p_list_hw_row_total'] = df_hw.p_list_hw * df_hw.com_quantity
df_hw['p_list_all_hw'] = df_hw.groupby(
    'quote_id').p_list_hw_row_total.transform(lambda x: x.sum())

#df_hw['p_pct_list_hw'] = df_hw.p_bid_hw / df_hw.p_list_hw
df_hw['p_delgl4_pct_list_hw'] = df_hw.p_delgl4_hw / df_hw.p_list_hw

#df_hw['p_bid_hw_row_total'] = df_hw.p_bid_hw * df_hw.com_quantity
#df_hw['p_bid_hw_total'] = df_hw.groupby(
#    'quote_id').p_bid_hw_row_total.transform(lambda x: x.sum())
df_hw['p_delgl4_hw_row_total'] = df_hw.p_delgl4_hw * \
                                 df_hw.com_quantity
df_hw['p_delgl4_hw_total'] = df_hw.groupby(
    'quote_id').p_delgl4_hw_row_total.transform(lambda x: x.sum())

# Derive scaled & normalized hardware cost metrics
df_hw['cost_hw_row_total'] = df_hw.cost_hw * df_hw.com_quantity
df_hw['cost_hw_total'] = df_hw.groupby(
    'quote_id').cost_hw_row_total.transform(lambda x: x.sum())
df_hw['cost_pct_list_hw'] = df_hw.cost_hw / df_hw.p_list_hw

# ------line 702
# Gloria: Starting from here, hw and hwma data are merged.
# Derive deal size as sum of the list prices of all HWMA & HW components
hwma_hw['p_list_hwma_hw'] = hwma_hw.p_list_all_hwma + \
                            hwma_hw.p_list_all_hw

#hwma_hw['p_hwma_pct'] = hwma_hw.p_bid_hwma_total / (
#            hwma_hw.p_bid_hwma_total + hwma_hw.p_bid_hw_total)

# Gloria: some engineering are currently in tss_lme.py code.
# ------Code reuse from tss_lme.py------
# ------line 66-80
#X['p_bid_log'] = np.log10(X.p_bid+0.1)
X['p_list_log'] = np.log10(X.p_list+0.1)
X['p_list_hwma_hw_log'] = np.log10(X.p_list_hwma_hw+0.1)
X['lncomlp'] = log10(X['p_list']+0.1)
X['lnconlp'] = log10(X['p_list_all_hwma']+0.1)

X = X.loc[~(X['coverage_days'] == 'other')]
X['coverage_days'].astype('int')
X['coverage_hours'].astype('int')
X['comp_duration_days'].astype('int')
X['response_time_lvl2'].astype('int')  # Gloria: this column name, and column value format may pending to change.
