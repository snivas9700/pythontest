# IBM Systems hardware - GTS-Tech Support Services pricing
# Aaron Slowey, Gloria Zhang, Jason Zhu, IBM Chief Analytics Office, 2018

import os
import importlib
import itertools
import numpy as np
import pandas as pd
import datetime as dt
import time as time
import re
from sklearn.preprocessing import StandardScaler, RobustScaler
import operator
# import pandas_profiling  # creates a data description in html
import matplotlib as mpl
import matplotlib.pyplot as plt
pd.set_option('precision', 3)
pd.set_option('display.memory_usage', True)
pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 10)
# import time
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# Specify module location(s) & import user-defined modules
# Windows
# sys_path_root = 'C:\\Users\\AARONSLOWEY\\Documents\\IBM_CAO'
# sys.path.append(sys_path_root + '\\Pricing\\TSS\\src\\pricing_tss_hw')
# Mac
sys_path_root = '~/Documents/Pricing/tss_hw/'
sys.path.append(sys_path_root)

import cleansing as prep
import upstream as upstr
# import winrate
import myplotlibrary4_3 as myplt
from plot_regional_settings import *
from PlotColors import *

# path_root = 'C:/Users/AARONSLOWEY/Documents/IBM_CAO/Pricing/TSS'
path_root = os.path.join(os.path.expanduser('~'), 'Documents', 'Pricing',
                         'tss_hw')
path_in = os.path.join(os.path.expanduser('~'), 'Documents', 'Pricing',
                       'tss_hw', 'data/')
# path_in = path_root + '/data/from_collaborator'
path_out = os.path.join(os.path.expanduser('~'), 'Documents', 'Pricing',
                        'tss_hw', 'output/')  #, 'training_data/')


def prep_hwma(path_in, path_out, year=2017):
    """Prepares IBM Systems (Power & Storage) hardware maintenance contract
    data assembled by Auguste Lacroix, which consist of Power and Storage
    inventory visible in BMT as
    installed in year 2013 with CHIS-registered active contract information
    (HWMA, HX EOS EXTension or WSU) with either Active, Stopped or Future
    billing info. The field comp_sched_ind tells if the record/component is
    Active, Stopped or Future as of 31 Jan 2018. The data include TSS
    service level (sl_sdesc), of which there are 63 distinct service levels,
    and date_calc_start and date_calc_stop

    Because there are several records for a given machine serial number, the
    field 'rn' can be used to slice the data, using a key value of 1 to
    display each machine serial number once.

    The data were originally in xlsb format (compressed Excel), and was
    converted to xlsx

    Prices and other financials are on an annual basis, unless otherwise
    indicated.  Other variables are on the HW or TSS component level,
    unless otherwise indicated (e.g., contract level).  Unless labeled as
    hardware attributes with _hw, all variables are TSS attributes.

    Args:
        path: folder containing the input data in Excel (xlsx) format

    Returns:
        Prepared data set (DataFrame) and a table logging the number of
        instances after operations that could result in a gain or loss
    """
    task = 'Load'
    hwma = pd.read_excel(path_in)
    # df_hwma = [pd.read_excel(path+subfolder2+file) for file in
    #                          files_hwma[:-2]]
    #
    # Concatenate list of DataFrames into one DataFrame -- active contracts
    # hwma = pd.concat(df_hwma[0:2])

    # Record current number of rows in the data set & place into a table
    n1 = len(hwma)
    process_log = pd.DataFrame([{'task': task, 'instances': n1, 'gain': 0}])

    # ------------------------------------------------------------------------
    # Remove whitespace around field names & values of select fields
    hwma = prep.nospace_lowercase(hwma, cols2rename={
        'contr_nbr': 'contract_number', 'model': 'machine_model',
        'ser_nbr': 'serial_number', 'typ': 'machine_type',
        'cust_name': 'customer_name', 'cust_nbr': 'customer_number', 'geo_cd':
        'geo', 'cntry': 'country_code', 'cntry_desc': 'country',
        'gross_amt_usd': 'p_list_per', 'bill_amt_usd': 'p_bid_per',
        'calc_start_date': 'date_calc_start',
        'calc_stop_date': 'date_calc_stop',
        'comp_start_date': 'date_comp_start',
        'comp_stop_date': 'date_comp_stop',
        'contr_start_date': 'date_contr_start',
        'contr_stop_date': 'date_contr_stop',
        'discnt_tot': 'discount0', 'fctr': 'p_uplift_comm',
        'sl_cd': 'sl_code', 'sl_cntct_nm': 'sl_cntct',
        'sl_onsite_nm': 'sl_onsite', 'sl_fix_time_nm': 'sl_fix_time',
        'sl_part_time_nm': 'sl_part_time', 'sl_cov_nm': 'sl_cov',
        'offer_nm': 'tss_type', 'offer_sdesc': 'tss_type_desc'},
        trim_values=True,
        fields=['country', 'contract_number', 'machine_model',
                'contr_nbr_chis', 'ff_chnl_sdesc', 'tss_type',
                'tss_type_desc'])

    # In a separate field, retain the trailing five characters of each
    # serial number
    hwma['serial5'] = hwma['serial_number'].str[-5:]

    # Change field types to string -- does not preserve when exported to csv
    # and read back in
    hwma.machine_type = [val.zfill(4) for val in hwma.machine_type.
        astype('str')]
    hwma.machine_model = [val.zfill(3) for val in hwma.machine_model.
        astype('str')]

    # Concatenate machine type & machine model to produce 'machine type model'
    hwma['mtm_tss'] = hwma.machine_type + hwma.machine_model

    # Some years in date files are 9999; i.e., a placeholder for an unknown
    # date; this will cause to_datetime to yield NaT (null) values
    # pd.to_datetime(hwma.date_comp_stop, errors='coerce')

    hwma.date_comp_stop = hwma.date_comp_stop.astype('str')
    hwma.date_contr_stop = hwma.date_contr_stop.astype('str')

    hwma.loc[hwma.date_comp_stop == '9999-12-31 00:00:00',
                 'date_comp_stop'] = '2099-12-31 00:00:00'
    hwma.loc[hwma.date_contr_stop == '9999-12-31 00:00:00',
                 'date_contr_stop'] = '2099-12-31 00:00:00'

    # Reformat date information for hwma
    hwma = prep.make_a_date(hwma, ['date_calc_start', 'date_calc_stop',
                                   'date_contr_start', 'date_contr_stop',
                                   'date_comp_start', 'date_comp_stop',
                                   'date_inst', 'date_warr_end',
                                   'date_srv_start', 'date_srv_stop'])

    # Assume TSS component services marked to stop in 1999 will instead end
    # 10 years after their start date
    hwma.loc[hwma.date_contr_stop.dt.year == 1999,
             'date_contr_stop'] = hwma.date_contr_start + pd.DateOffset(
             years=5)
    hwma.loc[hwma.date_comp_stop.dt.year == 1999,
             'date_comp_stop'] = hwma.date_comp_start + pd.DateOffset(
             years=5)

    # ----------------------------------------------------------------------------
    n2017_1 = len(hwma[hwma.date_comp_start.dt.year >= year])
    n2017_2 = len(hwma[(hwma.date_comp_start.dt.year >= year) &
                       (hwma.serial5.notnull())])

    # Select the time window; to see the TSS and ePricer match rate,
    # narrow the install time range relative to the range of ePricer ship dates
    # mach_ce_stat: 2 (under maintenance), 3 (no maintenance), or 9 (under
    #  base warranty, so not billing TSS)
    hwma = hwma[(hwma.date_inst >= '2014-01-01') &
                (hwma.date_inst <= '2018-03-01') &
                (hwma.date_inst < (hwma.date_warr_end - pd.DateOffset(days=
                 180))) & (hwma.mach_ce_stat != 3)]

    # Check instances where a TSS contract commenced prior to the
    # installation of associated hardware
    # hwma[hwma.date_contr_start < hwma.date_inst][['date_contr_start',
    #                                               'date_inst']]

    # ----------------------------------------------------------------------------
    task = 'Removed NaN (ser. no.)'
    hwma.dropna(subset=['serial_number'], inplace=True)

    # Check how many records were removed: record number of rows in DataFrame
    n2 = len(hwma)
    instance_loss = n2 - n1
    next_task = pd.DataFrame([{'task': task, 'instances': n2,
                               'gain': instance_loss}])
    process_log = process_log.append(next_task)

    task = 'Removed NaN (other fields)'
    hwma.dropna(subset=['contract_number', 'p_bid_per',
                        'p_list_per', 'date_calc_start',
                        'date_calc_stop', 'date_contr_start',
                        'date_contr_stop', 'date_comp_start',
                        'date_comp_stop'], inplace=True)

    # Check how many records were removed: record number of rows in DataFrame
    n3 = len(hwma)
    instance_loss = n3 - n2
    next_task = pd.DataFrame([{'task': task, 'instances': n3,
                             'gain': instance_loss}])
    process_log = process_log.append(next_task)

    task = 'Removed <0 billings & bill>gross & rn!=1'
    # rn is a slicer; if only records with rn = 1 are included, there will be
    # no repetitions of machine serial number, for which there might be
    # multiple records
    hwma = hwma[(hwma.p_bid_per > 0) &
                        (hwma.p_list_per >= hwma.p_bid_per) &
                        (hwma.rn == 1)]

    # Check how many records were removed: record number of rows in DataFrame
    n4 = len(hwma)
    instance_loss = n4 - n3
    next_task = pd.DataFrame([{'task': task, 'instances': n4,
                             'gain': instance_loss}])
    process_log = process_log.append(next_task)

    task = 'Remove contr extensions'
    # by removing duplicates with respect to select machine & contract
    # attributes where time-of-HW sale and extensions will share common
    # values, retaining the oldest instance representing the most likely
    # time-of-sale instance.  However, this approach is not removing many
    # instances.  Stefanie suggested another field "F0001"
    hwma = hwma.sort_values('date_contr_start', ascending=False).\
        drop_duplicates(subset=['machine_type', 'serial_number',
                                'contract_number'], keep='last')

    # Check how many records were removed: record number of rows in DataFrame
    n5 = len(hwma)
    instance_loss = n5 - n4
    next_task = pd.DataFrame([{'task': task, 'instances': n5,
                             'gain': instance_loss}])
    process_log = process_log.append(next_task)

    task = 'Remove flex warranties'
    # Flex warranties are provided with certain machine types (listed below)
    # Use pandas IndexSlice to
    storage_flex_bundles = [2078, 2147, 2422, 2423, 2424, 2810, 2812, 2833,
                            2834, 3403, 5147, 9837, 9838, 9843, 9848]
    # 2072, 2832, 2836, 2837, 3403, 9836

    mt_dict = {}
    for item in hwma.machine_type.unique():
        mt_dict[item] = item
    all_mt = mt_dict.keys()

    not_flex = [mt for mt in all_mt if mt not in
                storage_flex_bundles]

    hwma.set_index('machine_type', inplace=True)
    hwma.sort_index(inplace=True)

    # For a list of values, specify an axis parameter to loc
    idx = pd.IndexSlice
    hwma = hwma.loc(axis=0)[idx[not_flex]]
    hwma.reset_index(inplace=True)

    # Check how many records were removed: record number of rows in DataFrame
    n6 = len(hwma)
    instance_loss = n6 - n5
    next_task = pd.DataFrame([{'task': task, 'instances': n6,
                               'gain': instance_loss}])
    process_log = process_log.append(next_task)

    # Infer attachment (bundling) of hardware maintenance at point of hardware
    # sale
    # Engineer categorical variable
    # positive label = TSS contract is attached to HW sales
    # Rule 1. contract signed or service started within 60 days of hardware
    #  installation or
    # Rule 2. hardware installed before half a year to warranty end
    # Rule 3. service stopped on day warranty ended, mainly to address cases
    # where contract starts long after installation

    # o---o-----<-?->-------o---------------o---------o
    #    +60d             -365 d           end      +200d
    # |---|-attached starts
    #                       |----aftermarket starts---|
    hwma['sale_attached'] = 1.0 * \
         ((hwma.date_contr_start < hwma.date_inst + pd.DateOffset(days=60)) |
           (hwma.date_srv_start < hwma.date_inst + pd.DateOffset(days=60)) |
           (hwma.date_srv_stop == hwma.date_warr_end))
           # (hwma.date_approval_hw < hwma.date_inst + pd.DateOffset(days=60)) |

    # Provision category attached where dates used above are blank

    # Engineer categorical variable: no contract attached to HW sales
    # (i.e., client signed contract after warranty ended)
    # Rules: -contract sign date later than install date
    #  -contract sign date can be earlier or later than warranty expire date
    #  -contract end date has to be later than warranty date of expiration
    #  -Machine warranty ended at the time of exporting this data (important!)
    # Note: for warr_exit, because we only use 2016 install or later data,
    # many of the machines are still under warranty.  For machines under
    # warranty, we default the warr_exit to 0 (i.e., not exited), because we
    # don't know when exit will occur
    # hwma['sale_aftermarket'] = 1.0 * \
    #     ((hwma.date_contr_start < hwma.date_warr_end + pd.DateOffset(
    #         days=200)) &
    #      (hwma.date_contr_start > hwma.date_warr_end - pd.DateOffset(
    #              days=365)) &
    #      (hwma.date_contr_start > hwma.date_inst + pd.DateOffset(days=30)) &
    #      (hwma.date_contr_stop > hwma.date_warr_end))
    # penultimate condition may be too stringent, or not stringent enough?
    # Using contr or comp stop dates could be problematic in cases where
    # the original value was adjusted from 1999 to an assumed start + 10 years
    # If confident in sale_attached, derive sale_aftermarket as its inverse
    # hwma['sale_aftermarket'] = hwma.sale_attached.where(hwma.sale_attached
    #                                                     == 0, 1)

    # Engineer categorical variable: TSS attached to HW sales is solely a
    # warranty service upgrade (WSU)
    # Rule: contract sign or start date has to be earlier than warranty exp
    #  date.
    # Rule continue: AND contract end date has to be ealrier or stopped at
    # warranty expired date
    # hwma['sale_wsu_only'] = 1.0 * \
    #     (((hwma.date_contr_start < hwma.date_inst + pd.DateOffset(days=15)) |
    #       (hwma.date_srv_start < hwma.date_inst + pd.DateOffset(days=15))) &
    #      ((hwma.date_contr_stop <= hwma.date_warr_end) |
    #       (hwma.date_srv_stop <= hwma.date_warr_end) |
    #       (hwma.date_comp_stop <= hwma.date_warr_end)))

    # Derive component & contract values (list and realized) from various
    # dates
    hwma['comp_inst_year'] = hwma.date_inst.dt.year
    hwma['comp_duration_days'] = (hwma.date_comp_stop -
                                  hwma.date_comp_start).dt.days
    hwma['period_billing_days'] = (hwma.date_calc_stop -
                                   hwma.date_calc_start).dt.days
    hwma['period_billing_months'] = np.round(hwma.period_billing_days / 30)
    # hwma['contract_periods'] = np.ceil(hwma.comp_duration_days /
    #                                    hwma.period_billing_days)
    hwma['comp_periods'] = np.ceil(hwma.comp_duration_days /
                                   hwma.period_billing_days)
    # hwma['contract_list_value'] = hwma.p_list_per * hwma.contract_periods
    # hwma['contract_real_value'] = hwma.p_bid_per * hwma.contract_periods
    hwma['p_list_hwma'] = hwma.p_list_per * hwma.comp_periods
    hwma['p_bid_hwma'] = hwma.p_bid_per * hwma.comp_periods
    hwma['p_pct_list_hwma'] = hwma.p_bid_per / hwma.p_list_per
    hwma['discount_hwma'] = 1 - hwma.p_pct_list_hwma

    # Derive contract values from component values
    hwma['p_list_hwma_total'] = hwma.groupby('contract_number'). \
        p_list_hwma.transform(lambda x: x.sum())

    hwma['p_bid_hwma_total'] = hwma.groupby('contract_number'). \
        p_bid_hwma.transform(lambda x: x.sum())

    # Extract numeric data from strings
    # Consider any value as a category, including numeric values, because
    # NBD, etc. cannot be translated to a number without knowing when a
    # call for service happens; e.g., on a Friday, NBD is Monday, not 24 later
    # hwma.sl_fix_time = [re.sub('FXT', '', re.sub('COM', '', s)).strip() for
    #                     s in hwma.sl_fix_time.astype('str')]
    # # re.findall(r'\d+', 'FXT 48H COM')
    # hwma.sl_onsite = [re.sub('ORT', '', re.sub('TARGET', '', s)).strip() for
    #                     s in hwma.sl_onsite.astype('str')]
    # hwma.sl_part_time = [re.sub('FXT', '', re.sub('COM', '', s)).strip() for
    #                     s in hwma.sl_part_time.astype('str')]
    hwma.sl_fix_time = [re.sub('FXT', '', s).strip() for s in
                        hwma.sl_fix_time.astype('str')]
    # re.findall(r'\d+', 'FXT 48H COM')
    hwma.sl_onsite = [re.sub('ORT', '', s).strip() for s in
                      hwma.sl_onsite.astype('str')]
    hwma.sl_part_time = [re.sub('PAT', '', s).strip() for s in
                         hwma.sl_part_time.astype('str')]

    # In a new field, convert the continuous (numeric) committed service
    # uplift to a binary categorical variable
    hwma.loc[:, 'committed'] = hwma.p_uplift_comm.where(hwma.p_uplift_comm
                                                       == 0, 1)
    # Maintain null values
    # hwma.loc[:, 'committed'] = hwma.p_uplift_com.where(
    #         hwma.p_uplift_comm.isnull())

    task = 'Removed <0 bill periods'
    hwma = hwma[hwma.period_billing_months > 0]

    # Check how many records were removed: record number of rows in DataFrame
    n7 = len(hwma)
    instance_loss = n7 - n6
    next_task = pd.DataFrame([{'task': task, 'instances': n7,
                             'gain': instance_loss}])
    process_log = process_log.append(next_task)

    # For regression, artificially increase zero discounts by a tiny amount
    hwma[hwma.discount_hwma == 0].discount_hwma += 0.0001

    # To merge with GTMS, MSS (Supportline) data going forward; without
    # merging with other data sets, these new fields are useless
    # W EOS EXT refers to contracts where TSS continues to maintain machines
    # of which the End of Service date has been reached
    # task = 'Merge quote-level aggregates & remove HW EOS EXT Offerings'
    # hwma['part_type'] = 'HWMA'
    # hwma.loc[hwma.offer_nm == 'WSU', 'part_type'] = 'WSU'
    # hwma.loc[hwma.offer_nm == 'HW EOS EXT', 'part_type'] = 'other'
    # hwma = hwma[hwma.part_type != 'other']

    task = 'Removed EOS ext'
    hwma = hwma[hwma.tss_type != 'HW EOS EXT']

    # Check how many records were removed: record number of rows in DataFrame
    n8 = len(hwma)
    instance_loss = n8 - n7
    next_task = pd.DataFrame([{'task': task, 'instances': n8,
                             'gain': instance_loss}])
    process_log = process_log.append(next_task)

    # Log record counts for specific year
    task = str(year) + 'records'
    next_task = pd.DataFrame([{'task': task, 'instances': n2017_1,
                               'gain': np.nan}])
    process_log = process_log.append(next_task)

    task = str(year) + ' records with SN'
    next_task = pd.DataFrame([{'task': task, 'instances': n2017_2,
                               'gain': np.nan}])
    process_log = process_log.append(next_task)

    # Remove unneeded unreliable column(s)
    hwma.drop(['mmmc_mach_usd', 'rn', 'date_calc_start', 'date_calc_stop',
               'date_inst_sof'], axis=1, inplace=True)

    # Sort columns by name
    hwma.sort_index(axis=1, inplace=True)

    # Write list of field names to a text file
    prep.write_fields(hwma, 'hwma', path_out)

    # Print processing log
    process_log.index = np.arange(len(process_log))
    print('HWMA preparation summary')
    print(process_log[['task', 'instances', 'gain']])
    process_log.to_excel(path_out + 'prep_log_hwma.xlsx', index=False)

    return hwma, process_log


def map_svl_hw_hier(df, file_svc, file_taxon, fields_tss, fields_hw,
                    keys_tss, keys_hw, path_in, path_out):
    """
    Maps to sets of fields: TSS service level attributes & IBM Systems
    product taxonomy; TSS service level attributes are or were not included
    in the primary data set

    Args:
        df: Larger DataFrame to which to append the attributes
        file_svc (str): file containing the TSS service level attributes
        file_taxon (str): file containing the IBM Systems product taxonomy
        fields_hw (list): selects a subset of fields in the Systems taxonomy
        keys_tss (list): merge (join) key(s) for TSS service level attributes
        keys_hw (list): merge (join) key(s) for Systems taxonomy
        path_in (str): path where TSS service level attribute file is stored
        path_out (str): path where IBM Systems product taxonomy file is stored

    Returns:
        Input DataFrame with TSS service level attributes & Systems
        taxonomy fields mapped

    """
    # Defer to TSS client's interpretation of portions of sl_desc
    tic = time.time()
    print('Loading & processing TSS service level attributes...')
    service_attribs = pd.read_excel(path_in + file_svc, sheet_name='EMEA',
                                na_values='-')
    service_attribs = service_attribs[fields_tss]

    # Ensure consistency with keys prior to merging by reformatting
    service_attribs = prep.nospace_lowercase(service_attribs, cols2rename={
        'slc': 'sl_code', 'cov_value': 'sl_hours_days'}, trim_values=True,
                                             categorize=False)
    service_attribs.sl_code = service_attribs.sl_code.str.upper()

    service_attribs.loc[service_attribs.cnt_unit == 'h', 'sl_contact_hours']\
        = service_attribs.cnt_value
    service_attribs.loc[service_attribs.cnt_unit == 'd',
                        'sl_contact_hours'] = service_attribs.cnt_value * 24

    service_attribs.loc[service_attribs.ort_unit == 'h', 'sl_onsite_hours']\
        = service_attribs.ort_value
    service_attribs.loc[service_attribs.ort_unit == 'd', 'sl_onsite_hours']\
        = service_attribs.ort_value * 24

    service_attribs.loc[service_attribs.pat_unit == 'h', 'sl_part_hours'] = \
        service_attribs.pat_value
    service_attribs.loc[service_attribs.pat_unit == 'd', 'sl_part_hours'] = \
        service_attribs.pat_value * 24

    # These fields currently lack too many values
    # service_attribs.loc[service_attribs.fxt_unit == 'h', 'sl_fix_hours'] = \
    #     service_attribs.fxt_value
    # service_attribs.loc[service_attribs.fxt_unit == 'd', 'sl_fix_hours'] = \
    #     service_attribs.fxt_value * 24

    # service_attribs.loc[service_attribs.tat_unit == 'h',
    # 'sl_turnaround_hours'] =  service_attribs.tat_value
    # service_attribs.loc[service_attribs.tat_unit == 'd',
    # 'sl_turnaround_hours'] = service_attribs.tat_value * 24

    # Compute product of hours x days; e.g., 11 x 5 = 55 hours
    service_attribs[['sl_cov_hours', 'sl_cov_days']] = \
        service_attribs.sl_hours_days.str.split('x', expand=True)
    service_attribs['sl_hours_x_days'] = service_attribs.sl_cov_hours.astype(
            'float64') * service_attribs.sl_cov_days.astype('float64')

    # Map to df_hwma
    # Ensure key field of primary TSS data set is not a categorical
    df.sl_code = df.sl_code.astype('str')

    svl_hour_fields = ['sl_code', 'sl_hours_x_days', 'sl_contact_hours',
                       'sl_onsite_hours', 'sl_part_hours']
    df_svl = pd.merge(df, service_attribs[svl_hour_fields], how='left',
                      on=keys_tss)
    toc = time.time()
    print('Merged in {:.0f} seconds'.format(toc - tic))

    # Evaluate match rate
    df_svl_ij = pd.merge(df, service_attribs[svl_hour_fields], on=keys_tss)
    match1 = len(df_svl_ij[keys_tss[0]].unique()) / len(df[keys_tss[0]].unique())

    print('{:.0f} of {:.0f} ({:.1%}) TSS unique SLCs matched with '
          'TSS service level attributes'.format(len(df_svl_ij[keys_tss[0]].unique()),
                                                len(df[keys_tss[0]].unique()),
                                                match1))

    # Derive year of contract start from the start date
    df_svl['contr_start_year'] = df_svl.date_contr_start.dt.year

    tic = time.time()
    print('Loading & processing hardware taxonomy...')
    # Map product taxonomy
    prod_taxon = pd.read_excel(path_in + file_taxon)
    prod_taxon = prep.nospace_lowercase(prod_taxon, cols2rename={
        'level_0': keys_hw[0]}, trim_values=True)
    prod_taxon = prod_taxon[fields_hw]

    prefix = 'taxon_hw_'
    prod_taxon.columns = [prefix + x for x in fields_hw]

    # prod_taxon.machine_type = [val.zfill(4) for val in
    #                          prod_taxon.machine_type.astype('str')]
    # prod_taxon.machine_model = [val.zfill(3) for val in
    #                           prod_taxon.machine_model.astype('str')]
    # prod_taxon['match_flag'] = 1

    df_svl_prod = pd.merge(df_svl, prod_taxon, how='left',
        left_on=[x + '_tss' for x in keys_hw], right_on=[prefix + x for x
                                                         in keys_hw])

    # Evaluate match rate
    df_svl_prod_ij = pd.merge(df_svl, prod_taxon, left_on=[x + '_tss' for x
                                                           in keys_hw],
                           right_on=[prefix + x for x in keys_hw])
    toc = time.time()
    print('Merged in {:.0f} seconds'.format(toc - tic))

    match2 = len(df_svl_prod_ij.mtm_tss.unique()) / len(df.mtm_tss.unique())

    print('{:.0f} of {:.0f} ({:.1%}) unique MTMs in the TSS data set were '
          'matched with hardware taxonomic data'.format(len(
          df_svl_prod_ij.mtm_tss.unique()), len(df.mtm_tss.unique()), match2))

    df_svl_prod.sort_index(axis=1, inplace=True)
    prep.write_fields(df_svl_prod, 'df_svl_prod', path_out)

    return df_svl_prod


def add_ePIW_to_hwma(df_hwma, path, subfolder, path_out, year=2017,
                     join_method='inner', remove_losses_=True,
                     exclude_others_=True, drop_dup_serial_=True,
                     time_sample_=False, dim2match='country'):
    """Hardware & Software maintenance (SWMA) attributes
    Source: ePricer IW personnel (Dianne Reynolds: Nikhil Airun)

    Args:
        dim2match:
        path_out:
        remove_losses_:
        drop_dup_serial_:
        time_sample_:
        join_method:
        exclude_others_:
        df_hwma:
        path:
        path_out:
        year:
        subfolder:

    Returns:

    """

    def update_report(process_log, n_current, n_prior, task_='unspecified'):

        instance_loss = n_current - n_prior
        next_task = pd.DataFrame([{'task': task_, 'instances': n_current,
                                   'gain': instance_loss}])
        log = process_log.append(next_task)

        return log, n_current


    def prep_hw(path, time_sample=time_sample_,
                sample_year=year, remove_losses=remove_losses_,
                exclude_others=exclude_others_,
                drop_dup_serial=drop_dup_serial_,
                path_cmda = path + 'hardware/csv/'):
        """

        Args:
            sample_year: year of quote approval to sample
            time_sample: if True, sample data where quote approval year
            matches sample_year
            path_cmda (object): path to where CMDA data are stored (temp)
            path: path to where data are stored
            remove_losses (boolean):
            exlude_others (boolean):
            drop_dup_serial (boolean):

        Returns:

        """

        files_hw_csv = os.listdir(path)
        files_hw_csv.sort()

        # Load select columns from each file into a list of DataFrames
        hw_cols = ['QT_COUNTRY', 'QT_QUOTEID', 'QT_APPROVALDATE',
                   'QT_CHANNELTYPE', 'QT_CUSTOMERNB_CMR', 'QT_ClientSegCd',
                   'QTC_CRMSECTORNAME', 'QTC_CRMINDUSTRYNAME',
                   'QT_OPPORTUNITYID', 'QT_VALUESELLER', 'COM_COMPONENTID',
                   'COM_HWPLATFORMID', 'COM_MTM', 'COMW_MTM_SERIALNO',
                   'COM_RevDivCd', 'COM_CATEGORY', 'COM_UpgMES',
                   'COM_Quantity', 'COM_LISTPRICE', 'COM_ESTIMATED_TMC',
                   'COM_QuotePrice', 'COM_DelgPriceL4', 'QTW_WIN_IND',
                   'PRODUCT_BRAND', ' SYSTEM_TYPE', 'PRODUCT_GROUP',
                   'PRODUCT_FAMILY', 'DOM_BUY_GRP_ID', 'DOM_BUY_GRP_NAME']

        # Load one file
        # df_hw_probe = pd.read_csv(path_in + subfolder4 +
        #     'TSS_trainingset_2016-17_Part1.csv', nrows=1000,
        #                           usecols=hw_cols, encoding='latin-1')

        tic = time.time()
        print('Loading ePricer data sets...')
        df_hw_set = [pd.read_csv(path + file, usecols=hw_cols,
            encoding='latin-1', low_memory=False) for file in files_hw_csv
            if file != '.DS_Store']  # dtype={'SERIAL_NUM': 'category'},

        # Remove spurious columns
        for i in range(len(df_hw_set)):
            for field in df_hw_set[i].columns:
                if df_hw_set[i][field].isnull().sum() == len(df_hw_set[i]):
                    df_hw_set[i].drop(field, axis=1, inplace=True)
        toc = time.time()
        print('Loaded in {:.0f} seconds'.format(toc - tic))

        # Check & record number of rows in DataFrame
        task = 'Load 2013-2015 part 1'
        n_prior = len(df_hw_set[0])
        process_log = pd.DataFrame(
                [{'task': task, 'instances': n_prior, 'gain': 0}])
        task = 'Load 2013-2015 part 2'
        process_log, n_prior = update_report(process_log, len(df_hw_set[
                                                                  1]), n_prior, task)
        task = 'Load 2016-2017 part 1'
        process_log, n_prior = update_report(process_log, len(df_hw_set[
                                                                  2]), n_prior, task)
        task = 'Load 2016-2017 part 2'
        process_log, n_prior = update_report(process_log, len(df_hw_set[
                                                                  3]), n_prior, task)
        task = 'Load 2016-2017 part 3'
        process_log, n_prior = update_report(process_log, len(df_hw_set[
                                                                  4]), n_prior, task)
        task = 'Concatenate HW DataFrames'
        df_hw = pd.concat(df_hw_set, axis=0)
        process_log, n_prior = update_report(process_log, len(df_hw),
                                             n_prior, task)
        tic = time.time()
        print('Formatting hardware data fields & values...')
        df_hw = prep.nospace_lowercase(df_hw, cols2rename={
            'qt_region': 'region',
            'qt_country': 'country',
            'qt_quoteid': 'quote_id',
            'qt_approvaldate': 'date_approval_hw',
            'qt_channeltype': 'chnl_type_ep',
            'qt_customernb_cmr': 'client_number_cmr',
            'qt_clientsegcd': 'client_segment',
            'qtc_crmsectorname': 'sector',
            'qtc_crmindustryname': 'industry',
            'qt_opportunityid': 'opportunity_id',
            'qt_valueseller': 'value_seller',
            'qtw_win_ind': 'win_ind',
            'com_componentid': 'component_id',
            'com_delgpricel4': 'p_delgl4_hw',
            'com_hwplatformid': 'hw_platform_id',
            'com_mtm': 'mtm',
            'comw_mtm_serialno': 'serial_number_hw',
            'com_revdivcd': 'rev_div_code',
            'com_listprice': 'p_list_hw',
            'com_estimated_tmc': 'cost_hw',
            'com_quoteprice': 'p_bid_hw',
            'com_delgpriceL4': 'p_delegated_hw',
            'system_type': 'hw_type',
            'product_brand': 'hw_brand',
            'product_group': 'hw_group',
            'product_family': 'hw_family'}, trim_values=True)
        toc = time.time()
        print('Completed in {:.0f} seconds'.format(toc - tic))

        # Encode win_ind such that both 'n' & NaN indicate a loss; then
        # choose to use either the textual or encoded version per how
        # you want to treat blanks
        df_hw.loc[df_hw.win_ind == 'y', 'win_ind_code'] = 1
        df_hw.loc[df_hw.win_ind == 'n', 'win_ind_code'] = 0
        # df_hw.win_ind_code.value_counts()
        # df_hw.win_ind_code.isnull().sum()

        if remove_losses:
            # Remove the duplicate loss records
            task = 'Retained wins'
            # Opportunities of unknown w/l status are excluded
            df_hw = df_hw[df_hw.win_ind_code==1]
            process_log, n_prior = update_report(process_log, len(df_hw),
                                                 n_prior, task)
            task = 'Dropped blank/NaN SN'
            df_hw = df_hw[df_hw.serial_number_hw != '']
            df_hw.dropna(subset=['serial_number_hw'], inplace=True)
            process_log, n_prior = update_report(process_log, len(df_hw),
                                                 n_prior, task)
        else:
            task= 'Exclude blank win_ind'
            df_hw = df_hw[df_hw.win_ind_code != np.nan]
            process_log, n_prior = update_report(process_log, len(df_hw),
                                                 n_prior, task)

        if exclude_others:
            task = 'Retain HWMA'
            df_hw = df_hw[df_hw.com_category == 'h']  # |
                           # (df_hw.com_category == 's'))]
            # Remove records without customer numbers
            # Remove records where there is a missing value in product
            # categorization column
            # Remove records that are zero priced
            # Remove records where the quoted price (PofL) <= .01
            # Remove records without customer numbers
            process_log, n_prior = update_report(process_log, len(df_hw),
                                                 n_prior, task)

        # Retain just the trailing five characters & ensure letters are
        # uppercase
        df_hw['serial5'] = df_hw.serial_number_hw.str[-5:]
        df_hw.serial5 = df_hw.serial5.apply(lambda x: x.upper())

        # Standardize machine type & model values with leading zeroes, as needed
        df_hw['machine_type'] = df_hw.mtm.str[0:4]
        # df_hw.machine_type = [val.zfill(4) for val in
        #                       df_hw.machine_type.astype('str')]
        df_hw['machine_model'] = df_hw.mtm.str[-3:]
        # df_hw.machine_model = [val.zfill(3) for val in
        #                        df_hw.machine_model.astype('str')]

        tic = time.time()
        print('Derive contract values from component values...')
        # Multiply price & cost by unit quantity, as one instance can include
        # more than one unit, and then append the group total to each instance
        df_hw['p_list_hw_row_total'] = df_hw.p_list_hw * df_hw.com_quantity
        df_hw['p_list_hw_total'] = df_hw.groupby(
            'quote_id').p_list_hw_row_total.transform(lambda x: x.sum())

        col_quote = ['quote_id', 'com_category', 'machine_type',
                     'machine_model', 'serial5']
        df_hw['p_list_hw_withingroup'] = df_hw.groupby(col_quote). \
            p_list_hw_row_total.transform('sum')

        df_hw['p_pct_list_hw'] = df_hw.p_bid_hw / df_hw.p_list_hw
        df_hw['p_delgl4_pct_list_hw'] = df_hw.p_delgl4_hw / df_hw.p_list_hw

        df_hw['p_bid_hw_row_total'] = df_hw.p_bid_hw * df_hw.com_quantity
        df_hw['p_bid_hw_total'] = df_hw.groupby(
                'quote_id').p_bid_hw_row_total.transform(lambda x: x.sum())
        df_hw['p_bid_hw_withingroup'] = df_hw.groupby(col_quote). \
            p_bid_hw_row_total.transform('sum')
        df_hw['p_pct_hw_withingroup'] = df_hw.p_bid_hw_withingroup /  \
                                        df_hw.p_list_hw_withingroup

        df_hw['p_delgl4_hw_row_total'] = df_hw.p_delgl4_hw * \
                                         df_hw.com_quantity
        df_hw['p_delgl4_hw_total'] = df_hw.groupby(
                'quote_id').p_delgl4_hw_row_total.transform(lambda x: x.sum())

        # Compute contribution of the total line item value (bid price) as
        # a percentage of the total HW value in the quote
        df_hw['p_bid_hw_contrib'] = df_hw.p_bid_hw_row_total / \
                                    df_hw.p_bid_hw_total

        # Derive scaled & normalized hardware cost metrics
        df_hw['cost_hw_row_total'] = df_hw.cost_hw * df_hw.com_quantity
        df_hw['cost_hw_total'] = df_hw.groupby(
                'quote_id').cost_hw_row_total.transform(lambda x: x.sum())
        df_hw['cost_hw_withingroup'] = df_hw.groupby(
            col_quote).cost_hw_row_total.transform('sum')
        df_hw['count_hw_withingroup'] = df_hw.groupby(
            col_quote).p_bid_hw_row_total.transform('count')
        df_hw['cost_pct_list_hw'] = df_hw.cost_hw / df_hw.p_list_hw
        df_hw['cost_pct_hw_withingroup'] = df_hw.cost_hw_withingroup / \
                                        df_hw.p_list_hw_withingroup

        df_hw['gp_hw_withingroup'] = df_hw.p_bid_hw_withingroup - \
                                  df_hw.cost_hw_withingroup
        df_hw['gp_pct_hw_withingroup'] = df_hw.gp_hw_withingroup / \
                                      df_hw.p_bid_hw_withingroup
        toc = time.time()
        print('Completed in {:.0f} seconds'.format(toc - tic))

        # Derive time elements
        # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
        # pandas convenience function works in some cases
        # datetime.strptime() is a robust method of datetime (hence
        # datetime.datetime) creates a datetime object from a string representing a
        # date & time & a corresponding format string (timezone & offset
        # functionality improved from Python 2.7 to 3.2
        date_format = '%d %m %Y'
        df_hw.date_approval_hw = df_hw.date_approval_hw.apply(lambda x:
            dt.datetime.strptime(x, date_format))
        df_hw['date_approval_year'] = df_hw.date_approval_hw.dt.year
        df_hw['date_approval_month'] = df_hw.date_approval_hw.dt.month

        # Sample a particular period of data, per year of quote approval
        if time_sample:
            df_hw = df_hw[df_hw.date_approval_hw==sample_year]

        # ! Temporary !
        # Add CMDA indicator from a separate file
        # Auxilliary data: CMDA-processed hardware deals imply TSS attached to HW sale
        cmda = pd.read_csv(path_cmda + 'hw_quote_list-PMCB.csv', usecols=[
            'quote_id', 'TSSINCLUDED'], dtype={'quote_id': 'str'})
        # cmda.replace([np.inf, -np.inf], np.nan)
        # cmda.dropna(inplace=True)
        cmda.rename(columns={'TSSINCLUDED': 'cmda'}, inplace=True)
        # cmda = prep.nospace_lowercase(cmda, fields=['cmda'])
        cmda[cmda.cmda == ' '] = np.nan
        cmda.cmda.fillna(0, inplace=True)
        cmda.loc[cmda.cmda=='N', 'cmda'] = -1
        cmda.loc[cmda.cmda=='Y', 'cmda'] = 1

        df_hw.quote_id =df_hw.quote_id.astype('str')

        # Add the CMDA indicator to the HWMA-HW data; likely just for model dev
        df_hw = pd.merge(df_hw, cmda, on='quote_id', how='left')
        # End CMDA

        # Label instances
        # Sale approved in the 3rd month of the quarter
        df_hw['date_approval_eoq'] = np.where(((df_hw.date_approval_month.
                                                astype(int) % 3) == 0), 1, 0)

        # df_hw = pd.DataFrame({'com_upgmes': ['1', '1', '3', '5', '4', 'C']})

        # HW upgraded -- 1=new, 3=cancellation, 4=MES, 5=Upg, C=RPO
        # df_hw['upgrade_mes'] = 0
        df_hw.loc[((df_hw.com_upgmes.astype('str') == '4') | (
                df_hw.com_upgmes.astype('str') == '5')), 'upgrade_mes'] = 1
        df_hw.fillna({'upgrade_mes': 0}, inplace=True)

        # Client segment is E
        # df_hw['client_segment_E'] = 1 * df_hw[df_hw.client_segment == 'e']

        # ! Unlike COPRA-HW ePricer data, current ePricer data has no
        # bundle indicator?
        # df_hw['sale_attached'] =

        if drop_dup_serial:
            task = 'Dropped dup SN'  # mach type & mod
            # Ensure consistency between how duplicates are defined and how we
            # merge the two data sets
            # But be aware that merging on the following 3 fields will result in a
            # lower match rate; for now, try merge on just serial5
            # If keeping lost opportunities, do not drop (blank) serial nos.
            if remove_losses:
                df_hw.drop_duplicates(subset=col_quote, inplace=True)
            else:
                df_hw = pd.concat(
                        [df_hw[df_hw.win_ind_code==1].drop_duplicates(
                                subset=col_quote),
                        df_hw[df_hw.win_ind_code==0]], axis=0)
            process_log, n_prior = update_report(process_log, len(df_hw),
                                                 n_prior, task)

        # Sort columns by name
        df_hw.sort_index(axis=1, inplace=True)

        # Write list of field names to a text file
        prep.write_fields(df_hw, 'hw', path_out)

        return df_hw, process_log, n_prior


    df_hw, process_log, n_prior = prep_hw(path + subfolder)
    # df_hw, process_log = prep_hw(path_in + subfolder4, True, False, False)

    # Merge hardware attributes to HWMA data
    hwma_hw = pd.merge(df_hwma, df_hw, on=['serial5'], how=join_method)

    # Derive deal size as sum of the list prices of all HWMA & HW components
    hwma_hw['p_list_hwma_hw'] = hwma_hw.p_list_hwma_total + \
                                hwma_hw.p_list_hw_total

    # Compute proportion of bid and list at quote level
    hwma_hw['p_list_hwma_pct'] = hwma_hw.p_list_hwma_total / (
            hwma_hw.p_list_hwma_total + hwma_hw.p_list_hw_total)
    hwma_hw['p_bid_hwma_pct'] = hwma_hw.p_bid_hwma_total / (
            hwma_hw.p_bid_hwma_total + hwma_hw.p_bid_hw_total)

    # Check how many TSS HWMA instances can be matched to HW attributes via
    # hardware serial number
    if join_method != 'inner':
        hwma_hw_inner = pd.merge(df_hwma, df_hw, on=['serial5'])
    else:
        hwma_hw_inner = hwma_hw

    match1 = len(hwma_hw_inner) / len(df_hwma)
    match2 = len(hwma_hw_inner.serial5.unique()) / len(
            df_hwma.serial5.unique())

    # Remove redundant fields and trip off suffix appended by merge
    # List columns with names ending in _x (generated during merge)
    original_cols = [x for x in hwma_hw.columns if x.endswith('_x')]
    for col in original_cols:
        # use the duplicate column to fill the NaN's of the original column
        duplicate = col.replace('_x', '_y')
        # hwma_hw[col].fillna(hwma_hw[duplicate], inplace=True)

        # drop the duplicate
        hwma_hw.drop(duplicate, axis=1, inplace=True)

        # rename the original to remove the '_x'
        hwma_hw.rename(columns={col: col.replace('_x', '')}, inplace=True)

    # Check how many records matched
    task = 'HWMA <> HW matched'
    process_log, n_prior = update_report(process_log, len(hwma_hw_inner),
                                         n_prior, task)

    # Check how many HW records match 2017 HWMA contract components
    task = 'HWMA <> HW matched (2017)'
    n6 = len(pd.merge(df_hwma[
                          df_hwma.date_comp_start.dt.year >=
                          year], df_hw, on=['serial5']))
    process_log, _ = update_report(process_log, n6, n_prior, task)

    # Determine record match rate by dimension
    dim_list = []; m1 = []; m2 = []; m3 = []
    for dim in df_hwma[dim2match].unique():
        dim_list += [dim]
        m1_ = len(df_hwma[df_hwma[dim2match] == dim])
        m1 += [m1_]
        m2_ = len(pd.merge(df_hwma[df_hwma[dim2match] == dim][
            ['country', 'serial5']], df_hw[df_hw[dim2match] == dim][[
            'country', 'serial5']], on=['serial5']))
        m2 += [m2_]
        m3 += [round(m2_ / m1_, 2)]
    cntry_match = list(zip(dim_list, m1, m2, m3))

    process_log.index = np.arange(len(process_log))

    # Print processing log
    print('HWMA-HW preparation summary')
    print(process_log[['task', 'instances', 'gain']])
    print(
        '{:.0f} HWMA records ({:.1%}, {:.1%}) matched with HW attributes '
        'by SN'.format(len(hwma_hw_inner), match1, match2))

    hwma_hw.sort_index(axis=1, inplace=True)

    # Write list of field names to a text file
    prep.write_fields(df_hw, 'hw', path_out)

    process_log.to_excel(path_out + 'prep_log_hwma+hw.xlsx', index=True)

    return hwma_hw, df_hw, process_log, cntry_match


def log_transform(df, features, base=10):
    """
    Calculates logarithm of values in the features provided
    Args:
        df (DataFrame): input DataFrame
        features (list): fields in which to transform values
        base: choose either 10 or not 10, which will result in natural log

    Returns:
        Input DataFrame with log-transformed features as additional columns

    """

    # +1 ensures log(0 + 1) = 0, instead of log(0) = -infinity
    for feature in features:
        if base == 10:
            df[feature + '_log'] = np.log10(df[feature] + 1)
        else:
            df[feature + '_ln'] = np.log(df[feature] + 1)
    return df


def scale_data(df, features=None):
    """
    Using methods of scikit-learn's preprocessing class, standardizes
    values by subtracting the mean and dividing by the standard deviation,
    with 'robust' treatment of outliers (see documentation)

    Args:
        df (DataFrame): data set containing the fields to scale
        features (list): specific fields to scale; if None, scales any filed
        containing float values

    Returns:
        DataFrame with scaled values (original values overwritten)

    """
    # Standardize numeric (float) attributes to zero mean and a standard
    # deviation of 1
    df[df.select_dtypes(include=['float64'], exclude=['object', 'int64']).
        columns] = RobustScaler().fit_transform(
            df[df.select_dtypes(include=['float64'],
                              exclude=['object', 'int64']).columns])
    return df


# ----------------------------------------------------------------------------
# Prepare IBM Systems hardware maintenance contract data
# Identify the data file and the subfolder in which it is stored
subfolder1 = 'hwma/INV_EP_CONTRACT_TYPE_v5_20180302_2013FWDinstalls.xlsx'

# Call function to load & process the hardware maintenance data
hwma, process_log = prep_hwma(path_in + subfolder1, path_out)

# Save hardware maintenance data set & record a list of fields therein
hwma.to_csv(path_out+'hwma.csv', encoding='utf-8', index=False)
hwma.to_excel(path_out+'hwma.xlsx', index=False)
process_log.to_excel(path_out+'process_log_hwma.xlsx', index=False)

# Quality control: Hardware maintenance data
# Load preprocessed data, if necessary
hwma = pd.read_csv(path_out + 'hwma.csv')

# Count how many instances in which sale_attached = sale_aftermarket
same_label = len(hwma[(hwma.sale_attached == hwma.sale_aftermarket)])
both_positive = len(hwma[(hwma.sale_attached==1) & (
        hwma.sale_aftermarket==1)])
both_negative = len(hwma[(hwma.sale_attached==0) & (
        hwma.sale_aftermarket==0)])
print('Exclusivity violation, attachd & aftermrkt sales both '
      'positively or negatively labeled: {:,.0f} ({:.1%})'.format(
      same_label, same_label / len(hwma)))

print('Exclusivity violation, attachd & aftermrkt sales positively labeled'
      ': {:,.0f} ({:.1%} of identically labeled instances)'.format(
        both_positive, both_positive / same_label))

print('Exclusivity violation, attachd & aftermrkt sales negatively labeled'
      ': {:,.0f} ({:.1%} of identically labeled instances)'.format(
        both_negative, both_negative / same_label))

print('2017 installations labeled TSS-attached: {:,.0f} ({:.1%})'.format(
      len(hwma[(hwma.sale_attached==1) & (hwma.date_inst.dt.year==2017)]),
      len(hwma[(hwma.sale_attached==1) & (hwma.date_inst.dt.year==2017)]) /
          len(hwma[hwma.date_inst.dt.year==2017])))

print('WSUs labeled TSS-attached: {:,.0f} ({:.1%})'.format(
      len(hwma[(hwma.sale_attached==1) & (hwma.tss_type=='wsu')]),
      len(hwma[(hwma.sale_attached==1) & (hwma.tss_type=='wsu')]) /
          len(hwma[hwma.tss_type=='wsu'])))

# ----------------------------------------------------------------------------
# Augment hardware maintenance data with TSS service level & Systems product
# identifiers
# Identify the file containing the TSS service levels
file_tss_svc = 'tss_service_levels.xlsx'  # 'TSS_service_lvl_map_v6.xlsx'

# Prescreen of new TSS service level detail table
svl_attribs_cols = ['SLC', 'COV VALUE', 'CNT VALUE', 'CNT UNIT', 'ORT VALUE',
                    'ORT UNIT', 'PAT VALUE', 'PAT UNIT', 'FXT VALUE',
                    'FXT UNIT', 'TAT VALUE', 'TAT UNIT']

# Identify the file containing the Systems product identifiers
# file_prod_map = 'Product hierarchy_Power_Storage_v8.xlsx'
file_hw_taxon = 'EACM_product_map.xlsx'

# Call function to load, process, and merge TSS service level & Systems
# product identifiers to the HWMA data
hwma_svl_hw_taxon = map_svl_hw_hier(hwma, file_tss_svc, file_hw_taxon,
    fields_tss=svl_attribs_cols,
    fields_hw=['mtm', 'level_2', 'level_3', 'level_4'],
    keys_tss=['sl_code'], keys_hw=['mtm'],
    path_in=path_root + '/data/', path_out=path_out)

# Save the merged data set
hwma_svl_hw_taxon.to_csv(path_out + 'hwma.csv', index=False)
# hwma_svl_hw_map = pd.read_csv(path_out + 'hwma_svl_hw_map.csv')

# Inspection
features=['sl_contact_hours', 'sl_onsite_hours', 'sl_part_hours', 'sl_hours_x_days']
len(hwma_svl_hw_taxon)
hwma_svl_hw_taxon[features][0:5]
hwma_svl_hw_taxon[features].isnull().sum()

# ----------------------------------------------------------------------------
# Augment hardware maintenance data with IBM Systems product attributes
# Identify subfolders containing data files
subfolder2 = 'hardware/csv/'
# subfolder3 = 'hardware/xlsx/'
subfolder4 = 'hardware/ePricerIW/'

# Call function to load, process, and merge Systems attributes to HWMA data,
# optionally retrieving the Systems attribute portion, plus a ledger of
# instances tracked through various operations that could remove or add them
hwma_hw, df_hw, process_log2, cntry_match = add_ePIW_to_hwma(
        hwma_svl_hw_taxon, path_in, subfolder4, path_out,
        remove_losses_=False, exclude_others_=True, time_sample_=False)

# Transform feature(s) & place in a separate column (leaving original intact)
hwma_hw = log_transform(hwma_hw, ['p_list_hwma', 'p_list_hw',
                                  'p_list_hwma_hw'])
hwma_hw.sort_index(axis=1, inplace=True)

# Save the merged data set & record a list of fields therein
hwma_hw.to_csv(path_out + 'hwma_hw.csv', index=False)
prep.write_fields(hwma_hw, 'hwma_hw', path_out)
process_log2.to_excel(path_out+'process_log_hwma_hw.xlsx', index=False)

# Refresh list of annotated training data fields
complete_list = pd.read_csv(path_out + 'hwma_hw_cols.txt', header=None)
complete_list.columns=['Field']
annotated_list = pd.read_excel(path_out + 'hwma_hw_fields_explained.xlsx')
new_list = pd.merge(complete_list, annotated_list, on='Field', how='left')
new_list.to_excel(path_out+'hwma_hw_fields_explained_.xlsx', index=False)

# END TRAINING DATA PREPARATION

# ----------------------------------------------------------------------------
# For QC purposes, produce an inclusive (outer-joined) data set
hwma_hw_oj = pd.merge(hwma_svl_hw_map, df_hw, on=['serial5'],
                   how='outer')
hwma_hw_cmda_oj = pd.merge(hwma_hw_oj, cmda[['quote_id', 'cmda']],
                        on='quote_id', how='left')
hwma_hw_cmda_oj.to_csv(path_out+'hwma_hw_outer_join.csv', index=False)


# Quality control: Hardware maintenance with Systems attributes
hwma_hw = pd.read_csv(path_out + 'hwma_hw.csv', encoding='latin-1')
hwma_gbo = hwma_hw.groupby(['quote_id', '', 'machine_model',
                 ])  # 'com_category' results in predominantly just singletons
hwma_gbo_1 = hwma_gbo.agg({'p_pct_list': [np.size, 'mean', 'std']})
hwma_gbo_1.sort_index(inplace=True)

# ----------------------------------------------------------------------------
# Analysis of training data
# Quote-MTM-level variation
df_hw['discount_hw'] = 1 - df_hw.p_pct_list_hw

# Define level of aggregation
indices = ['quote_id', 'mtm', 'com_category', 'serial5']

# Define variables of interest
targets = ['discount_hw', 'p_list_hw', 'cost_hw']  # 'p_pct_list_hw'

gbo = df_hw.groupby(indices)
df_pvar = gbo[targets].agg([np.size, np.std, np.min, np.max,
                           np.mean]).\
    rename(columns={'size': 'machines', 'std': 'st_dev', 'mean': 'avg'})

df_pvar.sort_index(inplace=True)

# Calculate standard error
# How do you place with other discount_hw sub-columns?
df_pvar['discount_hw', 'st_error'] = df_pvar['discount_hw', 'st_dev'] /  \
                                   df_pvar['discount_hw', 'avg']

df_pvar.sort_index(axis=1, inplace=True)

# How can you sort by st_error?
# df_pvar.sort_values(df_pvar.xs(('discount_hw', 'st_error')), ascending=False,
#                     inplace=True)

df_pvar.drop([('cost_hw', 'machines')], axis=1, inplace=True)
df_pvar.drop([('p_list_hw', 'machines')], axis=1, inplace=True)

# df_pvar.to_excel(path_out+'pvar_draft.xlsx')
df_pvar[(df_pvar['discount_hw', 'machines'] > 1.) &
        (df_pvar['discount_hw', 'amin'] > 0)].to_excel(path_out +
                                                    'p_variation_hw.xlsx')

# gbo.agg({target: np.size, target: np.std})

df_pvar.machines.sum()
len(df_pvar[df_pvar.std_error != 0]) / len(df_pvar)

df_hw.set_index(indices, inplace=True)
df_hw.sort_index(inplace=True)
len(df_hw[df_hw.duplicated(subset=['serial5'])].serial5)
df_hw_h = df_hw.xs('h', level=2)

len(df_hw_h.duplicated(subset=['serial5']))

# ----------------------------------------------------------------------------
df_hw.reset_index(inplace=True)
indices = ['quote_id', 'mtm', 'com_category']

len(df_hw)
df_hw.drop_duplicates(subset=['quote_id', 'mtm', 'com_category',
                              'serial5'], inplace=True)
df_hw.set_index(indices, inplace=True)
hwsn = df_hw.serial5

hwu = df_hw.unstack('com_category')

hwu_ = hwu[hwu['m', 'serial5']]

len(df_hw[(df_hw.com_category=='h') &
          ((df_hw.serial5 == '') |
            df_hw.serial5.isna())])

# ----------------------------------------------------------------------------
# Attach rates according to CMDA
cntry = 'france'
yr = 2017
df_hws = df_hw[(df_hw.country == cntry) & (df_hw.date_approval_year == yr)]
m1 = len(df_hws[df_hws.cmda != np.nan])
m1b = np.size(df_hws[df_hws.cmda != np.nan].quote_id.unique())

m2 = len(df_hws[df_hws.cmda == 1])
m2b = np.size(df_hws[df_hws.cmda == 1].quote_id.unique())

m3 = len(df_hws[df_hws.cmda == -1])
m3b = np.size(df_hws[df_hws.cmda == -1].quote_id.unique())

# Are these lengths correct, or is there multiple counting
m2 / m1 * 100
(m2 + m3) / m1 * 100

m2b / m1b * 100
(m2b + m3b) / m1b * 100

# ----------------------------------------------------------------------------
# Waterfall visualization of data volume as the set is processed
# The approach shifts the work into structuring the data rather than
# plotting bars one by one in a for loop
def roundrobin(*iterables):
    """
    Interleaves strings; e.g., ('ABC', 'D', 'EF') --> A D E B F C

    Args:
        *iterables: arrays or lists

    Returns:
        Generator (use list() to objectify and view)

    """
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))

# Load the log of data processing
df = pd.read_excel(path_out + 'prep_log_hwma+hw.xlsx')

# Sample a portion, if necessary
cdata = df[5:-1].reset_index()

# Derive numpy arrays from DataFrame columns
v1 = cdata.instances.values
v2 = cdata.gain.values

# Interleave (intercalate) the two arrays, forming one array of alternating
#  number of data instances, change in instance count,...
interleaved = list(roundrobin(v1, v2[1:]))

# Form an array of values to feed into Matplotlib's bar function as
# baseline values, consisting of a zero, alternating with the previous
# instance count
baselines = list(roundrobin(round(len(cdata)/2)*[0], v1[:-1]))

# importlib.reload(myplt)

myplt.waterfall(interleaved, baselines, path_out, scale=1,
                frmt='{:,.0f}', transp_=0.9,
                filename='data_prep_waterfall_ePIW')
plt.show()


# ----------------------------------------------------------------------------
# Attached sale attribution: Compare engineering logic to CMDA indicator
# Check how many deals were & weren't processed via CMDA (remainder are NaN)
len(hwma_hw_cmda[hwma_hw_cmda.cmda=='Y'])  # 3,863
len(hwma_hw_cmda[hwma_hw_cmda.cmda=='N'])  # 5,727

# Check how many deals occurred in each year of record (2,417 installed 2017)
hwma_hw_cmda[hwma_hw_cmda.cmda=='Y'].date_inst.dt.year.value_counts()

blogic = hwma_hw_cmda[['date_inst', 'date_approval_hw', 'date_contr_start',
                       'date_srv_start', 'sale_attached',
                       'sale_aftermarket', 'cmda']]

# blogic.loc[: 'cmda'] = np.where(blogic.cmda=='Y', 1, 0)
# blogic.loc[: 'cmda'] = np.where(blogic.cmda==np.nan)
# blogic[blogic.cmda=='Y'] = 1
# blogic[blogic.cmda=='N'] = 0

# Dates are not formatted properly
time_format = '%Y-%m-%d'
blogic['date_inst'] = blogic.date_inst.astype('str').apply(lambda x:
    dt.datetime.strptime(x, time_format))

# Approval dates: 2016-09-06T00:00:00.000000000 (category)
# Convert to strings & remove T00...
blogic['date_approval_hw'] = blogic.date_approval_hw.astype('str').apply(
        lambda x: x.strip('T00:00:00.000000000'))
# Convert to datetime64[ns]
blogic['date_approval_hw'] = blogic.date_approval_hw.apply(lambda x:
    dt.datetime.strptime(x, time_format))

blogic['date_contr_start'] = blogic.date_contr_start.astype('str').apply(
        lambda x: dt.datetime.strptime(x, time_format))

# TypeError: incompatible type [datetime64[ns]] for a datetime/timedelta
# operation
blogic['lag1'] = blogic.date_inst - blogic.date_approval_hw
blogic['lag2'] = blogic.date_inst - blogic.date_contr_start
blogic['lag3'] = blogic.date_contr_start - blogic.date_approval_hw
blogic['lag4'] = blogic.date_inst - blogic.date_srv_start

# Retain just days component in new fields
blogic.loc[:, 'lag1_'] = blogic.lag1.astype('timedelta64[D]')
blogic.loc[:, 'lag2_'] = blogic.lag2.astype('timedelta64[D]')
blogic.loc[:, 'lag3_'] = blogic.lag3.astype('timedelta64[D]')
blogic.loc[:, 'lag4_'] = blogic.lag4.astype('timedelta64[D]')

myplt.histogram(blogic[blogic.cmda=='Y'].lag1_, path_out, nbins=20,
                title1='Contract approval to installation', ylabel1='Count',
                xlabel1='Lag (months)', scale=30, y_rot=90,
                filename='cmda_lag1')

myplt.histogram(blogic[blogic.cmda=='Y'].lag2_, path_out, nbins=20,
                title1='Contract start to installation', ylabel1='Count',
                xlabel1='Lag (months)', scale=30, y_rot=90,
                filename='cmda_lag2')

myplt.histogram(blogic[blogic.cmda=='Y'].lag3_, path_out, nbins=20,
                title1='Contract approval to start', ylabel1='Count',
                xlabel1='Lag (months)', scale=30, y_rot=90,
                filename='cmda_lag3')

bundle_label = 'N'
cdata1n = blogic[(blogic.cmda==bundle_label) &
                (blogic.lag1_ > -366) & (blogic.lag1_ < 366) &
                (blogic.date_inst.dt.year == 2017)]
cdata2n = blogic[(blogic.cmda==bundle_label) &
                (blogic.lag2_ > -366) & (blogic.lag2_ < 366) &
                (blogic.date_inst.dt.year == 2017)]
cdata3n = blogic[(blogic.cmda==bundle_label) &
                (blogic.lag3_ > -366) & (blogic.lag3_ < 366) &
                (blogic.date_inst.dt.year == 2017)]
cdata4n = blogic[(blogic.cmda==bundle_label) &
                (blogic.lag4_ > -366) & (blogic.lag4_ < 366) &
                (blogic.date_inst.dt.year == 2017)]

myplt.histogram(cdata1.lag1_, cdata1n.lag1_, path_out, nbins=20,
    title1='Contract/quote approval to installation',
    ylabel1='Count', xlabel1='Lag (months)', scale=30, y_rot=90,
    filename='cmda_lag1r')
myplt.histogram(cdata2.lag2_, cdata2n.lag2_, path_out, nbins=20,
    title1='Contract start to installation',
    ylabel1='Count', xlabel1='Lag (months)', scale=30, y_rot=90,
    filename='cmda_lag2r')
myplt.histogram(cdata3.lag3_, cdata3n.lag3_, path_out, nbins=20,
    title1='Contract/quote approval to start',
    ylabel1='Count', xlabel1='Lag (months)', scale=30, y_rot=90,
    filename='cmda_lag3r')
myplt.histogram(cdata4.lag4_, cdata4n.lag4_, path_out, nbins=20,
    title1='HWMA service start to installation',
    ylabel1='Count', xlabel1='Lag (months)', scale=30, y_rot=90,
    filename='cmda_lag4r')

blogic[blogic.cmda=='Y'].lag1.astype('timedelta64[D]').hist()
plt.title('Lag: Contract approval to installation')
plt.show()


# ----------------------------------------------------------------------------
# Win rate modeling
# Load model output
hwma_m1 = pd.read_csv(path_out + 'in_sample1.csv', encoding='latin-1')

# Synthesize random Gaussian distribution of data
# Specify mean and standard deviation
mu = 0.8
sigma = mu / 20
y_sim = sigma * np.random.randn(10000) + mu

# Convert numpy array to pandas DataFrame
y_sim_ = pd.DataFrame(y_sim, columns=['p_pct_list'])
bins_sim, count_sim = winrate.winprob_fit(y_sim_, 'p_pct_list', path_out,
    file1='winprob_sim_pdf_1', file2='winprob_sim_cdf_1')

# All HWMA component
hwma_m1.loc[:, 'p_bid_hat'] = 10 ** hwma_m1.y_pred

series = hwma_m1.p_pct_list / hwma_m1.y_pred
importlib.reload(winrate)
bins_tss, count_tss = winrate.winprob_fit(series[series<5], path_out,
                                          xlab='p / p_hat',
    file1='winprob_hwma_pphat_pdf_3', file2='winprob_hwma_pphat_cdf_3')  #

# F2F Power
hwma_hw.set_index(['platform', 'chnl'], inplace=True)
hwma_hw.sort_index(inplace=True)
# idx = pd.MultiIndex
# dfs = hwma_hw.loc[idx['Power', 'F2F']]
dfs = hwma_hw.xs('Power').xs('F2F')
dfs2 = hwma_hw.xs('Power')

bins_tss, count_tss = winrate.winprob_fit(dfs2, 'p_pct_list', path_out,
    ylab='Win probability', file1='winprob_hwma_Power_pdf_1',
                                  file2='winprob_hwma_Power_cdf_1')


# ----------------------------------------------------------------------------
# Descriptive analysis -- Predictions
# Load model results
hwma_m1 = pd.read_excel(path_out + 'regression_results/' +
                        'lme_hwma5_report.xls', sheet_name='predictions')
mape = np.mean(np.absolute(hwma_m1.error_pct))

hwma_m1['error_pct_abs'] = np.absolute(hwma_m1.error_pct)

mape_country = hwma_m1.groupby('country').agg({'error_pct_abs': np.mean})

hwma_m1['dealsize_quintile'] = pd.qcut(hwma_m1.p_list_hwma_hw_log, 5,
                                   labels=np.arange(1,6))

mape_dealsize = hwma_m1.groupby('dealsize_quintile').agg({'error_pct_abs':
                                                            np.mean}) * 100

# Write list of field names to a text file
prep.write_fields(hwma_m1, 'hwma_m2', path_out)

# hwma_m1.rename(columns={'lncomqp': 'y_log', 'value_score': 'y_hat'},
#                inplace=True)

hwma_m1['y'] = hwma_m1.p_pct_list
hwma_m1.y_pred = 10 ** hwma_m1.y_pred

importlib.reload(myplt)


def model_output_desc(df, path_out, y='y', y_hat='y_hat', cat1='', cat2='',
                      xlab='', ylab='', scl_x=1, scl_y=1,
                      graphics_file_suffix=''):
    """

    Args:
        df: DataFrame containing model input & output
        y: historical values of target variable
        y_hat: predicted values of target variable
        cat1: category to add dimensions to visualizations
        cat2: additional category to enrich visualizations
        path_out: folder to save graphics

    Returns:
        Nothing; saves graphics

    """

    df.y_log = np.log10(df[y])
    df.y_hat_log = np.log10(df[y_hat])

    # Convert y_hat to same units as y & compute residuals
    df['residual'] = df[y_hat] - df[y]
    df['residual_log'] = np.log10(df[y_hat] / df[y])

    # y_hat vs y

    # To use a categorical variable with textual values, you must numerically
    # encode those values to map colors; this is simple when the variable
    # is binary; however, if there are more than two categories,
    # an additional column containing the sum of of the other binary
    # categoricals produced by get_dummies or other encoder needs to be used

    if len(cat1) > 0:
        dfs = pd.get_dummies(df[[y, y_hat, cat1]])
        scolor = dfs[cat1]
    else:
        scolor = '0.5'

    myplt.scat(df[y], df[y_hat], path_out, scale_x=scl_x, scale_y=scl_y,
               transp_=0.3, figwidth_=1.1 * fig_ht,
               symbolcolor=scolor, cmap_='tab20c',
               filename='m1_yhat_vs_y' + graphics_file_suffix,
               xlabel1=xlab, ylabel1=ylab, medians=True, diag=True)

    # Residuals (as a percentage of predicted value)
    # Distribution
    # cdata = df[df.residual_K < 1.1].residual_K
    myplt.histogram(df.residual / df[y_hat], path_out, barcolor1 =
                    ibm_bluehues[1], scale = 1, norm=False, logged=False,
                    nbins=20, y_rot=90,
                    filename='m2_r_pct_dist' + graphics_file_suffix,
                    xlabel1='Residual (% predicted price)', ylabel1='Count')

    # Versus y_hat, linear scale
    # myplt.scat(df[y_hat], df.residual / df[y_hat], path_out, scale_x=1e3,
    #            scale_y=1e-2, filename='m1_r_vs_yhat' + graphics_file_suffix,
    #            symbolcolor = scolor, cmap_='bwr',
    #            xlabel1=xlab, ylabel1=ylab)

    # if len(hwma_m1[cat1].unique() > 2):
    #     for i in range(len(hwma_m1[cat1].unique())):
    #         hwma_m1_s['cat1_sum'] += hwma_m1_s.iloc[:, -i]
    #
    # myplt.scat(hwma_m1_s.y, hwma_m1_s.value_score,
    #            path_out,
    #            scale_x=1e3,
    #            scale_y=1e3, filename='color_test',
    #            symbolcolor=hwma_m1_s.chnl_BP,  #  cat1_sum
    #            cmap_ = 'cool',
    #            xlabel1='Price ($K)', ylabel1='Predicted price')
    # Versus y_hat, log scale
    # myplt.scat(df.y_hat_log, df.residual_log, path_out,
    #            filename='m1_r_vs_yhat_log',
    #            xlabel1='Predicted price (log $)', ylabel1='Residual (log $)')

# Call function to perform a prescribed set of descriptive analyses of the
# predictive model output
xlab = 'Predicted price ($K)'
ylab = 'Residual (% predicted price)'

model_output_desc(hwma_m1, path_out, y='p_pct_list_hwma', y_hat='y_pred',
                  cat1='',
                  xlab='Price (% list)', ylab = 'Predicted price (% list)',
                  graphics_file_suffix='_99')

# View first several rows of select columns
hwma_m1[['y', 'y_hat', 'value_score']][0:10]

# Compare (scaled) coefficients
features = ['lnlp', 'lnhwlp', 'lnds', 'p_uplift_comm', 'weekly_hours',
            'response_time_lvl1', 'response_time_lvl2', 'response_type',
            'comp_duration_days', 'committed', 'prod_div',
            'product_category', 'level_2', 'sale_hwma_attached',
            'sale_hwma_aftermarket', 'auto_renewal_flg', 'channel_id',
            'ff_chnl_sdesc', 'offering_short_desc', 'reg_cd', 'country',
            'cmda', 'mcc', 'sector_sdesc', 'pymnt_opt', 'srv_dlvr_meth',
            'com_category']

hwma_m1[features].describe().to_excel(path_out + 'feature_stats.xlsx')

feature_variance = pd.DataFrame(columns=['feature', 'variance'])

for feature in features:

    row = np.std(hwma_m1[feature])
    feature_variance.append(row)

lme_params = pd.read_excel(path_out + 'lme_params_20.xls',
                           names=['coefficient'])

lme_params_ = lme_params[features]

coeff_eval = pd.merge(lme_params_, hwma_m1[['']], left_index='on',
                      right_index='on')

# Ad-hoc analysis & testing
# Generate miniaturized data sets for testing
dfs2 = hwma_hw.sample(frac=.1)
# idx = pd.IndexSlice[:]
# dfs.set_index(['chnl', 'country'], inplace=True)
# dfs.sort_index(inplace=True)
dfs.loc[idx].p_pct_list
# upstr.distributions(dfs, ['p_pct_list', 'discount_hwma'], indices=[
# 'country',
#     'chnl'], xsections=[('UNITED KINGDOM', 'BP'), ('SWITZERLAND', 'F2F')])

# Proportion of attached TSS deals
len(hwma_hw[hwma_hw.sale_hwma_attached==1])/len(hwma_hw)  # ! aggregate to contract number?

# Distribution of TSS prices -- attached vs aftermarket
hwma_hw[hwma_hw.sale_hwma_attached==1]) / len(hwma_hw)

features_desc = ['p_pct_list', 'contract_real_value']

# itertools could construct lists of xsections
xs = [('Storage', 'FRANCE', 'BP'), ('Storage', 'BELGIUM', 'BP'),
      ('Power', 'FRANCE', 'BP'), ('Power', 'BELGIUM', 'BP')]
xs2 = [('Power'), ('Storage')]
xs3=[(0), (1)]

# Single attribute, entire data set
upstr.distributions(hwma_hw, ['p_pct_list'])

# Multiple attributes, entire data set
# upstr.distributions(hwma_hw, features_desc)

# Multiple attributes, one or more cross sections
upstr.distributions(hwma_hw, ['discount_hwma'], indices=[
    'sale_hwma_attached'],
              xsections=xs3, cumul=True)  # , 'country', 'chnl']

x = np.log10(dfs[features_cluster[6]])
y = dfs.p_pct_list
cdata = hwma_hw[(hwma_hw[features_cluster[6]] >= 1) &
                (hwma_hw.platform == 'Storage')]
x = np.log10(cdata[features_cluster[6]])
y = cdata.p_pct_list

myplt.scat(x, y, path_out + 'graphs/', scale, title1='',
           symbolcolor=cdata.sale_hwma_attached, transp_=0.3,
           xlabel1='Contract value (log USD)',
           ylabel1='Component price (% list)',
           filename='hwma_Storage2')  # symbolsize=cdata.comp_duration_days/10,


myplt.hexbin(x, y, path_out + 'graphs/', filename='hwma_Power', grid=20,
             xlabel1='Contract value (log USD)', title1='Power',
             ylabel1='Component price (% list)',
             cblabel='', colormap='Greys')

# ----------------------------------------------------------------------------
# Clustering
features1 = ['auto_renewal_flg', 'chnl', 'committed',
             'comp_duration_days', 'comp_sched_ind',
             'contract_list_value', 'contract_real_value',
             'p_list', 'hours_of_cover_as_is',
             'hours_of_cover_bycao', 'hours_of_cover_bytss',
             'level_2_byhw', 'machine_model_x', 'machine_type_x',
             'offer_nm', 'response_time_lvl1_bytss',
             'response_time_lvl2_bytss', 'sale_hwma_attached']

features2 = ['chnl', 'comp_duration_days', 'contract_real_value',
             'sale_hwma_attached']

features3 = ['contract_list_value', 'contract_real_value',
             'p_list', 'comp_duration_days']

fields_services = {'service_level': ['sl_fix_time', 'sl_onsite',
                                     'sl_part_time', 'committed',
                                     'coverage_hours_days'],
                    'contract': ['offer_nm'],
                    'misc': []}

    # 'sl_sdesc', 'srv_dlvr_meth',

dfs.set_index(['platform', 'prod_div', 'machine_type_x'], inplace=True)
dfs.sort_index(inplace=True)

importlib.reload(upstr)
dfc, train_X, _, _ = upstr.cluster(hwma_hw, features2, levels=20,
clusters=8,
    draw_dendrogram=True, title1='Hierarchical clustering',
    xlabel1='Dissimilarity', ylabel1='Component index or (merged count)',
    filename='whole', path_out=path_out + '/graphs/')


# Add the cluster label to the DataFrame index
hwma_hw_c.set_index('cluster', append=True, inplace=True)

# List indices by which to determine the heterogeneity of clusters
indices1 = ['cluster', 'platform', 'prod_div', 'machine_type',
            'machine_model']

# Describe each cluster
# Average
hwma_hw_c.groupby(level='cluster').agg({'p_pct_list': np.mean,
                                        'p_list': np.mean,
                                        'machine_type': 'nunique'})

hwma_hw_c.groupby(level=indices1)

# Heterogeneity -- define?

# To plot a dendrogram, load an array of the data required
file_array = 'ward_cluster_children.npy'
children = np.load(path_out + file_array)
upstr.plot_dendrogram_external(children, path_out,
    filename='agglom_cluster_mod4features')

# ----------------------------------------------------------------------------
# Simple linear regression
# Prod_hierarchy: platform, prod_div, machine_type
# price_discount
# Possible attributes: p_list, mmmc_mach_usd, total_comp_real,
# total_comp_list,
# total_comp_mmmc, p_list
    #, total_contr_list, total_contr_mmmc, total_contr_real
    #, term_calc_bymonth, chnl, reg_cd, offer_nm, committed_bycao, response_type_bycao, hours_of_cover_bycao, response_time_bycao
# Do need additionals: offer_nm, service_lvl
# Other possibles: auto_renewal_flg, pymnt_opt

md = smf.ols("price_discount ~ total_comp_mmmc + term_calc_bymonth + "
"prod_div + reg_cd + chnl + offer_nm + committed_bycao + hours_of_cover_bycao + response_time_bycao", df_hwma)
mdf = md.fit()
print(mdf.summary())
