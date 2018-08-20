# IBM Systems hardware - GTS-Tech Support Services pricing
# Aaron Slowey, Gloria Zhang, Jason Zhu, IBM Chief Analytics Office, 2018

import os
import numpy as np
import pandas as pd
import datetime as dt
import time as time
from sklearn.preprocessing import RobustScaler
from utils import load_data  # , upstream as upstr


def to_category(df):
    """
    To reduce the size of a DataFrame & streamline operations thereon,
        converts non-numeric (object dtype) to categorical (category dtype)

    Args:
        df: DataFrame containing fields of strings (object dtype)

    Returns:
        DataFrame with categorical values in place of object values
    """
    df[df.select_dtypes(include=['object'], exclude=['float64', 'int64']
        ).columns] = df.select_dtypes(include=['object'], exclude=[
        'float64', 'int64']).apply(lambda x: x.astype('category'))
    return df


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
            df[feature + '_log'] = np.log10(df[feature] + 2)
        else:
            df[feature + '_ln'] = np.log(df[feature] + 2)
        df[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def make_a_date(df, fields):
    """
    Converts values to Pandas DateTime format

    Args:
        df: DataFrame containing the fields to be reformatted
        fields: list of fields to reformat

    Returns:
        Reformatted DataFrame
    """
    for field in fields:
        df[field] = pd.to_datetime(df[field], errors='coerce')  # format='%Y/%m/%d',
    return df


def map_svl(df, file_svc, fields_tss, keys_tss, path_in, path_out):
    """
    Maps TSS service level attributes, which are or were not included in
    the primary data set

    Args:
        df: Larger DataFrame to which to append the attributes
        file_svc (str): file containing the TSS service level attributes
        fields_tss (list):  select TSS service level attributes
        keys_tss (list): merge (join) key(s) for TSS service level attributes
        path_in (str): path where TSS service level attribute file is stored
        path_out (str): path where IBM Systems product taxonomy file is stored

    Returns:
        Input DataFrame with TSS service level attributes

    """
    # Defer to TSS client's interpretation of portions of sl_desc
    tic = time.time()
    print('Loading & processing TSS service level attributes...')
    service_attribs = pd.read_excel(path_in + file_svc, sheet_name='EMEA',
                                na_values='-')[fields_tss].drop_duplicates()

    # Ensure consistency with keys prior to merging by reformatting
    service_attribs = nospace_lowercase(service_attribs,
                                        cols2rename={'slc': 'sl_code','cov_value': 'sl_hours_days'},
                                        trim_values=True, categorize=False)
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

    # Compute product of hours x days; e.g., 11 x 5 = 55 hours
    service_attribs[['sl_cov_hours', 'sl_cov_days']] = \
        service_attribs.sl_hours_days.str.split('x', expand=True)
    service_attribs['sl_hours_x_days'] = service_attribs.sl_cov_hours.astype(
            'float64') * service_attribs.sl_cov_days.astype('float64')

    # Map to primary TSS data
    # Ensure the key of the primary TSS data set is a string, not categorical
    df.sl_code = df.sl_code.astype('str')

    svl_fields = ['sl_code', 'sl_hours_x_days', 'sl_contact_hours', 'sl_onsite_hours', 'sl_part_hours']
    svl = pd.merge(df, service_attribs[svl_fields], on=keys_tss, how='left')
    toc = time.time()
    print('Merged in {:.0f} seconds'.format(toc - tic))

    # Evaluate match rate
    match1 = len(svl[keys_tss[0]].unique()) / len(df[keys_tss[0]].unique())

    print('{:.0f} of {:.0f} ({:.1%}) TSS unique SLCs matched with '
          'TSS service level attributes'.format(len(svl[keys_tss[0]].unique()),
                                                len(df[keys_tss[0]].unique()),
                                                match1))

    # Derive year of contract start from the start date
    svl['contr_start_year'] = svl.date_contr_start.dt.year

    return svl, service_attribs


def map_hw_taxon(df, file_taxon, fields_hw, keys, path_in, path_out):
    """
    Maps IBM Systems product taxonomy; TSS service level attributes are or
    were not included in the primary data set

    Args:
        df: Larger DataFrame to which to append the attributes
        file_taxon (str): file containing the IBM Systems product taxonomy
        fields_hw (list): select fields in the Systems taxonomy
        keys (list): merge (join) key(s) for Systems taxonomy
        path_in (str): path where TSS service level attribute file is stored
        path_out (str): path where IBM Systems product taxonomy file is stored

    Returns:
        Input DataFrame with Systems taxonomy fields mapped

    """
    tic = time.time()
    print('Loading & processing hardware taxonomy...')
    # Map product taxonomy
    prod_taxon = pd.read_excel(path_in + file_taxon).drop_duplicates()
    prod_taxon = nospace_lowercase(prod_taxon, cols2rename={
        'level_0': keys[0]}, trim_values=True)
    prod_taxon = prod_taxon[fields_hw]

    prefix = 'taxon_hw_'
    prod_taxon.columns = [prefix + x for x in fields_hw]

    print("Length of DF:", len(df))
    df_taxon = pd.merge(df, prod_taxon, how='left', left_on=keys, right_on=[prefix + x for x in keys])

    # Evaluate match rate
    print("Length of DF_taxon:", len(df_taxon))
    df_taxon_ij = pd.merge(df, prod_taxon, left_on=keys, right_on=[prefix + x for x in keys])
    toc = time.time()
    print('Merged in {:.0f} seconds'.format(toc - tic))

    match2 = len(df_taxon_ij.mtm.unique()) / len(df.mtm.unique())

    print('{:.0f} of {:.0f} ({:.1%}) unique MTMs in the TSS data set were '
          'matched with hardware taxonomic data'.format(len(
          df_taxon_ij.mtm.unique()), len(df.mtm.unique()), match2))

    df_taxon.sort_index(axis=1, inplace=True)
    write_fields(df_taxon, 'hwma', path_out)

    return df_taxon, prod_taxon


def map_cmda(df, path, label_):
    """
    Add CMDA indicator from a separate file.
    CMDA-processed hardware deals imply TSS attached to HW sale
    Note that quote id's could have leading zeroes and consist
    entirely of numbers; hence, import as a string and ensure the
    values are seven characters long, adding leading zeroes as needed

    Args:
        df:
        path:

    Returns:

    """

    cmda = load_data(path, label=label_, usecols_=['quote_id', 'TSSINCLUDED'])

    cmda.quote_id = [val.zfill(7) for val in cmda.quote_id.astype('str')]
    cmda.rename(columns={'tssincluded': 'cmda'}, inplace=True)
    # cmda = nospace_lowercase(cmda, fields=['cmda'])
    cmda[cmda.cmda == ' '] = np.nan
    cmda.cmda.fillna(0, inplace=True)
    cmda.loc[cmda.cmda == 'N', 'cmda'] = -1
    cmda.loc[cmda.cmda == 'Y', 'cmda'] = 1

    # Leaving the trinomial cmda indicator in tact for tracking purposes,
    # derive a bundle indicator
    cmda.loc[(cmda.cmda == -1) | (cmda.cmda == 1), 'bundled'] = 1
    cmda.fillna({'bundled': 0}, inplace=True)

    df.quote_id = [val.zfill(7) for val in df.quote_id.astype('str')]
    # Add the CMDA indicator to the HWMA-HW data; likely just for model dev
    print('CMDA Before:', len(df))
    df = pd.merge(df, cmda, on='quote_id', how='left')
    print('CMDA After: ', len(df))
    return df


def nospace_lowercase(df, trim_values=False, cols2rename={}, fields=[], categorize=True):
    """
    Remove whitespace from & lowercase-format column names & alphabetically
    arrange columns. Optionally rename columns. Optionally remove
    whitespace, replace dashes with spaces, and convert to lowercase values in
    selected or all non-numeric fields. Lastly, convert object dtype to
    category dtype.

    Args:
        df: Input DataFrame
        trim_values: Boolean on whether to trim values of fields
        fields: fields to trim the values of; if left empty, function will
        trim all non-numeric fields
        cols2rename: Dictionary of old & new column names

    Returns:
        Processed DataFrame
    """
    # Remove leading and trailing whitespace
    df.columns = df.columns.str.lstrip()
    df.columns = df.columns.str.rstrip()

    # Replace spaces in field names with underscores
    df.columns = df.columns.str.replace(' ', '_')

    # Change field name format to lowercase, rename & alphabetize

    df.columns = map(str.lower, df)

    df.rename(columns=cols2rename, inplace=True)

    df.sort_index(axis=1, inplace=True)

    # Remove leading and trailing whitespace
    # Select fields (columns)
    if trim_values:
        if len(fields) > 0:
            for field in fields:
                df[field] = df[field].str.strip().str.lower().str.replace(
                        '-', ' ')
        else:
            # Change field name format to lowercase & reorder alphabetically
            df[df.select_dtypes(include=['object'], exclude=['float64', 'int64']).
                columns] = df.select_dtypes(include=['object'], exclude=['float64',
                'int64']).apply(lambda x: x.str.strip().str.lower().str.
                                replace('-', ' '))

    if categorize:
        df = to_category(df)

    return df


def merge_tss_hw(df_tss, df_hw, path_out, process_log=pd.DataFrame(),
                 year=2016, join_method='inner', dim2match='country'):
    """
    Args:
        df_tss:
        df_hw:
        process_log:
        path_out:
        year:
        join_method:
        dim2match:

    Returns:
    """

    df_hw.drop('mtm', axis=1, inplace=True)

    # Merge hardware attributes to HWMA data
    tss_hw = pd.merge(df_tss, df_hw, on=['serial5'], how=join_method)

    # Derive deal size as sum of the list prices of all HWMA & HW components
    tss_hw['p_list_hwma_hw'] = tss_hw.p_list_hwma_total + tss_hw.p_list_hw_total

    # Compute proportion of bid and list at quote level
    tss_hw['p_list_hwma_pct'] = tss_hw.p_list_hwma_total / (
            tss_hw.p_list_hwma_total + tss_hw.p_list_hw_total)
    tss_hw['p_bid_hwma_pct'] = tss_hw.p_bid_hwma_total / (
            tss_hw.p_bid_hwma_total + tss_hw.p_bid_hw_total)

    # Check how many TSS HWMA instances can be matched to HW attributes via
    # hardware serial number
    if join_method != 'inner':
        tss_hw_ij = pd.merge(df_tss, df_hw, on=['serial5'])
    else:
        tss_hw_ij = tss_hw

    match1 = len(tss_hw_ij) / len(df_tss)
    match2 = len(tss_hw_ij.serial5.unique()) / len(df_tss.serial5.unique())

    # Remove redundant fields and trip off suffix appended by merge
    # List columns with names ending in _x (generated during merge)
    original_cols = [x for x in tss_hw.columns if x.endswith('_x')]
    for col in original_cols:
        # use the duplicate column to fill the NaN's of the original column
        duplicate = col.replace('_x', '_y')

        # drop the duplicate
        tss_hw.drop(duplicate, axis=1, inplace=True)

        # rename the original to remove the '_x'
        tss_hw.rename(columns={col: col.replace('_x', '')}, inplace=True)

    # Check how many records matched
    task = 'HWMA <> HW matched'
    n_prior = len(tss_hw)
    process_log, n_prior = update_report(process_log, len(tss_hw_ij),
                                         n_prior, task)

    # Check how many HW records match 2017 HWMA contract components
    task = 'HWMA <> HW matched (2016)'
    n6 = len(pd.merge(df_tss[
                          df_tss.date_comp_start.dt.year >=
                          year], df_hw, on=['serial5']))
    process_log, _ = update_report(process_log, n6, n_prior, task)

    # Determine record match rate by dimension
    dim_list = []; m1 = []; m2 = []; m3 = []
    for dim in df_tss[dim2match].unique():
        dim_list += [dim]
        m1_ = len(df_tss[df_tss[dim2match] == dim])
        m1 += [m1_]
        m2_ = len(pd.merge(df_tss[df_tss[dim2match] == dim][
            ['country', 'serial5']], df_hw[df_hw[dim2match] == dim][[
             'country', 'serial5']], on=['serial5']))
        m2 += [m2_]
        m3 += [round(m2_ / m1_, 2)]
    cntry_match = list(zip(dim_list, m1, m2, m3))

    process_log.index = np.arange(len(process_log))

    # Print processing log
    print('TSS-HW preparation summary')
    print(process_log[['task', 'instances', 'gain']])
    print(
        '{:,.0f} HWMA records ({:.1%}, {:.1%}) matched with HW attributes '
        'by SN'.format(len(tss_hw_ij), match1, match2))
    print('----------------------------------------------------------------')
    print('Match rate by ' + dim2match + ':')
    print(cntry_match)

    tss_hw.sort_index(axis=1, inplace=True)

    # Write list of field names to a text file
    write_fields(tss_hw, 'tss_hw', path_out)
    process_log.to_excel(path_out + 'prep_log_hwma+hw.xlsx', index=True)

    return tss_hw, process_log, cntry_match


def prep_hw(path_in, path_out, label_, year=2016, time_sample=False, remove_losses=False,
            exclude_others=True, drop_dup_serial=True, process_log=pd.DataFrame()):
    """

    Args:
        process_log:
        year: year of quote approval to sample
        time_sample: if True, sample data where quote approval year
        matches year
        path_in: path to where data are stored
        path_out: path to where data save
        remove_losses (boolean):
        exclude_others (boolean):
        drop_dup_serial (boolean):

    Returns:

    """

    # Load select columns from each file into a list of DataFrames
    hw_cols = ['QT_REGION', 'QT_COUNTRY', 'QT_QUOTEID', 'QT_APPROVALDATE', 'QT_APPROVALSTATUS', 'QT_CHANNELTYPE',
               'QT_CUSTOMERNB_CMR', 'QT_ClientSegCd', 'QTC_CRMSECTORNAME', 'QTC_CRMINDUSTRYNAME', 'QT_OPPORTUNITYID',
               'QT_VALUESELLER', 'COM_COMPONENTID', 'COM_MTM', 'COMW_MTM_SERIALNO', 'COM_CATEGORY', 'COM_UpgMES',
               'COM_Quantity', 'COM_LISTPRICE', 'COM_ESTIMATED_TMC', 'COM_QuotePrice', 'COM_DelgPriceL4',
               'QTW_WIN_IND', 'DOM_BUY_GRP_ID']

    tic0 = time.time()
    print('------------------------------------------------------------')
    print('Loading ePricer data sets...')

    df_hw1 = load_data(path_in, label=label_, usecols_=hw_cols)
    toc = time.time()
    print('{:,.0f} records with {:.0f} fields loaded in {:.0f} seconds'.
        format(len(df_hw1), len(df_hw1.columns), toc - tic0))

    tic = time.time()
    print('Formatting hardware data fields & values...')
    df_hw = nospace_lowercase(df_hw1.copy(), cols2rename={
        'qt_region': 'region',
        'qt_country': 'country',
        'qt_quoteid': 'quote_id',
        'qt_approvaldate': 'date_approval_hw',
        'qt_approvalstatus': 'approval_status',
        'qt_channeltype': 'chnl_ep',
        'qt_customernb_cmr': 'client_number_cmr',
        'qt_clientsegcd': 'client_segment',
        'qtc_crmsectorname': 'sector',
        'qtc_crmindustryname': 'industry',
        'qt_opportunityid': 'opportunity_id',
        'qt_valueseller': 'value_seller',
        'qtw_win_ind': 'won',
        'com_componentid': 'component_id',
        'com_delgpricel4': 'p_delgl4_hw',
        'com_mtm': 'mtm',
        'comw_mtm_serialno': 'serial_number_hw',
        'com_listprice': 'p_list_hw',
        'com_estimated_tmc': 'cost_hw',
        'com_quoteprice': 'p_bid_hw',
        'com_delgpriceL4': 'p_delegated_hw'}, trim_values=True)

    toc = time.time()
    print('Completed in {:.0f} seconds'.format(toc - tic))

    n_prior = len(df_hw)

    df_hw.won.cat.rename_categories({'y': 1, 'n': 0}, inplace=True)
    df_hw.chnl_ep.cat.rename_categories({'d': 0, 'i': 1}, inplace=True)

    if remove_losses:
        # Remove the duplicate loss records
        task = 'Retained wins'
        # Opportunities of unknown w/l status are excluded
        df_hw = df_hw[df_hw.won == 1]
        process_log, n_prior = update_report(process_log, len(df_hw), n_prior, task)

        task = 'Dropped blank/NaN SN'
        df_hw = df_hw[df_hw.serial_number_hw != '']
        df_hw.dropna(subset=['serial_number_hw'], inplace=True)
        process_log, n_prior = update_report(process_log, len(df_hw), n_prior, task)

    task = 'Dropped blank/NaN win_ind'
    df_hw = df_hw[df_hw.won != '']
    df_hw.dropna(subset=['won'], inplace=True)
    process_log, n_prior = update_report(process_log, len(df_hw),
                                         n_prior, task)

    if exclude_others:
        task = 'Retain HWMA'
        df_hw = df_hw[(df_hw.com_category == 'h') | (df_hw.com_category == 's')]
        # Additional exclusions -- not currently implemented
        # Remove records without customer numbers
        # Remove records where there is a missing value in product
        # categorization column
        # Remove records that are zero priced
        # Remove records where the quoted price (PofL) <= .01
        # Remove records without customer numbers
        process_log, n_prior = update_report(process_log, len(df_hw), n_prior, task)

    # Derive time elements

    date_format = '%d %m %Y'
    df_hw.date_approval_hw = df_hw.date_approval_hw.apply(lambda x:
        dt.datetime.strptime(x, date_format))
    df_hw['date_approval_year'] = df_hw.date_approval_hw.dt.year
    df_hw['date_approval_month'] = df_hw.date_approval_hw.dt.month

    # Sample a particular period of data, per year of quote approval
    if time_sample:
        df_hw = df_hw[df_hw.date_approval_year == year]

    # Retain just the trailing five characters & ensure letters are
    # uppercase
    df_hw['serial5'] = df_hw.serial_number_hw.str[-5:]
    df_hw.serial5 = df_hw.serial5.apply(lambda x: x.upper())

    # Standardize machine type & model values with leading zeroes, as needed
    df_hw['machine_type'] = df_hw.mtm.str[0:4]
    df_hw['machine_model'] = df_hw.mtm.str[-3:]

    tic = time.time()
    print('Derive contract values from component values...')

    df_hw['p_list_hw_row_total'] = df_hw.p_list_hw * df_hw.com_quantity
    df_hw['p_list_hw_total'] = df_hw.groupby('quote_id').p_list_hw_row_total.transform(lambda x: x.sum())

    col_quote = ['quote_id', 'com_category', 'machine_type','machine_model', 'serial5', 'component_id']
    df_hw['p_list_hw_withingroup'] = df_hw.groupby(col_quote).p_list_hw_row_total.transform('sum')

    df_hw['p_pct_list_hw'] = df_hw.p_bid_hw / df_hw.p_list_hw
    df_hw['p_delgl4_pct_list_hw'] = df_hw.p_delgl4_hw / df_hw.p_list_hw

    df_hw['p_bid_hw_row_total'] = df_hw.p_bid_hw * df_hw.com_quantity
    df_hw['p_bid_hw_total'] = df_hw.groupby(
            'quote_id').p_bid_hw_row_total.transform(lambda x: x.sum())
    df_hw['p_bid_hw_withingroup'] = df_hw.groupby(col_quote).p_bid_hw_row_total.transform('sum')
    df_hw['p_pct_hw_withingroup'] = df_hw.p_bid_hw_withingroup / df_hw.p_list_hw_withingroup

    df_hw['p_delgl4_hw_row_total'] = df_hw.p_delgl4_hw * df_hw.com_quantity

    df_hw['p_delgl4_hw_total'] = df_hw.groupby('quote_id').p_delgl4_hw_row_total.transform(lambda x: x.sum())

    # Compute contribution of the total line item value (bid price) as
    # a percentage of the total HW value in the quote
    df_hw['p_bid_hw_contrib'] = df_hw.p_bid_hw_row_total / df_hw.p_bid_hw_total

    # Derive scaled & normalized hardware metrics
    df_hw['cost_hw_row_total'] = df_hw.cost_hw * df_hw.com_quantity
    df_hw['cost_hw_total'] = df_hw.groupby('quote_id').cost_hw_row_total.transform(lambda x: x.sum())
    df_hw['cost_hw_withingroup'] = df_hw.groupby(col_quote).cost_hw_row_total.transform('sum')
    df_hw['count_hw_withingroup'] = df_hw.groupby(col_quote).p_bid_hw_row_total.transform('count')
    df_hw['cost_pct_list_hw'] = df_hw.cost_hw / df_hw.p_list_hw
    df_hw['cost_pct_hw_withingroup'] = df_hw.cost_hw_withingroup / df_hw.p_list_hw_withingroup
    df_hw['gp_hw_withingroup'] = df_hw.p_bid_hw_withingroup - df_hw.cost_hw_withingroup
    df_hw['gp_pct_hw_withingroup'] = df_hw.gp_hw_withingroup / df_hw.p_bid_hw_withingroup
    toc = time.time()
    print('Completed in {:.0f} seconds'.format(toc - tic))

    # Label instances
    # Sale approved in the 3rd month of the quarter
    df_hw['date_approval_eoq'] = np.where(((df_hw.date_approval_month.astype(int) % 3) == 0), 1, 0)

    df_hw.loc[((df_hw.com_upgmes.astype('str') == '4') | (df_hw.com_upgmes.astype('str') == '5')), 'upgrade_mes'] = 1
    df_hw.fillna({'upgrade_mes': 0}, inplace=True)


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
            df_hw = pd.concat([df_hw[df_hw.won == 1].drop_duplicates(subset=col_quote),df_hw[df_hw.won == 0]], axis=0)

        process_log, n_prior = update_report(process_log, len(df_hw), n_prior, task)

    # Remove unneeded fields
    df_hw.drop('com_quantity', axis=1, inplace=True)

    # Sort columns by name
    df_hw.sort_index(axis=1, inplace=True)

    # Write list of field names to a text file
    write_fields(df_hw, 'hw', path_out)

    print('-----------------------------------------------------------')
    print('Total HW preparation time {:.0f} seconds'.format(time.time() - tic0))
    return df_hw, process_log, n_prior


def prep_tss(path_in, path_out, label_, year=2016, time_sample=True):
    """
    Prepares IBM Systems (Power & Storage) hardware maintenance contract
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
        year (int): Year that could sample data or provide process data
        label_ (str): input data files must contain this string
        path_out (str): folder to which to save file(s)
        path_in (str): folder containing the input data in Excel (xlsx) format
        time_sample: Boolean authorizing sampling a particular year

    Returns:
        Prepared data set (DataFrame) and a table logging the number of
        instances after operations that could result in a gain or loss

    """

    task = 'Load'
    tic0 = time.time()
    print('---------------------------------------------------------------')
    print(task + 'ing TSS data...')
    # Specify fields to load; note, however, that pandas' usecols only works
    # with read_excel if it is a comma-separated list of Excel column letters
    # and column ranges (e.g. 'A:E' or 'A,C,E:F'); it cannot work using a
    # list of field names
    hwma = load_data(path_in, label=label_)  # usecols=[]

    # Record current number of rows in the data set & place into a table
    n1 = len(hwma)
    toc = time.time()
    print('{:,.0f} records with {:.0f} fields loaded in {:.0f} '
          'seconds'.format(n1, len(hwma.columns), toc - tic0))
    process_log = pd.DataFrame([{'task': task, 'instances': n1, 'gain': 0}])

    # ------------------------------------------------------------------------
    tic = time.time()
    print('Formatting TSS data...')
    # Remove whitespace around field names & values of select fields
    hwma = nospace_lowercase(hwma, cols2rename={
        'contr_nbr': 'contract_number', 'model': 'machine_model',
        'ser_nbr': 'serial_number', 'typ': 'machine_type',
        'cust_name': 'customer_name', 'cust_nbr': 'customer_number', 'geo_cd':
        'geo', 'cntry': 'country_code', 'cntry_desc': 'country',
        'gross_amt_usd': 'p_list_per', 'bill_amt_usd': 'p_bid_per',
        'calc_start_date': 'date_calc_start',
        'calc_stop_date': 'date_calc_stop',
        'chnl': 'chnl_tss',
        'comp_start_date': 'date_comp_start',
        'comp_stop_date': 'date_comp_stop',
        'contr_start_date': 'date_contr_start',
        'contr_stop_date': 'date_contr_stop',
        'discnt_tot': 'discount0', 'fctr': 'p_uplift_comm',
        'reg_cd': 'market', 'sl_cd': 'sl_code', 'sl_cntct_nm': 'sl_cntct',
        'sl_onsite_nm': 'sl_onsite', 'sl_fix_time_nm': 'sl_fix_time',
        'sl_part_time_nm': 'sl_part_time', 'sl_cov_nm': 'sl_cov',
        'offer_nm': 'tss_type', 'offer_sdesc': 'tss_type_desc'},
        trim_values=True,
        fields=['country', 'contract_number', 'machine_model',
                'contr_nbr_chis', 'tss_type', 'tss_type_desc'])

    # Encode sales channel by changing the category labels
    hwma.chnl_tss.cat.rename_categories({'F2F': 0, 'BP': 1}, inplace=True)

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
    hwma['mtm'] = hwma.machine_type + hwma.machine_model

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
    hwma = make_a_date(hwma, ['date_calc_start', 'date_calc_stop',
                              'date_contr_start', 'date_contr_stop',
                              'date_comp_start', 'date_comp_stop',
                              'date_inst', 'date_warr_end',
                              'date_srv_start', 'date_srv_stop'])

    # Assume TSS component services marked to stop in 1999 will instead end
    # 10 years after their start date
    hwma.loc[hwma.date_contr_stop.dt.year == 1999, 'date_contr_stop'] = hwma.date_contr_start + pd.DateOffset(years=5)
    hwma.loc[hwma.date_comp_stop.dt.year == 1999, 'date_comp_stop'] = hwma.date_comp_start + pd.DateOffset(years=5)

    toc = time.time()
    print('Formatted in {:.0f} seconds'.format(toc - tic))

    # ------------------------------------------------------------------------
    tic = time.time()
    print('Sampling TSS data...')
    n2017_1 = len(hwma[hwma.date_comp_start.dt.year >= year])
    n2017_2 = len(hwma[(hwma.date_comp_start.dt.year >= year) & (hwma.serial5.notnull())])

    task = 'Remove non-maintenance'
    # mach_ce_stat: 2 (under maintenance), 3 (no maintenance), or 9 (under
    # base warranty, so not billing TSS)
    hwma = hwma[hwma.mach_ce_stat != 3]

    # Check how many records were removed: record number of rows in DataFrame
    n2 = len(hwma)
    instance_loss = n2 - n1
    next_task = pd.DataFrame([{'task': task, 'instances': n2,
                               'gain': instance_loss}])
    process_log = process_log.append(next_task)

    # Sample a particular period of data, per year of quote approval
    if time_sample:
        task = '2016-2018 installations'
        hwma = hwma[(hwma.date_inst.dt.year >= year) &
                    (hwma.date_inst.dt.year <= 2018) &
                    (hwma.date_inst < (hwma.date_warr_end - pd.DateOffset(days=180)))]

        # Check how many records were removed: record number of rows in DataFrame
        n3_ = len(hwma)
        instance_loss = n3_ - n2
        next_task = pd.DataFrame([{'task': task, 'instances': n3_,
                                   'gain': instance_loss}])
        process_log = process_log.append(next_task)
        # Overwrite previous record count
        n2 = n3_

    task = 'Removed null SN'
    hwma.dropna(subset=['serial_number'], inplace=True)

    n3 = len(hwma)
    instance_loss = n3 - n2
    next_task = pd.DataFrame([{'task': task, 'instances': n3,
                               'gain': instance_loss}])
    process_log = process_log.append(next_task)

    task = 'Removed NaN (other fields)'
    hwma.dropna(subset=['contract_number', 'p_bid_per',
                        'p_list_per', 'date_calc_start',
                        'date_calc_stop', 'date_contr_start',
                        'date_contr_stop', 'date_comp_start',
                        'date_comp_stop'], inplace=True)

    # Check how many records were removed: record number of rows in DataFrame
    n4 = len(hwma)
    instance_loss = n4 - n3
    next_task = pd.DataFrame([{'task': task, 'instances': n4,'gain': instance_loss}])

    process_log = process_log.append(next_task)

    task = 'Removed <0 billings & bill>gross & rn!=1'
    # rn is a slicer; if only records with rn = 1 are included, there will be
    # no repetitions of machine serial number, for which there might be
    # multiple records
    hwma = hwma[(hwma.p_bid_per > 0) &
                (hwma.p_list_per >= hwma.p_bid_per) &
                (hwma.rn == 1)]

    # Check how many records were removed: record number of rows in DataFrame
    n5 = len(hwma)
    instance_loss = n5 - n4
    next_task = pd.DataFrame([{'task': task, 'instances': n5,
                             'gain': instance_loss}])
    process_log = process_log.append(next_task)

    task = 'Remove contr extensions'
    # by removing duplicates with respect to select machine & contract
    # attributes where time-of-HW sale and extensions will share common
    # values, retaining the oldest instance representing the most likely
    # time-of-sale instance.  However, this approach is not removing many
    # instances.  Stefanie suggested another field "F0001"
    hwma = hwma.sort_values('date_contr_start', ascending=False).\
        drop_duplicates(subset=['machine_type', 'serial_number','contract_number'], keep='last')

    # Check how many records were removed: record number of rows in DataFrame
    n6 = len(hwma)
    instance_loss = n6 - n5
    next_task = pd.DataFrame([{'task': task, 'instances': n6,
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
    n7 = len(hwma)
    instance_loss = n7 - n6
    next_task = pd.DataFrame([{'task': task, 'instances': n7,
                               'gain': instance_loss}])
    process_log = process_log.append(next_task)

    toc = time.time()
    print('Sampled in {:.0f} seconds'.format(toc - tic))

    tic = time.time()
    print('Engineering features...')
    # Infer bundling of services at point of hardware sale & engineer a
    # categorical variable:
    # positive label = TSS contract is attached to HW sales
    # Rule 1. contract signed or service started within 60 days of hardware
    #  installation or
    # Rule 2. hardware installed before half a year to warranty end
    # Rule 3. service stopped on day warranty ended, mainly to address cases
    # where contract starts long after installation

    # o--o------<-?->-------o---------------o---------o
    #   +30d             -365 d           end      +200d
    # |---|-attached starts
    #                       |----aftermarket starts---|
    hwma['tss_bundled'] = 1.0 * \
        ((hwma.date_contr_start < hwma.date_inst + pd.DateOffset(days=30)) |
         (hwma.date_srv_start < hwma.date_inst + pd.DateOffset(days=30)) |
         (hwma.date_srv_stop == hwma.date_warr_end))

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
    # If confident in tss_bundled, derive sale_aftermarket as its inverse
    # hwma['sale_aftermarket'] = hwma.tss_bundled.where(
    # hwma.tss_bundled == 0, 1)

    hwma['comp_inst_year'] = hwma.date_inst.dt.year
    hwma['comp_duration_days'] = (hwma.date_comp_stop - hwma.date_comp_start).dt.days

    hwma['comp_duration_days'].loc[(hwma.comp_duration_days > 1825) | (hwma.comp_duration_days < 0)] = 1825

    hwma['comp_duration_months'] = hwma.comp_duration_days / 30
    hwma['period_billing_days'] = (hwma.date_calc_stop - hwma.date_calc_start).dt.days
    hwma['period_billing_months'] = np.round(hwma.period_billing_days / 30)

    hwma['comp_periods'] = np.ceil(hwma.comp_duration_days / hwma.period_billing_days)

    hwma['p_list_hwma'] = hwma.p_list_per * hwma.comp_periods
    hwma['p_bid_hwma'] = hwma.p_bid_per * hwma.comp_periods

    hwma['p_pct_list_hwma'] = hwma.p_bid_per / hwma.p_list_per
    hwma['discount_hwma'] = 1 - hwma.p_pct_list_hwma

    # Derive contract values from component values
    hwma['p_list_hwma_total'] = hwma.groupby('contract_number'). \
        p_list_hwma.transform(lambda x: x.sum())

    hwma['p_bid_hwma_total'] = hwma.groupby('contract_number'). \
        p_bid_hwma.transform(lambda x: x.sum())

    # In a new field, convert the continuous (numeric) committed service
    # uplift to a binary categorical variable
    hwma.loc[:, 'committed'] = hwma.p_uplift_comm.where(hwma.p_uplift_comm
                                                        == 0, 1)

    task = 'Removed <0 bill periods'
    hwma = hwma[hwma.period_billing_months > 0]

    # Check how many records were removed: record number of rows in DataFrame
    n8 = len(hwma)
    instance_loss = n8 - n7
    next_task = pd.DataFrame([{'task': task, 'instances': n8,
                             'gain': instance_loss}])
    process_log = process_log.append(next_task)

    # For regression, artificially increase zero discounts by a tiny amount
    hwma[hwma.discount_hwma == 0].discount_hwma += 0.0001


    task = 'Removed EOS ext'
    hwma = hwma[hwma.tss_type != 'HW EOS EXT']

    # Check how many records were removed: record number of rows in DataFrame
    n9 = len(hwma)
    instance_loss = n9 - n8
    next_task = pd.DataFrame([{'task': task, 'instances': n9,
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

    toc = time.time()
    print('Features engineered in {:.0f} seconds'.format(toc - tic))
    print('---------------------------------------------------------------')

    # Remove unneeded & unreliable column(s)
    hwma.drop(['mmmc_mach_usd', 'rn', 'date_calc_start', 'date_calc_stop',
               'date_inst_sof', 'ff_chnl_sdesc', 'iw_manr', 'mach_ce_stat',
               'mach_stat_fut', 'mcc', 'origin', 'period_billing_days',
               'period_billing_months', 'platform', 'pymnt_opt',
               'prod_div', 'sector_sdesc', 'tss_type_desc', 'sl_cntct',
               'sl_cov'], axis=1, inplace=True)

    # Sort columns by name
    hwma.sort_index(axis=1, inplace=True)

    # Write list of field names to a text file
    write_fields(hwma, 'hwma', path_out)

    # Print processing log
    process_log.index = np.arange(len(process_log))
    print('HWMA preparation summary')
    print(process_log[['task', 'instances', 'gain']])
    print('---------------------------------------------------------------')
    print('Total TSS preparation time {:.0f} seconds'.format(time.time() -
                                                             tic0))

    return hwma, process_log


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


def update_report(process_log, n_current, n_prior, task_='unspecified'):
    instance_loss = n_current - n_prior
    next_task = pd.DataFrame([{'task': task_, 'instances': n_current,
                               'gain': instance_loss}])
    log = process_log.append(next_task)

    return log, n_current


def write_fields(df, file_prefix, path):
    """
    Produces a vertical list of DataFrame's field names

    Args:
        df: DataFrame
        file_prefix: name of text file
        path: local folder in which to save the text file

    Returns:
        Nothing explicit; creates a text file on a local hard drive
    """
    df_cols = open(path + file_prefix + '_cols' + '.txt', 'w')
    for item in list(df.columns):
        df_cols.write("%s\n" % item)


def gtms_prep(file_gtms, file_mtm_map, directory):

    df = pd.read_csv(os.path.join(directory, file_gtms), low_memory=False)
    mtm_map = pd.read_csv(os.path.join(directory, file_mtm_map), low_memory=False)

    df_gtms = nospace_lowercase(df, cols2rename={'type_c': 'machine_type'
                                                , 'model': 'machine_model'
                                                , 'ser_nbr_or_mkt_id': 'serial_number'
                                                , 'offer_nm': 'tss_type'
                                                , 'contr_nbr_chis': 'contract_number'
                                                , 'reg_cd': 'market'},
                                trim_values=True,
                                fields=['serial_number', 'machine_type', 'machine_model',
                                        'contract_number', 'channel_contract', 'tss_type',
                                        'ff_chnl_cd_chis', 'market'])

    # Create lower case country column
    df_gtms['country'] = df_gtms.cntry_nm.str.lower()

    # Retain just the trailing five characters
    df_gtms['serial5'] = df_gtms['serial_number'].str[-5:]

    # Change field types to string -- does not preserve when exported to csv and
    df_gtms.machine_type = [val.zfill(4) for val in df_gtms.machine_type.astype('str')]
    df_gtms.machine_model = [val.zfill(3) for val in df_gtms.machine_model.astype('str')]

    print(len(df_gtms))
    df_gtms = df_gtms[df_gtms.tss_type == 'tms'].copy()
    print(len(df_gtms))
    df_gtms = df_gtms[df_gtms.channel_contract.isin(['face_to_face_a', 'bp_channel_ghj'])].copy()
    print(len(df_gtms))

    df_gtms['chnl_tss'] = (df_gtms['ff_chnl_cd_chis'] != 'a') * 1
    df_gtms['chnl_tss'] = df_gtms['chnl_tss'].astype(str)

    df_gtms['mtm'] = df_gtms.machine_type.map(str) + df_gtms.machine_model

    df_gtms['comp_stop_date'] = df_gtms.comp_stop_date.astype('str')
    df_gtms['contr_stop_date'] = df_gtms.contr_stop_date.astype('str')
    df_gtms['calc_stop_date'] = df_gtms.calc_stop_date.astype('str')

    df_gtms = make_a_date(df_gtms, ['contr_start_date', 'contr_stop_date', 'comp_start_date', 'comp_stop_date',
                                    'calc_start_date', 'calc_stop_date'])

    df_gtms.loc[df_gtms.comp_stop_date.dt.year == 1999,
                'comp_stop_date'] = df_gtms['comp_start_date'] + pd.DateOffset(years=5)
    df_gtms.loc[df_gtms.contr_stop_date.dt.year == 1999,
                'contr_stop_date'] = df_gtms['contr_stop_date'] + pd.DateOffset(years=100)

    df_gtms = df_gtms[df_gtms.contr_start_date.dt.year > 2015].copy()

    # Remove negative/null bill amount line item
    df_gtms = df_gtms[df_gtms.bill_amt_usd > 0].copy()
    # Remove line item with bill>gross
    df_gtms = df_gtms[df_gtms.gross_amt_usd >= df_gtms.bill_amt_usd].copy()

    df_gtms['p_pct_list_hwma'] = df_gtms.bill_amt_usd / df_gtms.gross_amt_usd

    df_gtms['bill_period_length'] = (df_gtms.calc_stop_date - df_gtms.calc_start_date).dt.days
    df_gtms['comp_duration'] = (df_gtms.comp_stop_date - df_gtms.comp_start_date).dt.days
    df_gtms['comp_periods'] = df_gtms.comp_duration / df_gtms.bill_period_length

    df_gtms = df_gtms[(df_gtms.bill_period_length > 300)].copy()
    print(len(df_gtms))

    df_gtms['p_list_hwma'] = df_gtms.gross_amt_usd * df_gtms.comp_periods
    df_gtms['p_bid_hwma'] = df_gtms.bill_amt_usd * df_gtms.comp_periods

    # df_gtms['costfactor'] = .4
    # df_gtms['cost'] = df_gtms['p_list_hwma'] * df_gtms['costfactor']

    # Add mtm map
    mtm_map = mtm_map[['level_0', 'level_1', 'level_2', 'level_3', 'level_4']]
    mtm_map = mtm_map.rename(columns={'level_0': 'taxon_hw_level_0',
                                      'level_1': 'taxon_hw_level_1',
                                      'level_2': 'taxon_hw_level_2',
                                      'level_3': 'taxon_hw_level_3',
                                      'level_4': 'taxon_hw_level_4'})

    df_gtms = pd.merge(df_gtms, mtm_map, left_on='mtm', right_on='taxon_hw_level_0', how='left')

    col_gtms = ['contract_number', 'tss_type', 'chnl_tss', 'mtm', 'serial5', 'market', 'country',
                'p_pct_list_hwma', 'p_list_hwma', 'p_bid_hwma', 'taxon_hw_level_4', 'taxon_hw_level_3',
                'taxon_hw_level_2', 'taxon_hw_level_0', 'contr_start_date', 'contr_stop_date',
                'comp_start_date', 'comp_stop_date', 'machine_type', 'machine_model', 'sl_cd']


    df_gtms = df_gtms[col_gtms]

    df_gtms = df_gtms.rename(index=str, columns={"taxon_hw_level_0": "taxon_hw_mtm",
                                                 'sl_cd': 'sl_code',
                                                 'contr_start_date': 'date_contr_start',
                                                 'contr_stop_date': 'date_contr_stop',
                                                 'comp_start_date': 'date_comp_start',
                                                 'comp_stop_date': 'date_comp_stop'})

    df_gtms.date_contr_start = pd.to_datetime(df_gtms.date_contr_start)

    df_gtms = df_gtms.loc[df_gtms['p_pct_list_hwma'] >= 0.1]
    df_gtms['tss_bundled'] = 0
    df_gtms['committed'] = 0

    return df_gtms