"""
    Script for Cleansing COPRA HW Raw Data
    --------------------------------------    
    
    Code written:   19 May 2016 by SaiNaveen Sare & Glenn Melzer
    Previous update:    5 Jan 2018 by Yuxi Chen
    Recent update:    27 Feb 2018 by Aaron Slowey

    The purpose of this script is to read in the .CSV file of the
    ePricer IW QMF query and then create an output file of the
    correct format for use in the COPRA_OP training process.  The
    output file will likely be smaller than the input file.
    
    This version of the algorithm includes logic that removes duplicate
    lost quotes.  The removed quotes appear to be earlier versions of a 
    final approved quote.  The logic compares a pair of related quotes and
    creates a score that determines how closely related they are.  If the
    score is below a 'dup_cotoff' threshold, then the older of the two 
    quotes is considered a duplicate and it is removed.  The scoring criteria
    is based on comparing for each quote:  
        1) Total quote proposed price
        2) Quote leading brand proposed price
        3) Number of components in the quote
        4) The date of the quote

    INPUTS (defined in the script below):
      data_path = the path to the folder where the data files are stored
      file_in = the name of the .csv file to be read in
      file_out = the name of the cleansed .csv file to be written out
      analysis_out = the name of the analysis file the documents the 
         cleansing process statistics
      QuotesOut = the name of the file that summarizes all of the 
         processed quotes.  This file shows what quotes were removed
         for duplicate lost quote removal and how many quote versions
         existed.
      remove_all_losses = (True/False) this is used to indicate whether all
         loss data should be removed.
      dup_cutoff = the value used for removing duplicate quotes.  This 
         parameter is used when the remove_all_losses parameter is set
         to False.  The dup_cutoff value is typically set between 0 and 1.
         A value of 0 means a quote is a duplicate if it is an exact match.
         A value of 1 means a quote may be considered a duplicate even if
         it is quite different.  A typical value is 0.2 which will 
         remove about 50% of potential duplicate quotes.
      
    OUTPUT:
      The file_out file is written

    INSTRUCTIONS FOR USE:
      1) Go to SECTION A of the code and fill in the correct data_path,
         file_in, and file_out names for your situation.
         NOTE:  The remove_all_losses parameter is set to True or False.
         If True, all loss records will be removed.  If False, normal
         cleansing may remove some of the loss records.
      2) Go to SECTION C of the code and define how the data from the
         file_in source table (labeled InputData in the section) will
         be defined and stored in the file_out table (labeled OutputData
         in the section).  The file_out table needs to be in the format
         that can be used by COPRA_OP.
      3) Go to SECTION D of the code and define what records should be
         removed from the file_out table.  This is where records that
         have problems are removed.
"""

import math
import pandas as pd
import numpy as np
from pandas import DataFrame
import time
from collections import OrderedDict
import os


def match_df(df1, df2):
    joined_df = DataFrame(dict(df1=df1, df2=df2)).reset_index().dropna()
    # joined_df = joined_df.drop_duplicates(inplace=True)
    ##added above for debugging purpose
    return joined_df


def insert_missing_cols(df, cmp_cols):
    df_cols = df.columns
    if np.array_equal(cmp_cols.values, df_cols.values):
        return df
    else:
        counter = -1
        for i in range(len(cmp_cols)):
            counter += 1
            if cmp_cols[i] != df_cols[counter]:
                df.insert(i, cmp_cols[i], 'N/A')
                counter -= 1
        return df


# This function appends a row to the report_df to describe how the OutputData has changed
def report_update(report_df, OutputData, index_name):
    # Create a country_count DataFrame of OutputData counts by country
    OutputData[index_name] = 1
    country_count = OutputData.loc[:, ('CountryCode', index_name)].groupby(
        'CountryCode').sum().T
    country_count['Total'] = country_count.sum(axis=1)
    country_columns = pd.DataFrame(index=[index_name],
                                   columns=report_df.columns[:(len(
                                       report_df.columns) / 3)]).fillna(0)
    for i in list(country_count.columns[:]):
        country_columns.set_value(index_name, i, country_count.iloc[0][i])

    # Determine new row column values
    delta_columns = ['%s_%s' % (country, 'delta') for country in
                     country_columns]
    final_row = report_df.ix[-1][country_columns.columns]
    # delta_values = country_count.loc[index_name].values - final_row.values
    delta_values = country_columns.loc[index_name].values - final_row.values
    delta_count = pd.DataFrame([delta_values.tolist()], columns=delta_columns,
                               index=country_count.index)

    pct_columns = ['%s_%s' % (country, 'pct') for country in country_columns]
    initial_records = report_df.loc['Initial Number of Records'][
        country_columns.columns]
    precentage = (delta_values / initial_records) * 100
    pct_df = pd.DataFrame([precentage.tolist()], columns=pct_columns,
                          index=country_count.index)
    pct_df = pct_df.applymap(lambda x: '%.2f%%' % x)

    # Append last row to report_df
    final_df = pd.concat([country_columns, delta_count, pct_df], axis=1)
    report_df.loc[index_name] = final_df.loc[index_name].values
    del OutputData[index_name]

    return report_df


# ----------------------------------------------------------------------------
# SECTION A
# Define the required path, file names, and parameters
data_path = '~/Box Sync/Cognitive Pricing/COPRA_HW/WorkStreams/NA_Pricing/Data/Sample_data/YuxiTestNA_oldprodmap_0104/'
file_in = 'NA_trainingset_raw_bundled_YC_010418.csv'
file_out = 'NA_trainingset_cleanWL_unique_com_SBC_bundled.csv'
analysis_out = 'NA_trainingset_analysis_SBC.csv'
QuotesOut = 'NA_trainingset_QuotesOut_SBC.csv'

remove_all_losses = True  # False removes only duplicate losses
dup_cutoff = 0.2  # threshold below which duplicates are deleted,
# when remove_all_losses=False)

# Reset EMEA SW TMC costs in months before July 2017?  Code section C+
# remove when no longer needed
EMEA_SW_Cost_Cleanup = False

# start the timer
tic = time.time()

# Read the import file for processing
print('    Reading historical data input file:             ')
print('     ', file_in)
InputData = pd.read_csv(data_path + file_in, low_memory=False,
                        lineterminator='\n').fillna('')  # add lineterminator = '\n', Yuxi 12/16/17
print('      Number of input records:      '), len(InputData.index)
print('        ...Processing Records...')

# ----------------------------------------------------------------------------
# SECTION B
# Initialize the report_df DataFrame with record counts by country
initial_records = 'Initial Number of Records'
print('          >determine initial number of records')

# Determine the Countries
# InputData['CountryCode'] = InputData['QUOTEID'].str.strip().str[-2:]
InputData['CountryCode'] = InputData[
    'QT_COUNTRYCODE'].str.strip()  # Edited by YC 11/03/17
InputData[initial_records] = 1
# Find the record count for each country
country_count = InputData[['CountryCode', initial_records]].groupby(
    'CountryCode').sum().T
country_count['Total'] = country_count.sum(axis=1)
# Load the DataFrame
delta_columns = OrderedDict(
        (country, '%s_%s' % (country, 'delta')) for country in
        country_count.columns)
delta_count = country_count.rename(columns=delta_columns)
pct_columns = ['%s_%s' % (country, 'pct') for country in
               country_count.columns]
pct_df = pd.DataFrame([['%s%%' % 100.00] * len(pct_columns)],
                      columns=pct_columns, index=country_count.index)
report_df = pd.concat([country_count, delta_count, pct_df], axis=1)
# print 'report_df: ', report_df

# ----------------------------------------------------------------------------
# SECTION C
# Create and Load the OutputData with data needed for COPRA_OP
OutputData = pd.DataFrame()  # Create an empty OutputData DataFrame

# Adding all columns to empty data frame as per hierarchy
OutputData['QuoteID'] = InputData['QT_QUOTEID'].astype(
    str).str.strip()  # remove blanks in text # Edit YC 11/03/17
OutputData['CountryCode'] = InputData[
    'CountryCode'].str.strip()  # pull off the last 2 character in Quote ID (Country code)

# OutputData['ChannelID'] = InputData['QT_CHANNELID'].str.strip()  # remove blanks in text # Edit field name YC 11/03/17
OutputData['ChannelID'] = InputData[
    'QT_CHANNELTYPE']  # New Field Edit field name YC 11/03/17

OutputData['CustomerNumber'] = InputData[
    'QT_CUSTOMERNB_CMR'].str.strip()  # remove blanks in text
OutputData['CRMSectorName'] = InputData[
    'QTC_CRMSECTORNAME'].str.strip()  # remove blanks in text
OutputData['CRMIndustryName'] = InputData[
    'QTC_CRMINDUSTRYNAME'].str.strip()  # remove blanks in text
OutputData['DomBuyGrpID'] = InputData[
    'DOM_BUY_GRP_ID'].str.strip()  # remove blanks in text
OutputData['DomBuyGrpNam'] = InputData[
    'DOM_BUY_GRP_NAME'].str.strip()  # remove blanks in text
OutputData['OpportunityID'] = InputData[
    'QT_OPPORTUNITYID'].str.strip()  # remove blanks in text

# Add HWPlatFormID, UFC and componentID
OutputData['ComHWPlatFormID'] = InputData['COM_HWPLATFORMID']
OutputData['UFC'] = InputData['FET_FEATURE'].str.strip()
OutputData['Com_ComponentID'] = InputData['COM_COMPONENTID']

# Add special bid codes filed Edited by Yuxi 11/21/17
OutputData['SBC1'] = InputData['COM_Specialbidcode1']
OutputData['SBC2'] = InputData['COM_Specialbidcode2']
OutputData['SBC3'] = InputData['COM_Specialbidcode3']
OutputData['SBC4'] = InputData['COM_Specialbidcode4']
OutputData['SBC5'] = InputData['COM_Specialbidcode5']
OutputData['SBC6'] = InputData['COM_Specialbidcode6']

# Comment out the below line as it not being used anymore --- Bonnie
# OutputData['BPQuoteStatus'] = InputData['BPQUOTESTATUS']

# Edited by Bonnie
# OutputData['Value_seller'] = 1 * (InputData['QT_VALUESELLER'].str.strip() =='Value Seller')
OutputData['Value_seller'] = InputData['QT_VALUESELLER']  # Edited by YC
# OutputData['ValueSeller(1/0)'] = OutputData['BPQuoteStatus'].isin([31]).astype(int)

OutputData['ComListPrice'] = InputData['COM_LISTPRICE'] * InputData[
    'COM_Quantity']
OutputData['ComTMC'] = InputData['COM_ESTIMATED_TMC'] * InputData[
    'COM_Quantity']
# Edited by YC 11/13
OutputData['UnitComListPrice'] = InputData['COM_LISTPRICE']
OutputData['UnitComTMC'] = InputData['COM_ESTIMATED_TMC']

# Edited by Bonnie
OutputData['ComQuotePrice'] = InputData['COM_QuotePrice'] * InputData[
    'COM_Quantity']
OutputData['ComDelgPriceL4'] = InputData['COM_DelgPriceL4'] * InputData[
    'COM_Quantity']

####### Edited by Bonnie#######
OutputData['ComQuantity'] = InputData['COM_Quantity']
#####################

OutputData['WinLoss'] = (InputData[
                             'COMW_WIN_IND'] == 'Y') * 1  # wins should be given a "1" value
OutputData['ComRevCat'] = InputData[
    'COM_CATEGORY'].str.strip()  # remove blanks in text
OutputData['ComRevDivCd'] = InputData[
    'COM_RevDivCd'].str.strip()  # remove blanks in text
OutputData['ComBrand'] = InputData[
    'PRODUCT_BRAND'].str.strip()  # remove blanks in text
OutputData['ComGroup'] = InputData[
    'PRODUCT_GROUP'].str.strip()  # remove blanks in text
OutputData['ComFamily'] = InputData[
    'PRODUCT_FAMILY'].str.strip()  # remove blanks in text
# the training "a" ensures COPRA_OP will treat this as text not numeric data
OutputData['ComMT'] = InputData['COM_MTM'].str[:4] + 'a'
# the training "a" ensures COPRA_OP will treat this as text not numeric data
OutputData['ComMTM'] = InputData['COM_MTM'].str[:7] + 'a'

# Edited by Yuxi Chen 12/16/17: new product map
# OutputData['LVL4'] = InputData['LVL4'].str.strip()
# OutputData['LVL3'] = InputData['LVL3'].str.strip()
# OutputData['LVL2'] = InputData['LVL2'].str.strip()
# OutputData['LVL1'] = InputData['LVL1'].str.strip()

# Creating the Year column from QUOTE_DATE of InputData file
# Edited by Bonnie Bao
OutputData['Year'] = InputData['QT_APPROVALDATE'].astype(str).str[
                     :4]  # Edited by YC
# Creating the month column from QUOTE_DATE of InputData file
OutputData['Month'] = InputData['QT_APPROVALDATE'].astype(str).str[4:6]
# if Month=12,9,6,3 then EndOfQtr=1 else 0
OutputData['EndOfQtr'] = np.where(
        ((OutputData['Month'].astype(int) % 3) == 0), 1, 0)
OutputData['ClientSegCd'] = InputData[
    'QT_ClientSegCd'].str.strip()  # remove blanks in text
# UpgradeFlag definition: 1=new, 3=cancellation, 4=MES, 5=Upg, C=RPO (Record Purposes Only)
# df.column_name = df.column_name.astype(np.int64)
OutputData['UpgMES'] = (InputData.COM_UpgMES.astype(
    str) >= '4') * 1  # ['COM_UPGRADEFLAG'] >= 4)*1 # Edited by YC need check?
# InputData['CLIENT_SEG_CD']='E' then 1 in below columns else 0
OutputData['ClientSeg=E'] = InputData['QT_ClientSegCd'].str.strip().isin(
        ['E']).astype(int)
# Creating the log of the list Price
OutputData['ComLogListPrice'] = np.log10(OutputData['ComListPrice'])
# OutputData['ChannelID']=F,G,H,I,J,K,M then below column should be 1 else 0
# OutputData['Indirect(1/0)'] = OutputData['ChannelID'].str.strip().isin(['F', 'G', 'H', 'I', 'J', 'K', 'M', 'U']).astype(int)
OutputData['Indirect(1/0)'] = (OutputData[
                                   'ChannelID'] == 'I') * 1  # edited by YC 11/7/17
# creating ComQuotePricePofL and ComDelgPriceL4PofL columns
OutputData['ComQuotePricePofL'] = 1.0 * OutputData['ComQuotePrice'] / \
                                  OutputData['ComListPrice']
OutputData['ComDelgPriceL4PofL'] = 1.0 * OutputData['ComDelgPriceL4'] / \
                                   OutputData['ComListPrice']
# OutputData['ComCostPofL'] = 1.0 * OutputData['ComTMC'] / OutputData['ComListPrice']

# DomClientID is missing in NA sample data
# OutputData['DomClientID'] = InputData['DOM_CLIENT_ID'].str.strip()

#### Data fields required by HW dashboard ### Edited by Bonnie Bao
OutputData['System_type'] = InputData['SYSTEM_TYPE'].str.strip()
OutputData['Quote_date'] = InputData['QT_APPROVALDATE']  # Edited by YC

# Edited by Yuxi: add bundled information -- 12/16/17
OutputData['Bundle'] = InputData['Bundle'].str.strip()
OutputData['bundled'] = InputData['bundled']
OutputData.loc[OutputData.bundled == 0, 'Bundle'] = 'NB'
del OutputData['bundled']

# create the blank columns to be  filled in by COPRA_OP (these empty columns are required)
OutputData['ComLowPofL'] = ''
OutputData['ComMedPofL'] = ''
OutputData['ComHighPofL'] = ''
OutputData['ComMedPrice'] = ''
OutputData['DealSize'] = ''
OutputData['LogDealSize'] = ''
OutputData['ComPctContrib'] = ''

# SECTION C+
# the following code section is for fixing SW costs in EMEA for transactions before July 2017
# this section should be removed once all historical data is past June 2017
if EMEA_SW_Cost_Cleanup == True:
    for i in range(len(OutputData.index)):
        date_calc = (int(OutputData.Year[i]) - 2015) * 12 + int(
                OutputData.Month[
                    i])  # this is the number of months after Dec 2015
        if (OutputData.ComRevCat[i] == 'S') and (
                date_calc < 31):  # use 31 for SW components before Jul 2017
            test = OutputData.ComTMC[i] / OutputData.ComQuotePrice[i]
            if test <= .13:  # if tmc/quote price < 13%, make TMC 2.5% of list price
                OutputData.set_value(i, 'ComTMC',
                                     OutputData.ComListPrice[i] * .025)
            elif (test > .13) and (
                    test <= .5):  # if tmc/quote price > 13% and <= 50%, make TMC 10% of list price
                OutputData.set_value(i, 'ComTMC',
                                     OutputData.ComListPrice[i] * .1)
            elif (test > .5) and (
                    test <= .9):  # if tmc/quote price > 50% and <= 90%, make TMC 25.6% of list price
                OutputData.set_value(i, 'ComTMC',
                                     OutputData.ComListPrice[i] * .256)
            else:  # if tmc/quote price > 90%, make TMC 62% of list price
                OutputData.set_value(i, 'ComTMC',
                                     OutputData.ComListPrice[i] * .62)

# Cost percentage of list price should be updated as we assign new values to ComTMC of SW line items
OutputData['ComCostPofL'] = 1.0 * OutputData['ComTMC'] / OutputData[
    'ComListPrice']

# Remove duplicate line items

# ----------------------------------------------------------------------------
# SECTION D
# Remove non-compliant records from InputData that don't meet the requirements of COPRA_OP

# >> Remove records of lost quotes (if remove_all_losses flag above is set to True)
if remove_all_losses == True:
    OutputData = OutputData[(OutputData['WinLoss'] == 1)]
    print('          >losses removed')
    report_df = report_update(report_df, OutputData, 'Remove all loss quotes')

# >> Remove records that are neither HW nor SW (maintenance)
OutputData = OutputData[
    ((OutputData['ComRevCat'] == 'H') | (OutputData['ComRevCat'] == 'S'))]
report_df = report_update(report_df, OutputData, 'Remove maintenance records')

# >> Remove records without customer numbers
OutputData = OutputData[(OutputData['CustomerNumber'] != '')]
print('          >records without customer numbers removed')
report_df = report_update(report_df, OutputData,
                          'Remove records with missing customer numbers')

# >> Remove records where there is a missing value in product categorization coluumn
OutputData = OutputData[(OutputData['ComRevCat'] != '')]
# New Product map -- Yuxi Chen 12/16/17
OutputData = OutputData[(OutputData['ComRevDivCd'] != '')]
OutputData = OutputData[(OutputData['ComBrand'] != '')]
OutputData = OutputData[(OutputData['ComGroup'] != '')]
OutputData = OutputData[(OutputData['ComFamily'] != '')]
OutputData = OutputData[(OutputData['ComMT'] != 'a')]
OutputData = OutputData[(OutputData['ComMTM'] != 'a')]
# OutputData = OutputData[(OutputData['LVL4'] != '')]
# OutputData = OutputData[(OutputData['LVL3'] != '')]
# OutputData = OutputData[(OutputData['LVL2'] != '')]
# OutputData = OutputData[(OutputData['LVL1'] != '')]
print('          >records with missing product hierarchy removed')
report_df = report_update(report_df, OutputData,
                          'Remove records with missing product hierarchy')

# >> Remove records that are zero priced
# OutputData = OutputData[(OutputData['ComListPrice'] != 0)]
# OutputData = OutputData[(OutputData['ComQuotePrice'] != 0)]
OutputData = OutputData[
    (OutputData['ComListPrice'] > 0)]  # remove -1's Edited by Yuxi 12/20/17
OutputData = OutputData[(OutputData['ComQuotePrice'] > 0)]

print('          >records with zero price removed')
report_df = report_update(report_df, OutputData, 'Remove zero priced records')

# >> Remove records where the quoted price (PofL) <= .01
OutputData = OutputData[(OutputData['ComQuotePricePofL'] > .01)]
print('          >records with >= 99% discount removed')
report_df = report_update(report_df, OutputData,
                          'Remove quote prices with >= 99% disc')

# Final processing of the OutputData & report_df DataFrames
OutputData = OutputData.drop_duplicates(['QuoteID', 'Com_ComponentID'])

# >> Remove the duplicate loss records
quote_summary = DataFrame()  # this creates a DataFrame for analyzing duplicate quotes
output_first = OutputData.groupby('QuoteID').first()
output_first['QuoteMonthID'] = 12 * (
            output_first.Year.astype(int) - 2015) + output_first.Month.astype(
    int)

quote_summary = output_first.loc[:, ('CountryCode', 'ChannelID',
                                     'CustomerNumber', 'WinLoss',
                                     'QuoteMonthID')]

com_quote_price = OutputData.loc[:, ('QuoteID', 'ComQuotePrice')].groupby(
    'QuoteID').sum().rename(
    columns={'ComQuotePrice': 'TotalQuoteQuotedPrice'})
total_components = OutputData.loc[:, ('QuoteID', 'ComQuotePrice')].rename(
    columns={'ComQuotePrice': 'TotalQuoteComCount'})
total_components = total_components.groupby('QuoteID').count()

# New product map -- Yuxi Edited 12/16/17
rev_div_cd = OutputData.loc[:,
             ('QuoteID', 'ComRevDivCd', 'ComQuotePrice')].groupby(
        ('QuoteID', 'ComRevDivCd')).sum().reset_index()
rev_div_cond = rev_div_cd.groupby('QuoteID')['ComQuotePrice'].transform(
    max) == rev_div_cd['ComQuotePrice']
rev_div_cd = rev_div_cd[rev_div_cond].set_index('QuoteID').rename(
    columns={'ComRevDivCd': 'LeadRevDivCd',
             'ComQuotePrice': 'LeadRevDivCdQuotePrice'})
quote_summary = pd.concat(
        [quote_summary, com_quote_price, total_components, rev_div_cd],
        axis=1)
# level4 = OutputData.loc[:, ('QuoteID', 'LVL4', 'ComQuotePrice')].groupby(('QuoteID', 'LVL4')).sum().reset_index()
# level4_cond = level4.groupby('QuoteID')['ComQuotePrice'].transform(max) == level4['ComQuotePrice']
# level4 = level4[level4_cond].set_index('QuoteID').rename(columns={'LVL4': 'LeadLevel4', 'ComQuotePrice': 'LeadLevel4QuotePrice'})
# quote_summary = pd.concat([quote_summary, com_quote_price, total_components, level4], axis=1)


quote_summary['DeleteCount'] = 1
quote_summary['Delete'] = 0

# New product map -- Yuxi Chen Edited 12/16/17
quote_summary = quote_summary.reset_index().sort_values(
        ['CountryCode', 'CustomerNumber', 'ChannelID', 'LeadRevDivCd',
         'QuoteMonthID', 'QuoteID'])
# quote_summary = quote_summary.reset_index().sort_values(['CountryCode', 'CustomerNumber', 'ChannelID', 'LeadLevel4', 'QuoteMonthID', 'QuoteID'])

w1 = w2 = w3 = w4 = 0.25

# New product map -- Yuxi Chen Edite 12/16/17
grouped = quote_summary.groupby(
        ('CountryCode', 'CustomerNumber', 'ChannelID', 'LeadRevDivCd'))
# grouped = quote_summary.groupby(('CountryCode', 'CustomerNumber', 'ChannelID', 'LeadLevel4'))
for name, group in grouped:
    if len(group) == 1:
        continue

    base_list = []
    for i in range(len(group.index)):
        base_quote = pd.Series()
        base_index = ''
        counter = 0

        for index, row in group.iterrows():

            counter += 1
            if row['WinLoss'] == 1:
                continue

            if not base_index:
                if counter in base_list:
                    continue
                base_list.append(counter)
                base_index = index
                base_quote = row
                continue

            test_quote = row
            # New product map -- Yuxi Chen Edite 12/16/17
            # the following score indicates how different two quotes are.  A score of zero indicates identical quotes
            score = math.sqrt(w1 * (
                        float(test_quote['TotalQuoteQuotedPrice']) / float(
                        base_quote['TotalQuoteQuotedPrice']) - 1) ** 2 \
                              + w2 * (float(
                    test_quote['LeadRevDivCdQuotePrice']) / float(
                    base_quote['LeadRevDivCdQuotePrice']) - 1) ** 2 \
                              + w3 * ((float(
                    test_quote['TotalQuoteComCount']) + 10) / (float(
                    base_quote['TotalQuoteComCount']) + 10) - 1) ** 2 \
                              + w4 * ((float(
                    test_quote['QuoteMonthID']) - float(
                    base_quote['QuoteMonthID'])) / 12) ** 2)
            # score = math.sqrt( w1 * (float(test_quote['TotalQuoteQuotedPrice']) / float(base_quote['TotalQuoteQuotedPrice']) - 1) ** 2 \
            #                 + w2 * (float(test_quote['LeadLevel4QuotePrice']) / float(base_quote['LeadLevel4QuotePrice']) - 1) ** 2 \
            #                 + w3 * ((float(test_quote['TotalQuoteComCount']) + 10) / (float(base_quote['TotalQuoteComCount']) + 10) - 1) ** 2 \
            #                 + w4 * ((float(test_quote['QuoteMonthID']) - float(base_quote['QuoteMonthID'])) / 12) ** 2)

            if score < dup_cutoff:  # a score below the dup_cutoff indicates that the older quote should be removed
                quote_summary.ix[base_index, 'Delete'] = 1
                quote_summary.ix[index, 'DeleteCount'] += quote_summary.ix[
                    base_index, 'DeleteCount']
                break

removed_quotes = quote_summary[quote_summary['Delete'] == 1]['QuoteID']
OutputData = OutputData[~OutputData.QuoteID.isin(removed_quotes)]

###### NA training dataset is pulling from the right table source so we don't need to replace division codes with power/storage indicator for NA at this moment
## Replacing '2D','2K','2W','U5','Y4','72' values in ComRevDivCd with 'Storage'
# OutputData.loc[OutputData.ComRevDivCd.isin(['2D','2K','2W','U5','Y4','72']),'ComRevDivCd'] = 'Storage'
## Replacing non storage values in ComRevDivCd with 'Power'
# OutputData.loc[~(OutputData.ComRevDivCd == 'Storage'),'ComRevDivCd'] = 'Power'

quote_summary.to_csv(data_path + QuotesOut)
print('          >duplicate loss records removed')
report_df = report_update(report_df, OutputData,
                          'Remove duplicate loss records')

# ----------------------------------------------------------------------------
# SECTION E
# sorting OutputData
# New product map, Yuxi Chen Edited 12/16/17
OutputData.sort_values(['QuoteID', 'ComRevCat', 'ComMTM'], inplace=True)
# OutputData.sort_values(['QuoteID', 'ComRevCat', 'LVL1'], inplace=True)
# resetting the index
OutputData.reset_index(drop=True, inplace=True)
print('      Number of output records:     '), len(OutputData.index)

####Keep end of quarter quotes only
# OutputData = OutputData[OutputData['EndOfQtr']==1]

# Exporting the final OutputData to csv file
OutputData.to_csv(data_path + file_out, index=False)
print('    Writing historical data output file:')
print('      ', file_out)

# Add summary lines for the report_df DataFrame
summary_data = pd.DataFrame()
summary_data['CountryCode'] = OutputData['CountryCode']
summary_data['QuoteDate'] = OutputData['Month'].astype(str) + '-' + \
                            OutputData['Year'].astype('str')
summary_data['QuoteDate'] = pd.to_datetime(summary_data['QuoteDate'])

min_date = summary_data.groupby('CountryCode').min()
max_date = summary_data.groupby('CountryCode').max()
min_date['QuoteDate'] = min_date['QuoteDate'].dt.month.map(str) + '-' + \
                        min_date['QuoteDate'].dt.year.map(str)
max_date['QuoteDate'] = max_date['QuoteDate'].dt.month.map(str) + '-' + \
                        max_date['QuoteDate'].dt.year.map(str)
min_date.loc['Total'] = ''
max_date.loc['Total'] = ''
min_date = min_date.rename(columns={'QuoteDate': 'Oldest Transaction Date'}).T
max_date = max_date.rename(
    columns={'QuoteDate': 'Youngest Transaction Date'}).T
summary_headers = report_df.columns[:len(report_df.columns) / 3]

total_clean_win = OutputData.loc[:, ('CountryCode', 'WinLoss')].groupby(
    'CountryCode').sum()
total_clean_win.loc['Total'] = total_clean_win.sum()

win_loss_pct_values = match_df(total_clean_win.loc[:, 'WinLoss'],
                               report_df.ix[-1][summary_headers])
clean_win_percent = (win_loss_pct_values['df1'] / win_loss_pct_values[
    'df2']) * 100
total_cleaned_quotes = OutputData.loc[:,
                       ('QuoteID', 'CountryCode', 'WinLoss')].groupby(
        ('CountryCode', 'QuoteID')).count().reset_index().loc[:,
                       ('CountryCode', 'WinLoss')].groupby(
    'CountryCode').count()
total_cleaned_quotes.loc['Total'] = total_cleaned_quotes.sum()
cleaned_win_quotes = OutputData.loc[:,
                     ('CountryCode', 'WinLoss', 'QuoteID')].groupby(
        ('CountryCode', 'QuoteID')).sum().reset_index().loc[:,
                     ('CountryCode', 'WinLoss')]
cleaned_win_quotes['WinLoss'] = np.where((cleaned_win_quotes['WinLoss'] > 0),
                                         1, 0)
cleaned_win_quotes = cleaned_win_quotes.groupby('CountryCode').sum()
cleaned_win_quotes.loc['Total'] = cleaned_win_quotes.sum()
cleaned_win_percent = (cleaned_win_quotes.astype(
    float) / total_cleaned_quotes.astype(float)) * 100
cleaned_win_percent = cleaned_win_percent.applymap(lambda x: '%.2f%%' % x)
total_clean_win = total_clean_win.rename(
    columns={'WinLoss': 'Total # of cleaned win records'}).T
total_cleaned_quotes = total_cleaned_quotes.rename(
    columns={'WinLoss': 'Total # of cleaned quotes'})
cleaned_win_quotes = cleaned_win_quotes.rename(
    columns={'WinLoss': 'Total # of cleaned win quotes'})
cleaned_win_percent = cleaned_win_percent.rename(
    columns={'WinLoss': 'Cleaned win quote percent'})
total_clean_win = total_clean_win.applymap(lambda x: "{:,}".format(x))
total_cleaned_quotes = total_cleaned_quotes.applymap(
    lambda x: "{:,}".format(x))
cleaned_win_quotes = cleaned_win_quotes.applymap(lambda x: "{:,}".format(x))
cwp_text = 'Cleaned win record percent'
total_clean_win.loc[cwp_text] = clean_win_percent.values
total_clean_win.loc[cwp_text] = total_clean_win.loc[cwp_text].apply(
    lambda x: '%.2f%%' % x)
summary_stats = pd.concat(
        [min_date, max_date, total_clean_win, total_cleaned_quotes.T,
         cleaned_win_quotes.T, cleaned_win_percent.T])
summary_stats = insert_missing_cols(summary_stats, summary_headers)
for cols in report_df.columns[len(report_df.columns) / 3:]:
    summary_stats[cols] = ''

# Attach summary lines to the report_df DataFrame
num_columns = report_df.columns[:len(report_df.columns) / 3 * 2]
report_df[num_columns] = report_df[num_columns].astype(int)
report_df[num_columns] = report_df[num_columns].applymap(
    lambda x: "{:,}".format(x))
report_df = pd.concat([report_df, summary_stats])

# Export the report_df DataFrame to csv file
report_df.to_csv(data_path + analysis_out)
print('    Writing cleaning analysis file:')
print('      ', analysis_out)

# Display the run time
toc = time.time()
print('    Processing time (seconds):      ', round(toc - tic, 2))

print('->Data cleansing ends:')
