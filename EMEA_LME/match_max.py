"""

"""
# ePricer data
subfolder4 = '/hardware/xlsx/'
files_hw_csv = os.listdir(path_in + subfolder3)
files_hw_excel = os.listdir(path_in + subfolder4)

# Load select columns from each file into a list of DataFrames
hw_cols = ['BRANDNAME', 'CHANNEL_ID', 'COM_CATEGORY', 'COM_LISTPRICE',
           'COMPONENTID', 'MODEL', 'PRODID', 'PRODUCT_CATEGORY',
           'SERIAL_NUM', 'TYPE']
df_hw_set1 = [pd.read_csv(path_in+subfolder3+file, usecols=hw_cols,
                          dtype={'SERIAL_NUM': 'category'},
                          encoding='latin-1') for file in files_hw_csv]

df_hw_set2 = [pd.read_excel(path_in+subfolder4+file, usecols=hw_cols,
              dtype={'SERIAL_NUM': 'category'}) for file in files_hw_excel]

# Test-load a single file
# df_hw_probe = pd.read_csv(path_in+subfolder3+files_hw[3], nrows=10,
#     encoding='latin-1')  # usecols=hw_cols

# Remove spurious columns
for i in range(len(df_hw_set1)):
    for field in df_hw_set1[i].columns:
        if df_hw_set1[i][field].isnull().sum() == len(df_hw_set1[i]):
            df_hw_set1[i].drop(field, axis=1, inplace=True)

task = 'Concatenate HW DataFrames'
df_hw = pd.concat((df_hw_set1[0], df_hw_set1[1], df_hw_set2[0], df_hw_set2[
    1]), axis=0)

# Check & record number of rows in DataFrame
m_hw_1 = len(df_hw)
instance_loss = 0
process_log = pd.DataFrame(
    [{'Task': task, 'Instances': m_hw_1, 'Gain': 0}])

df_hw = prep.nospace_lowercase(df_hw, cols2rename={
    'model': 'machine_model',
    'type': 'machine_type',
    'serial_num': 'serial_number',
    'com_listprice': 'p_hw_com_list'}, trim_values=True)

df_hw = df_hw[df_hw.serial_number != '']
df_hw.dropna(subset=['serial_number'], inplace=True)  # no NaNs

df_hw['serial5'] = df_hw['serial_number'].str[-5:]

df_hw[['serial_number', 'serial5']].to_excel(path_out+'hw_serials.xlsx',
                                             index=False)

df_hw.drop_duplicates(subset=['serial5'], inplace=True)

# Check that all serial5 values are in fact five characters
print('{:.0f} rows'.format(len(df_hw)))
df_hw.loc[:, 'sn_length'] = df_hw.serial5.str.count


hwmas = hwma[['serial5', 'p_list']]
hwmas[['serial5']].to_excel(
        path_out+'hwma_serials.xlsx', index=False)

mini_inner = pd.merge(hwmas, df_hw, on=['serial5'])
len(mini_inner)/len(hwmas)

# ----------------------------------------------------------------------------
hw_dups = df_hw[df_hw.duplicated(['serial5', 'machine_type',
                                  'machine_model'])]
len(df_hw_set1[0])
len(df_hw_set1[1])
len(df_hw_set2[0])
len(df_hw_set2[1])

len(df_hw_set1[0]) + len(df_hw_set1[1]) + len(df_hw_set2[0]) + len(
        df_hw_set2[1])

len(df_hw)

# ----------------------------------------------------------------------------
list(df_hw_sets[0])
list(df_hw_sets[1])
list(df_hw_sets[2])
list(df_hw_sets[3])

prep.write_fields(df_hw_sets[3], 'hw_set4', path_out)

df_hw_sets[3][0:10]

len(hwmas.serial5.unique())
len(df_hw.serial_number.unique())
len(mini_inner.serial5.unique())/len(hwmas.serial5.unique())
