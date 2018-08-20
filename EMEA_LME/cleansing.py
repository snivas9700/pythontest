# Hardware-TSS pricing
# Aaron Slowey, Gloria Zhang, IBM Chief Analytics Office, 2018

import pandas as pd
# import pandas_profiling
pd.set_option('precision', 3)
pd.set_option('display.width', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('display.memory_usage', True)

# Check if path_out was created; if not, create a Results folder within
# path_in
# if not os.path.isdir(path_out):
#     os.mkdir(path_in + '/Results')
# if os.path.exists(path_out):
#     pass
# elif (~os.path.exists(path_out) & (path_out != '')):
#     try:
#         os.mkdir(path_out)
#     except WindowsError:
#         pass
# elif path_out == '':
#     try:
#         os.mkdir(os.path.join(path_in, 'Results'))
#     except WindowsError:
#         pass

# os.path.join(path_root, file_in)

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


def nospace_lowercase(df, trim_values=False, cols2rename={}, fields=[],
                      categorize=True):
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

