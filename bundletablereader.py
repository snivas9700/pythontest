import pandas as pd
import numpy as np


class bundletablereader:

    """
    Read the storage groupings file and convert it into pandas-friendly format
    created by: Namrata Sarkar
    created on : December 7, 2017
    """
    def __init__(self):
        pass
    
    def read(self, filename, data_path):
        # read the storage groupings table
        bundletable = pd.read_csv(data_path + "/" + filename)

        # formatting the table into pandas-friendly format
        bundletable = bundletable.dropna(axis=1, how="all").iloc[:, 0:5].reset_index(
            drop=True)  # dropping all NA columns
        headerloc = np.where(bundletable.iloc[:, 0] == 'ComMTM')[0]
       # extracting the column headers if the file is in unformatted style
        if(headerloc.tolist() != []):
            columnnames = bundletable.iloc[headerloc, :].reset_index(
                drop=True).values.tolist()
        # converting the unformatted table into pandas friendly format
            bundletable = bundletable.iloc[headerloc[0] +
                                         1:, :].reset_index(drop=True)
            bundletable.columns = columnnames  # setting the column header
        return bundletable
