# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 05:12:26 2018

@author: harish
"""

import pandas as pd
from config import VALIDATORPRODTABLE_NA,VALIDATORPRODTABLE_JP,VALIDATORPRODTABLE_EMEA

class HWReader:
    """Read HW validator products csv into pandas.df
    """

    def __init__(self):
        pass

    def read_table(self, region_geo):
        """Returning a validator data frame for the region specified
        :param table_file: region (NA/JP/EMEA)
        :return a new pandas.df
        """
        if (region_geo == 'NA'):
            validator_table_path =  VALIDATORPRODTABLE_NA
        elif (region_geo == 'JP'):
            validator_table_path =  VALIDATORPRODTABLE_JP
        else:
            validator_table_path =  VALIDATORPRODTABLE_EMEA
        
        validator_rules_na = pd.read_csv(validator_table_path, low_memory=False)
        
        return validator_rules_na
