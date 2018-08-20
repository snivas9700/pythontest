# -*- coding: utf-8 -*-
"""
Created on Mon Dec 4 12:04:06 2017

@author: SaiNaveen

Read sub region geo mapping table to pandas.df.
"""

import pandas as pd

class GeoMapTableReader:
    """Read sub region geo mapping table to pandas.df.
    """

    def __init__(self):
        pass
        #self._geomap = pd.DataFrame()

    def read(self, geomap_path):
        """Returning a contry sub region geo mapping data frame
        :param table_file: path of the sub region geo mapping table
        :retur a new pandas.df
        """

        self._geomap = pd.read_csv(geomap_path, low_memory=False)
        return self._geomap
