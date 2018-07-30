# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 06:04:09 2017

@author: Ravindranath
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 16:53:25 2017

@author: Ravindranath
"""

import sys
sys.path.append('C:/ModelFactory/')

import unittest
import numpy as np
import pandas as pd
from prodconsolidator import ProdConsolidator
# for testing dataframes
from pandas.util.testing import assert_frame_equal

#from pandas.api.types import is_numeric_dtype


class prodconsolidatorTester(unittest.TestCase):
    """
    Class to unit test the prodconsolidator class
    """

    def test_consolidatorfilter(self):
        """
        Test that the dataframe read in equals what you expect
        """
        input = pd.read_csv('C:/Users/IBM_ADMIN/Desktop/1XN/unittestdata.csv')

        expected = pd.read_csv(
            'C:/Users/IBM_ADMIN/Desktop/1XN/quote_df_response_new.csv')

        pc = ProdConsolidator(input)
        test_result = pc.consolidate(input)
        #input.to_csv('C:/ModelFactory/input.csv')
        #test_result.to_csv('C:/ModelFactory/out1.csv')
        #expected.to_csv('C:/ModelFactory/out2.csv')
        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        test_result.to_csv('C:/Users/IBM_ADMIN/Desktop/1XN/test_result.csv')
        '''        
        for i in range(50):
            print type(expected.iloc[:, i][0])
            print type(test_result.iloc[:, i][0])
            print 'hello'
        '''
        test_result['concat0'] = pd.Series(
            test_result.fillna('').values.tolist())
        expected['concat0'] = pd.Series(expected.fillna('').values.tolist())
        #print df

        for i in range(len(test_result)):
            test_result.ix[i, 'concat'] = ''.join(
                str(elt) for elt in test_result.ix[i, 'concat0'])
        for i in range(len(test_result)):
            expected.ix[i, 'concat'] = ''.join(
                str(elt) for elt in expected.ix[i, 'concat0'])

        test_result1 = pd.DataFrame(test_result['concat'])
        test_result1['concat'] = '0' + test_result1['concat']
        expected1 = pd.DataFrame(expected['concat'])

        for i in range(len(test_result1)):
            test_result1.ix[i, 'concat'] = test_result1.ix[i, 'concat'].strip()
            test_result1.ix[i, 'concat'] = test_result1.ix[
                i, 'concat'].replace(" ", "")

        for i in range(len(test_result1)):
            expected1.ix[i, 'concat'] = expected1.ix[i, 'concat'].strip()
            expected1.ix[i, 'concat'] = expected1.ix[i, 'concat'].replace(
                " ", "")

        assert_frame_equal(
            expected1.reindex_axis(sorted(expected1.columns), axis=1),
            test_result1.reindex_axis(sorted(expected1.columns), axis=1))


if __name__ == '__main__':
    unittest.main()
