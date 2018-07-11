"""
Created on Thu Dec 07 15:37:55 2017

@author: Chaitra
"""
import pandas as pd
import os
import sys 
import numpy as np
sys.path.insert(0,'.')
import pandasql
from pandasql import *
import unittest
from pandas.util.testing import assert_frame_equal
from bundledetector import BundleDetector

class Testbundledetector(unittest.TestCase):
    """
    Class to unit test the BundleDetector class
    """
    
    def test_detect(self):
        """
        unittests for the split bundle and nonbundle data
        """        
        data_path='./requirements/'
        bundle ='Bundle_Formated.csv'
        quote_data = 'NA_Training_input.csv' 
        bdl_df = pd.read_csv(data_path + bundle)
        qte_df = pd.read_csv(data_path + quote_data)
        
        bdl_df.reset_index(drop=True,inplace=True)
        qte_df.reset_index(drop=True,inplace=True)

        n = pd.read_csv(data_path + 'nonbundle_table.csv') # nonbundle reference data
        b = pd.read_csv(data_path + 'bundle_table.csv') # bundle reference data
        
        n.reset_index(drop=True,inplace=True)
        b.reset_index(drop=True,inplace=True)

        bn_df = BundleDetector(bdl_df,qte_df)
        b_df,n_df = bn_df.detect(bdl_df,qte_df) # b_df(bundle),n_df(nonbundle) output of def detect
        
        test_result = b_df
        test_result['concat0'] = pd.Series(test_result.fillna('').values.tolist())
        expected = b
        expected['concat0'] = pd.Series(expected.fillna('').values.tolist())

        for i in range(len(test_result)):
            test_result.ix[i, 'concat'] = ''.join(
                str(elt) for elt in test_result.ix[i, 'concat0'])
        for i in range(len(test_result)):
            expected.ix[i, 'concat'] = ''.join(
                str(elt) for elt in expected.ix[i, 'concat0'])

        test_result1 = pd.DataFrame(test_result['concat'])
        expected1 = pd.DataFrame(expected['concat'])

        for i in range(len(test_result1)):
            test_result1.ix[i, 'concat'] = test_result1.ix[i, 'concat'].strip()
            test_result1.ix[i, 'concat'] = test_result1.ix[i, 'concat'].replace(" ", "")

        for i in range(len(test_result1)):
            expected1.ix[i, 'concat'] = expected1.ix[i, 'concat'].strip()
            expected1.ix[i, 'concat'] = expected1.ix[i, 'concat'].replace(" ", "")

        assert_frame_equal(
            expected1.reindex_axis(sorted(expected1.columns), axis=1),
            test_result1.reindex_axis(sorted(test_result1.columns), axis=1))
 

        test_result_2 = n_df
        test_result_2['concat0'] = pd.Series(test_result_2.fillna('').values.tolist())
        expected_2 = n
        expected_2['concat0'] = pd.Series(expected_2.fillna('').values.tolist())

        for i in range(len(test_result_2)):
            test_result_2.ix[i, 'concat'] = ''.join(
                str(elt) for elt in test_result_2.ix[i, 'concat0'])
        for i in range(len(test_result_2)):
            expected_2.ix[i, 'concat'] = ''.join(
                str(elt) for elt in expected_2.ix[i, 'concat0'])

        test_result1_2 = pd.DataFrame(test_result_2['concat'])
        expected1_2 = pd.DataFrame(expected_2['concat'])

        for i in range(len(test_result1_2)):
            test_result1_2.ix[i, 'concat'] = test_result1_2.ix[i, 'concat'].strip()
            test_result1_2.ix[i, 'concat'] = test_result1_2.ix[i, 'concat'].replace(" ", "")

        for i in range(len(test_result1_2)):
            expected1_2.ix[i, 'concat'] = expected1_2.ix[i, 'concat'].strip()
            expected1_2.ix[i, 'concat'] = expected1_2.ix[i, 'concat'].replace(" ", "")

        assert_frame_equal(
            expected1_2.reindex_axis(sorted(expected1_2.columns), axis=1),
            test_result1_2.reindex_axis(sorted(test_result1_2.columns), axis=1))

if __name__ == '__main__':
    unittest.main()
    
