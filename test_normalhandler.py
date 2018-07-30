# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 16:53:25 2017

@author: Ravindranath
"""

import sys
sys.path.insert(0, '.')

import unittest
import numpy as np
import pandas as pd
import BasePricingFunctions as BPF
import PricingEngine as PricingEngine
from normalhandler import Normalhandler
# for testing dataframes
from pandas.util.testing import assert_frame_equal

#from pandas.api.types import is_numeric_dtype
#This is the path of the Deployed Model Rules file
data_path = '.'

#Load the Model Rules
rules_df = pd.read_csv(data_path + 'DeployedModelRules 2017-08-01.csv'
                       )  #this is the file name of the optimal pricing rules
rules_df = rules_df.rename(columns={'model_ID': 'modelId'})
rules_df.set_index('modelId', inplace=True)

#Load the Models
#the following builds statements from text that can be executed
for i in range(len(rules_df.index)):
    #the following tests the segmentation file name for illegal "-" characters
    myString = rules_df.ix[i, 'seg_model_file_name']
    try:
        myString.index('-')
        print 'DeployedModelRules File Error in row', i + 1
        print '  An illegal ' "'-'" ' character is in seg file name:', myString
        print
    except:
        pass
#the following reads in the segmentation model into its dataframe name
for i in range(len(rules_df.index)):
    etext = rules_df.ix[i,
                        'seg_model_file_name'] + "_df = pd.read_csv(data_path + '" + rules_df.ix[i,
                                                                                                 'seg_model_file_name'] + ".csv', index_col=0 " + ')' + ".fillna('')"
    #print etext
    exec (etext)
    #the following set the segmentation table index using first through last columns
    first = rules_df.ix[i, 'first']
    last = rules_df.ix[i, 'last']
    #print 'first, last: ',first, last
    etext = "ProdHList = " + rules_df.ix[i,
                                         'seg_model_file_name'] + "_df.columns.tolist()"
    #print ProdHList
    exec (etext)
    ProdHList = ProdHList[ProdHList.index(first):ProdHList.index(last) + 1]
    #print ProdHList
    etext = rules_df.ix[i,
                        'seg_model_file_name'] + "_df.set_index(['" + "', '".join(
                            str(x) for x in ProdHList) + "'], inplace=True)"
    #print etext
    exec (etext)


class prodconsolidatorTester(unittest.TestCase):
    """
    Class to unit test the prodconsolidator class
    """

    def test_normalhandler(self):
        """
        Test that the dataframe read in equals what you expect
        """
        input = pd.read_csv(
            data_path + 'quote_df_request.csv')

        expected = pd.read_csv(
            data_path + 'quote_df_out_EMEA.csv')

        nh = Normalhandler()
        test_result = nh.handle_EMEA('CHW_DE', rules_df, input, data_path, 0,
                                     0, 0)
        #input.to_csv('C:/ModelFactory/input.csv')
        #test_result.to_csv('C:/ModelFactory/out1.csv')
        #expected.to_csv('C:/ModelFactory/out2.csv')
        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        test_result['concat0'] = pd.Series(
            test_result.fillna('').values.tolist())
        expected['concat0'] = pd.Series(expected.fillna('').values.tolist())
        
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

        s1 = test_result1['concat'][0]
        s2 = expected1['concat'][0]
        diff = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        print diff
        
        assert_frame_equal(
            expected1.reindex_axis(sorted(expected1.columns), axis=1),
            test_result1.reindex_axis(sorted(expected1.columns), axis=1))


if __name__ == '__main__':
    unittest.main()
