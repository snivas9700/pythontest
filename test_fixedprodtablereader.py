# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 16:53:25 2017
@author: Sai
"""

import unittest
import pandas as pd
from fixedprodtablereader import FixedProdTableReader
from pandas.util.testing import assert_frame_equal


class fixedProdTableReaderTester(unittest.TestCase):
    """
    Class to unit test the FixedProdTableReader class
    """

    def test_fixed_prod_table_reader(self):
        """
        Test that the dataframe read in equals what you expect
        """

        expected_df = pd.DataFrame(columns=['Condition', 'DiscountedPrice', 'Discount', 'Comments'],
                             data = [["df.loc[df.ComMTM == '5639-RH7a']", "item['ComListPrice']*0.82", 0.18, "Discount 18%"]])

        test_result = FixedProdTableReader().read('test_FixedDiscounting_Rules.csv')
        """
        reindex columns to ensure that columns are sorted and comparison
        doesn't result in failure
        """

        assert_frame_equal(
            expected_df.reindex_axis(sorted(expected_df.columns), axis=1),
            test_result.reindex_axis(sorted(expected_df.columns), axis=1))
