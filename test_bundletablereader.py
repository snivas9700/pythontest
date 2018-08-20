from bundletablereader import bundletablereader
import unittest
import pandas as pd
import os
from pandas.util.testing import assert_frame_equal


class test_bundletablereader(unittest.TestCase):

    """
    Created on Thu Dec 07 13:30:59 2017

    @author: Namrata Sarkar

    Unittest for bundletablereader.py
    """

    def test_read(self):
        data_path = os.path.dirname(__file__)
        # path where the imput file is present
        filename = "Copra storage groupings 20171101.xlsx"  # name of the input bundle file
        classobj = bundletablereader()
        testresult = classobj.read(filename, data_path)
        # test file with pandas-friendly formatted output file
        test_bundle = pd.read_excel(
            data_path + "/" + "test_result_bundle.xlsx")
        assert_frame_equal(testresult, test_bundle)


if __name__ == '__main__':
    unittest.main()
