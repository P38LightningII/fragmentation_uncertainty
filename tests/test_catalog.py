""" Unit Tests for Catalog Data """
import unittest
import numpy as np
import os
from datetime import datetime
from fragmentation_uncertainty import *
import pdb


class TestCatalogFunctions(unittest.TestCase):
    def test_extract_vimpel(self):
        vimpel_file = os.path.join(os.environ["FRAG_SRC"], 'catalog_files', 'orbits.20190401_09047B.txt')
        vimpel_data = extract_vimpel(vimpel_file)
        expected_sma = 27061.1
        expected_inc = 23.303 * np.pi/180
        expected_amr = 9.61e-03
        expected_std = 1
        expected_mon = 4

        np.testing.assert_almost_equal(expected_sma,vimpel_data['coes'][0])
        np.testing.assert_almost_equal(expected_inc,vimpel_data['coes'][2])
        np.testing.assert_almost_equal(expected_amr,vimpel_data['amr'])
        np.testing.assert_almost_equal(expected_std,vimpel_data['std_r'])
        np.testing.assert_almost_equal(expected_mon,vimpel_data['month'])


    def test_extract_vimpel_with_index(self):
        vimpel_file = os.path.join(os.environ["FRAG_SRC"], 'catalog_files', 'orbits.20190401_09047B.txt')
        vimpel_data = extract_vimpel(vimpel_file, index=4)
        expected_sma = 27067.9
        expected_inc = 23.277 * np.pi/180
        expected_amr = 3.03e-01
        expected_std = 33
        expected_mon = 4

        np.testing.assert_almost_equal(expected_sma,vimpel_data['coes'][0])
        np.testing.assert_almost_equal(expected_inc,vimpel_data['coes'][2])
        np.testing.assert_almost_equal(expected_amr,vimpel_data['amr'])
        np.testing.assert_almost_equal(expected_std,vimpel_data['std_r'])
        np.testing.assert_almost_equal(expected_mon,vimpel_data['month'])


    def test_extract_tle(self):
        tle_file = os.path.join(os.environ["FRAG_SRC"], 'catalog_files', 'fragm09047B_20200302tle.txt')
        # with open(tle_file, 'rb') as f:
        #     df = tle_dataframe(f.name)
        df = tle_dataframe(tle_file)
        tle_data = extract_tle(df.line1[0],df.line2[0])
        expected_ecc = 0.5198195
        expected_inc = 24.1647 * np.pi/180
        expected_epoch = datetime(2020, 2, 12, 0, 0, 0)

        np.testing.assert_almost_equal(expected_ecc,tle_data['coes'][1])
        np.testing.assert_almost_equal(expected_inc,tle_data['coes'][2])
        np.testing.assert_almost_equal(expected_epoch.timestamp(),tle_data['epoch'].timestamp())



if __name__ == "__main__":
    unittest.main()
