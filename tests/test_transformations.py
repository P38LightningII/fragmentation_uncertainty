""" Unit Tests for Coordinate Transformations """
import unittest
import numpy as np
import matplotlib.pyplot as plt
from fragmentation_uncertainty import *
import pdb


class TestTransformationFunctions(unittest.TestCase):
    def test_eci_to_rdx(self):
        """Convert state in ECI coordinates to RDX (radial, downrange, crossrange) coordinates
        """
        r_eci = np.array([40203, 6799, 35])
        v_eci = np.array([0.578, 0.924, -0.886])
        r_rdx_expected = np.array([40773.87441732757, 7.52620188e-13, 2.95652391e-13])
        v_rdx_expected = np.array([0.723222907, 1.20407833, 5.48959761e-17])
        dcm = xyz_to_rdx(r_eci, v_eci)  # DCM = [r;d;x] = ^X[L]^R = ^R[C]^X
        r_rdx = dcm @ r_eci
        v_rdx = dcm @ v_eci
        np.testing.assert_array_almost_equal(r_rdx_expected, r_rdx)
        np.testing.assert_array_almost_equal(v_rdx_expected, v_rdx)

    
    def test_eci_to_ico(self):
        """Convert state in ECI coordinates to ICO (intrack, crossrange, out-of-plane) coordinates
        """
        r_eci = np.array([40203, 6799, 35])
        v_eci = np.array([0.578, 0.924, -0.886])
        r_ico_expected = np.array([20994.544713802323, 34953.36789581449, 2.95652391e-13])
        v_ico_expected = np.array([1.40458392, 1.17353970e-18, 5.48959761e-17])
        dcm = xyz_to_ico(r_eci, v_eci)  # DCM = [i;c;o] = ^X[L]^I = ^I[C]^X
        r_ico = dcm @ r_eci
        v_ico = dcm @ v_eci
        np.testing.assert_array_almost_equal(r_ico_expected, r_ico)
        np.testing.assert_array_almost_equal(v_ico_expected, v_ico)


    def test_ico_to_rdx(self):
        """Convert state in ICO (intrack, crossrange, out-of-plane) to RDX (radial, downrange, crossrange) coordinates
        """
        r_ico = np.array([20994.544713802323, 34953.36789581449, 0])
        v_ico = np.array([1.40458392, 0, 0])
        r_rdx_expected = np.array([40773.87441732757, 7.52620188e-13, 2.95652391e-13])
        v_rdx_expected = np.array([0.723222907, 1.20407833, 5.48959761e-17])
        dcm = ico_to_rdx(r_ico, v_ico)  # DCM = ^I[L]^R = ^R[C]^I
        r_rdx = dcm @ r_ico
        v_rdx = dcm @ v_ico
        np.testing.assert_array_almost_equal(r_rdx_expected, r_rdx)
        np.testing.assert_array_almost_equal(v_rdx_expected, v_rdx) 

    
    def test_rv_to_coes(self):
        r_eci = np.array([40203, 6799, 35])
        v_eci = np.array([0.578, 0.924, -0.886])
        coes = rv2coes(r_eci, v_eci)
        np.testing.assert_almost_equal(coes[0],22674.935274131894)
        np.testing.assert_almost_equal(coes[1],0.8563413528257094)
        np.testing.assert_almost_equal(coes[2],0.8276577425206031)
        np.testing.assert_almost_equal(coes[3],3.309912972058377)
        np.testing.assert_almost_equal(coes[4],0.10304453307522499)
        np.testing.assert_almost_equal(coes[5],3.0373823778612907)

    
    def test_eci_to_rdx_from_coes(self):
        """Convert state in ECI coordinates to RDX (radial, downrange, crossrange) coordinates using Keplerian elements
        """
        r_eci = np.array([40203, 6799, 35])
        v_eci = np.array([0.578, 0.924, -0.886])
        coes = rv2coes(r_eci, v_eci)
        r_rdx_expected = np.array([40773.87441732757, 7.52620188e-13, 2.95652391e-13])
        v_rdx_expected = np.array([0.723222907, 1.20407833, 5.48959761e-17])
        dcm = eci_to_rdx(coes[3], coes[2], coes[4], coes[5])
        r_rdx = dcm @ r_eci
        v_rdx = dcm @ v_eci
        np.testing.assert_array_almost_equal(r_rdx_expected, r_rdx)
        np.testing.assert_array_almost_equal(v_rdx_expected, v_rdx)


    def test_coes_to_rv(self):
        coes = [22674.935274131894,0.8563413528257094,0.8276577425206031,3.309912972058377,
                0.10304453307522499,3.0373823778612907]
        r, v = coes2rv(coes, deg=False, mean_anom=False)
        r_expected = np.array([40203, 6799, 35])
        v_expected = np.array([0.578, 0.924, -0.886])

        np.testing.assert_array_almost_equal(r_expected, r)
        np.testing.assert_array_almost_equal(v_expected, v)



if __name__ == "__main__":
    unittest.main()
