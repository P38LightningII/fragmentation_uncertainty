""" Unit Tests for Unscented Transform Functions """
import unittest
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from fragmentation_uncertainty import *
import pdb


class TestUnscentedTransformFunctions(unittest.TestCase):
    def test_vimpel_covariance_matrix(self):
        sigma_r = [np.array([50,32,17])]
        sigma_v = 0.001
        covariance_matrix = get_covariance_from_noise(1,sigma_r,sigma_v,file_format='Vimpel')
        cov_mat_expected = np.array([[2500, 0, 0, 0, 0, 0],
                                     [0, 1024, 0, 0, 0, 0],
                                     [0, 0, 289, 0, 0, 0],
                                     [0, 0, 0, 1.0e-06, 0, 0],
                                     [0, 0, 0, 0, 1.0e-06, 0],
                                     [0, 0, 0, 0, 0, 1.0e-06]])

        check_posdef = is_pos_def(covariance_matrix[0])  # check if positive-definite
        self.assertTrue(True, check_posdef)
        np.testing.assert_array_almost_equal(cov_mat_expected, covariance_matrix[0])

    
    def test_tle_covariance_matrix(self):
        sigma_r = [np.array([50,32,17])]
        sigma_v = [np.array([0.02,0.005,0.001])]
        covariance_matrix = get_covariance_from_noise(1,sigma_r,sigma_v,file_format='TLE')
        cov_mat_expected = np.array([[2500, 0, 0, 0, 0, 0],
                                     [0, 1024, 0, 0, 0, 0],
                                     [0, 0, 289, 0, 0, 0],
                                     [0, 0, 0, 0.0004, 0, 0],
                                     [0, 0, 0, 0, 2.5e-05, 0],
                                     [0, 0, 0, 0, 0, 1.0e-06]])

        check_posdef = is_pos_def(covariance_matrix[0])  # check if positive-definite
        self.assertTrue(True, check_posdef)
        np.testing.assert_array_almost_equal(cov_mat_expected, covariance_matrix[0])


    def test_covariance_matrix_storage(self):
        sigma_r = [np.array([0.33747142, 0.23807244, 0.94467645]), np.array([14.92790254, 10.53103152, 41.78735477]), np.array([17.88598537, 12.61783933, 50.06785208]), np.array([0.67494284, 0.47614488, 1.88935291])]
        sigma_v = 0.001
        covariance_matrix = get_covariance_from_noise(len(sigma_r),sigma_r,sigma_v,file_format='Vimpel')
        cov_mat_expected0 = np.array([[0.11388695931681642, 0, 0, 0, 0, 0],
                                      [0, 0.0566784866875536, 0, 0, 0, 0],
                                      [0, 0, 0.8924135951846025, 0, 0, 0],
                                      [0, 0, 0, 1.0e-06, 0, 0],
                                      [0, 0, 0, 0, 1.0e-06, 0],
                                      [0, 0, 0, 0, 0, 1.0e-06]])
        cov_mat_expected2 = np.array([[319.90847265585404, 0, 0, 0, 0, 0],
                                      [0, 159.20986935769486, 0, 0, 0, 0],
                                      [0, 0, 2506.7898119047604, 0, 0, 0],
                                      [0, 0, 0, 1.0e-06, 0, 0],
                                      [0, 0, 0, 0, 1.0e-06, 0],
                                      [0, 0, 0, 0, 0, 1.0e-06]])

        check_posdef0 = is_pos_def(covariance_matrix[0])  # check if positive-definite
        check_posdef2 = is_pos_def(covariance_matrix[2])
        self.assertTrue(True, check_posdef0)
        self.assertTrue(True, check_posdef2)
        np.testing.assert_array_almost_equal(cov_mat_expected0, covariance_matrix[0])
        np.testing.assert_array_almost_equal(cov_mat_expected2, covariance_matrix[2])

    
    def test_sigma_points_default(self):
        r = np.array([0,0,0])
        v = np.array([0.1,0,0])
        sigma_r = 0.5
        sigma_v = 0.001
        mean = [np.hstack((r, v))]
        covariance = [block_diag((sigma_r**2)*np.eye(3), (sigma_v**2)*np.eye(3))]
        s_num, sigma_points, weight_mean, weight_cov = generate_sigma_points(1, 6, mean, covariance)

        sig_expected0 = mean[0]
        sig_expected2 = np.array([0, 0.12247449, 0, 0.1, 0, 0])
        sig_expected10 = np.array([0, 0, 0, 0.09975505, 0, 0])
        np.testing.assert_array_almost_equal(sig_expected0, sigma_points[0][0])
        np.testing.assert_array_almost_equal(sig_expected2, sigma_points[0][2])
        np.testing.assert_array_almost_equal(sig_expected10, sigma_points[0][10])
        np.testing.assert_almost_equal(-99, weight_mean[0])
        np.testing.assert_almost_equal(8.33333333, weight_mean[4])
        np.testing.assert_almost_equal(-96.01, weight_cov[0])
        # pdb.set_trace()

        ax = plt.figure().add_subplot(projection="3d")
        plt.plot(sigma_points[0][:,0],sigma_points[0][:,1],sigma_points[0][:,2],'.')
        ax.set_aspect("equal")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def test_sigma_points_vary_kappa_alpha(self):
        r = np.array([0,0,0])
        v = np.array([0.1,0,0])
        sigma_r = 0.5
        sigma_v = 0.001
        mean = [np.hstack((r, v))]
        covariance = [block_diag((sigma_r**2)*np.eye(3), (sigma_v**2)*np.eye(3))]

        ax = plt.figure().add_subplot(projection="3d")
        for kappa in range(-3,3,1):
            s_num, sigma_points, weight_mean, weight_cov = generate_sigma_points(1, 6, mean, covariance, kappa=kappa)
            plt.plot(sigma_points[0][:,0],sigma_points[0][:,1],sigma_points[0][:,2],'.',label=r"$\kappa$ = {}".format(kappa))
        ax.set_aspect("equal")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()

        ax = plt.figure().add_subplot(projection="3d")
        alpha = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
        for i in range(len(alpha)):
            s_num, sigma_points, weight_mean, weight_cov = generate_sigma_points(1, 6, mean, covariance, alpha=alpha[i])
            plt.plot(sigma_points[0][:,0],sigma_points[0][:,1],sigma_points[0][:,2],'.',label=r"$\alpha$ = {}".format(alpha[i]))
        ax.set_aspect("equal")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    unittest.main()
