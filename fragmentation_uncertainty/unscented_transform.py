"""Module containing functions pertaining to the Unscented Transform

================================================================================"""
# Written by: Arly Black

import numpy as np
from scipy.linalg import block_diag


def get_covariance_from_noise(num_obj:float,sigma_r:float,sigma_v:float,file_format:str) -> np.ndarray:
    """Obtain covariance matrix given position and velocity noise parameters for a given file format

    :param num_obj: number of objects
    :type num_obj: float
    :param sigma_r: [km] position noise
    :type sigma_r: float
    :param sigma_v: [km/s] velocity noise
    :type sigma_v: float
    :param file_format: file format of the data, 'Vimpel' or 'TLE'
    :type file_format: str
    :raises Exception: _description_
    :return: _description_
    :rtype: np.ndarray
    """
    for k in range(num_obj):
        if file_format == 'Vimpel':
            cov_r = np.diag([sigma_r[k][0]**2, sigma_r[k][1]**2, sigma_r[k][2]**2])
            cov_v = sigma_v ** 2 * np.eye(3)
        elif file_format == "TLE":
            cov_r = np.diag([sigma_r[k][0] ** 2, sigma_r[k][1] ** 2, sigma_r[k][2] ** 2])
            cov_v = np.diag([sigma_v[k][0] ** 2, sigma_v[k][1] ** 2, sigma_v[k][2] ** 2])
            # cov_r = sigma_r ** 2 * np.eye(3)
        else:
            raise Exception("Not a valid file format (use Vimpel or TLE")
        covariance = block_diag(cov_r, cov_v)
    
    return covariance


def generate_sigma_points(num_obj:float, nx:float, mean:np.ndarray, covariance:np.ndarray, version='scaled', **kwargs)->float:
    """Determine sigma points for unscented transform analysis
    Based on Julier and Uhlmann 1995

    :param num_obj: number of objects in the investigation
    :type num_obj: float
    :param nx: number of elements in the state
    :type nx: float
    :param mean: mean state vector of each object [km, km/s]
    :type mean: np.ndarray
    :param covariance: covariance matrix of each object
    :type covariance: np.ndarray
    :param version: version of unscented transform - either original (Julier) or scaled (van de Merwe) , defaults to 'scaled'
    :type version: str, optional
    :return: _description_
    :rtype: float

    Optional inputs:
        kappa:  scaling parameter - determines spread of sigma points. K>=0 guarantees positive semi-definiteness of cov matrix, 
                defaults to 0
        alpha:  scaling parameter - controls size of sig point distribution. 1e-4 <= alpha <= 1, defaults to 0.1
        beta:   tuning parameter that controls the weight of zeroth sigma point, defaults to 2
    """
    # parse inputs
    kappa_ = kwargs.get('kappa', 0)    # scaling parameter - determines spread of sigma points. K>=0 guarantees positive semi-definiteness of cov matrix
    alpha_ = kwargs.get('alpha', 0.1)  # scaling parameter - controls size of sig point distribution. 1e-4 <= alpha <= 1
    beta_ = kwargs.get('beta', 2)   # tuning parameter that controls the weight of zeroth sigma point

    # define scaling parameters
    lambda_ = alpha_ ** 2 * (nx + kappa_) - nx  # scaling parameter
    if version=='scaled':
        c = nx + lambda_  # scaled unscented transform definition - fixes scaling issues
    else:
        c = nx + kappa_   # original definition

    s_num = 2 * nx + 1  # number of sigma points
    sigma_points = np.zeros((num_obj, s_num, nx))  # initialize sigma points
    for k in range(num_obj):
        A = np.sqrt(c) * np.linalg.cholesky(covariance)
        Y = mean[k, :] * np.ones((nx, 1))
        sigma_points[k, :, :] = np.vstack((mean[k, :], Y + A, Y - A))  # sigma points for each object

    # Mean weights
    weight_mean = np.zeros(s_num)
    if version=='scaled':
        weight_mean[0] = lambda_ / c
    else:
        weight_mean[0] = kappa_ / c
    weight_mean[1: s_num] = 1 / (2 * c)

    # Covariance weights
    weight_cov = np.zeros(s_num)
    if version=='scaled':
        weight_cov[0] = lambda_ / c + (1 - alpha_ ** 2 + beta_)
    else:
        weight_cov[0] = kappa_ / c
    weight_cov[1: s_num] = 1 / (2 * c)

    return s_num, sigma_points, weight_mean, weight_cov


def is_pos_def(A:np.ndarray):
    """Check if matrix is positive definite
    A real matrix is a covariance matrix if and only if it is symmetric positive semi-definite. A consequence of this
    is that all diagonal elements must be non-negative.

    :param A: matrix
    :type A: np.ndarray
    :return: True or False
    :rtype: _type_
    """
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

