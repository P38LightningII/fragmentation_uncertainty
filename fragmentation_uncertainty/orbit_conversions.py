""" Database of functions related to orbital parameters and conversions
#   Functions:
#   > coes2rv - convert from classical orbital elements to position & velocity in ECI frame
#   > eci2perif - convert from ECI (Earth centered inertial) coordinates to perifocal (perigee, semi-latus rectum, and
#                 angular momentum)
#   > ecc_anomaly - calculate eccentric anomaly [rad]
#   > true_anomaly - calculate true anomaly from eccentricity and eccentric anomaly
#   > rv2coes - convert from position & velocity in ECI frame to classical orbital elements

================================================================================"""
# Written by: Arly Black, 2021
# Revised: Feb 2024

import datetime
import numpy as np
from numpy.linalg import norm
import planetary_data as pl
# from planetary_data.constants import earth  ## SORT THIS OUT!
import pdb


def coes2rv(coes:list, deg=False, mean_anom=False, mu=pl.earth['mu'])->np.ndarray:
    """Convert from classical orbital elements to position & velocity in ECI frame

    :param coes: classical orbital elements: [semi-major axis, eccentricity, inclination, 
                 right ascension of the ascending node, argument of perigee, true or mean anomaly]
    :type coes: list
    :param deg: option to input angles in degrees or radians - True = degrees, False = radians, defaults to False
    :type deg: bool, optional
    :param mean_anom: option to provide true anomaly or mean anomaly - True = mean anomaly, False = true anomaly, defaults to False
    :type mean_anom: bool, optional
    :param mu: gravitational constant, defaults to pl.earth['mu']
    :type mu: float, optional
    :return: [km, km/s] position and velocity vectors 
    :rtype: np.ndarray
    """
    d2r = np.deg2rad  # deg to rad
    if deg:  # angles in degrees
        if mean_anom:  # mean anomaly provided
            sma, ecc, inc, raan, aop, ma = coes  # [km, -, deg, deg, deg, deg] 
            ma *= d2r  # mean anomaly [rad]
        else:  # true anomaly provided
            sma, ecc, inc, raan, aop, ta = coes  #  [km, -, deg, deg, deg, deg] 
            ta *= d2r  # true anomaly [rad]
        inc *= d2r   # [rad]
        aop *= d2r   # [rad]
        raan *= d2r  # [rad]
    else:  # angles in radians
        if mean_anom:  # mean anomaly provided
            sma, ecc, inc, raan, aop, ma = coes  # [km, -, rad, rad, rad, rad] 
        else:  # true anomaly provided
            sma, ecc, inc, raan, aop, ta = coes  # [km, -, rad, rad, rad, rad] 

    if mean_anom:  # mean anomaly provided
        ecc_anom = ecc_anomaly([ma, ecc], mean_anom=True)
        ta = true_anomaly([ecc_anom, ecc])
    else:  # true anomaly provided
        ecc_anom = ecc_anomaly([ta, ecc], mean_anom=False)
    r_norm = sma*(1-ecc**2)/(1+ecc*np.cos(ta))  # length of r

    # calculate r and v vectors in perifocal frame
    r_perif = r_norm*np.array([np.cos(ta), np.sin(ta), 0])
    v_perif = np.sqrt(mu*sma)/r_norm*np.array([-np.sin(ecc_anom), np.cos(ecc_anom)*np.sqrt(1-ecc**2), 0])

    # rotation matrix from perifocal to ECI
    perif2eci = np.transpose(eci_to_perif(raan, aop, inc))
    pdb.set_trace()

    # calculate r and v vectors in inertial frame
    r = np.dot(perif2eci, r_perif)
    v = np.dot(perif2eci, v_perif)
    return r, v  # [km, km/s]


def ecc_anomaly(arr:list, mean_anom=True, tol=1e-8)->float:
    """Calculate eccentric anomaly

    :param arr: [rad] list with mean anomaly or true anomaly and eccentricity
    :type arr: list
    :param mean_anom: option to provide true anomaly or mean anomaly - True = mean anomaly, False = true anomaly, defaults to True
    :type mean_anom: bool, optional
    :param tol: tolerance for newton method - only applies if mean anomaly is given, defaults to 1e-8
    :type tol: float, optional
    :return: [rad] eccentric anomaly
    :rtype: float
    """
    if mean_anom:
        # Newton's method for iteratively finding eccentric anomaly
        ma, ecc = arr
        if ma < np.pi/2.0:
            ecc_anom0 = ma + ecc/2.0
        else:
            ecc_anom0 = ma - ecc
        for n in range(200):  # arbitrary max number of steps
            ratio = (ecc_anom0 - ecc*np.sin(ecc_anom0) - ma) / (1 - ecc*np.cos(ecc_anom0))
            if abs(ratio) < tol:
                if n == 0:
                    return ecc_anom0
                else:
                    return ecc_anom1
            else:
                ecc_anom1 = ecc_anom0-ratio
                ecc_anom0 = ecc_anom1
        # did not converge
        raise RuntimeError('Newton method for finding eccentric anomaly did not converge')
    else:
        ta, ecc = arr
        return 2 * np.arctan(np.sqrt((1-ecc)/(1+ecc)) * np.tan(ta/2.0))


def true_anomaly(ecc_anom:float, ecc:float)->float:
    """Find true anomaly

    :param ecc_anom: [rad] eccentric anomaly
    :type ecc_anom: float
    :param ecc: [rad] eccentricity
    :type ecc: float
    :return: [rad] true anomaly
    :rtype: float
    """
    return 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(ecc_anom/2.0))


def rv2coes(r:np.ndarray, v:np.ndarray, mu=pl.earth['mu'])->list:
    """Convert from position & velocity in ECI frame to classical orbital elements

    :param r: [km] position vector
    :type r: np.ndarray
    :param v: [km/s] velocity vector
    :type v: np.ndarray
    :param mu: gravitational constant, defaults to pl.earth['mu']
    :type mu: float, optional
    :return: classical orbital elements: [semi-major axis, eccentricity, inclination, 
             right ascension of the ascending node, argument of perigee, true or mean anomaly] (angles in rad)
    :rtype: list
    """
    r_mag = norm(r)           # [km]   position magnitude
    v_mag = norm(v)           # [km/s] velocity magnitude
    h_vec = np.cross(r, v)              # [km2/s] angular momentum vector 
    h_mag = norm(h_vec)       # [km2/s] angular momentum magnitude
    n_vec = np.cross([0, 0, 1], h_vec)  # line of nodes vector
    n_mag = norm(n_vec) 
    ecc_vec = np.cross(v, h_vec) / mu - r / r_mag  # eccentricity vector
    ecc = norm(ecc_vec)  # [-] eccentricity

    inc = np.arccos(h_vec[2] / h_mag)  # [rad] inclination
    raan = np.arccos(n_vec[0]/n_mag)  # [rad] right ascension of the ascending node
    aop = np.arccos(np.dot(n_vec, ecc_vec) / (n_mag * ecc))  # [rad] argument of periapsis
    ta = np.arccos(np.dot(ecc_vec, r) / (ecc * r_mag))  # [rad] true anomaly

    # Check if orbit is an ellipse
    eps = np.finfo(float).eps
    if np.abs(ecc-1.0) > eps:
        sma = 1/(2 / r_mag - v_mag ** 2 / mu)  # [km] semi-major axis 
    else:
        sma = np.inf
    # Check if orbit has wrapped around
    if n_vec[1] < 0:
        raan = 2*np.pi - raan
    # Check if orbit has wrapped around
    if np.dot(r, v) < 0:
        ta = 2*np.pi - ta
    if ecc_vec[2] < 0:
        aop = 2*np.pi - aop
    # Check if orbit is circular
    if ecc < 0.0001:
        ta = 0
        ecc = 0
        aop = 0

    coes = [sma, ecc, inc, raan, aop, ta]  # [km, -, rad, rad, rad, rad]
    return  coes 


def eci_to_perif(raan:float, inc:float, aop:float)-> np.ndarray:
    """ECI inertial to perifocal rotation matrix (eph)

    :param raan: [rad] right ascension of the ascending node
    :type raan: float
    :param inc: [rad] inclination
    :type inc: float
    :param aop: [rad] argument of perigee
    :type aop: float
    :return: 3x3 ECI to perifocal rotation matrix
    :rtype: np.ndarray
    """
    row0 = [-np.sin(raan)*np.cos(inc)*np.sin(aop) + np.cos(raan)*np.cos(aop),
            np.cos(raan)*np.cos(inc)*np.sin(aop) + np.sin(raan)*np.cos(aop), np.sin(inc)*np.sin(aop)]
    row1 = [-np.sin(raan)*np.cos(inc)*np.cos(aop) - np.cos(raan)*np.sin(aop),
            np.cos(raan)*np.cos(inc)*np.cos(aop) - np.sin(raan)*np.sin(aop), np.sin(inc)*np.cos(aop)]
    row2 = [np.sin(raan)*np.sin(inc), -np.cos(raan)*np.sin(inc), np.cos(inc)]
    return np.array([row0, row1, row2])


def eci_to_rdx(raan:float, inc:float, aop:float, ta:float)->np.ndarray:
    """Transformation matrix from ECI to radial/downrange/crossrange (RDX) frame
        * similar to ECI to perifocal frame, except x points to the object, instead of the periapse

    :param raan: [rad] right ascension of the ascending node
    :type raan: float
    :param inc: [rad] inclination
    :type inc: float
    :param aop: [rad] argument of perigee
    :type aop: float
    :param ta: [rad] true anomaly
    :type ta: float
    :return: 3x3 ECI to RDX transformation matrix
    :rtype: np.ndarray
    """
    arg_lat = aop + ta  # [rad] argument of latitude
    if (arg_lat % (2*np.pi)) > np.pi:
        inc = -inc
    row0 = [-np.sin(raan)*np.cos(inc)*np.sin(arg_lat) + np.cos(raan)*np.cos(arg_lat),
            np.cos(raan)*np.cos(inc)*np.sin(arg_lat) + np.sin(raan)*np.cos(arg_lat), np.sin(inc)*np.sin(arg_lat)]
    row1 = [-np.sin(raan)*np.cos(inc)*np.cos(arg_lat) - np.cos(raan)*np.sin(arg_lat),
            np.cos(raan)*np.cos(inc)*np.cos(arg_lat) - np.sin(raan)*np.sin(arg_lat), np.sin(inc)*np.cos(arg_lat)]
    row2 = [np.sin(raan)*np.sin(inc), -np.cos(raan)*np.sin(inc), np.cos(inc)]
    return np.array([row0, row1, row2])


def ico_to_rdx(r:np.ndarray, v:np.ndarray)->np.ndarray:
    """Transformation matrix from in-track/cross-track/out-of-plane (ICO) frame to radial/downrange/crossrange (RDX)

    :param r: [km] position vector
    :type r: np.ndarray
    :param v: [km/s] velocity vector
    :type v: np.ndarray
    :return: 3x3 ICO to RDX transformation matrix
    :rtype: np.ndarray
    """
    # RDX unit vectors
    rhat = r / norm(r)  # radial
    xhat = (np.cross(r, v)) / norm(np.cross(r, v))  # crossrange
    dhat = np.cross(xhat, rhat)  # downrange

    # ICO unit vectors
    ihat = v / norm(v)  # in-track = direction of velocity
    ohat = (np.cross(r, v)) / norm(np.cross(r, v))  # out-of-plane
    chat = np.cross(-ohat, ihat)  # cross-track
    # chat = np.cross(ihat, ohat)  # same thing

    theta = np.arccos(np.dot(dhat, ihat))  # [rad] rotation angle from ICO to RDX
    dcm_ico2rdx = np.array([[np.sin(theta), np.cos(theta), 0], [np.cos(theta), -np.sin(theta), 0], [0, 0, 1]])
    return dcm_ico2rdx


def xyz_to_rdx(r:np.ndarray, v:np.ndarray)->np.ndarray:
    """Transformation matrix from inertial xyz coordinate frame to radial/downrange/crossrange (RDX)

    :param r: [km] position vector 
    :type r: np.ndarray
    :param v: [km/s] velocity vector
    :type v: np.ndarray
    :return: 3x3 xyz to RDX transformation matrix
    :rtype: np.ndarray
    """
    rhat = r / norm(r)
    xhat = (np.cross(r, v)) / norm(np.cross(r, v))
    dhat = np.cross(xhat, rhat)

    dcm_xyz2rdx = np.vstack((rhat, dhat, xhat))  # transformation matrix from inertial to radial, downrange, crossrange
    return dcm_xyz2rdx


def xyz_to_ico(r:np.ndarray, v:np.ndarray)->np.ndarray:
    """Transformation matrix from inertial xyz coordinate frame to in-track/cross-track/out-of-plane (ICO)

    :param r: [km] position vector 
    :type r: np.ndarray
    :param v: [km/s] velocity vector
    :type v: np.ndarray
    :return: 3x3 xyz to ICO transformation matrix
    :rtype: np.ndarray
    """
    ihat = v / norm(v)  # in-track = direction of velocity
    ohat = (np.cross(r, v)) / norm(np.cross(r, v))  # out-of-plane
    chat = np.cross(-ohat, ihat)  # cross-track

    dcm_xyz2ico = np.vstack((ihat, chat, ohat))  # transformation matrix from inertial to in-track/cross-track/out-of-plane
    return dcm_xyz2ico
