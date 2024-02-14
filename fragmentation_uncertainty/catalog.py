"""Module containing functions pertaining to TLE files
Functions include: extracting TLE file data and propagation of TLE files

================================================================================"""
# Written by: Arly Black

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday, SGP4_ERRORS
from datetime import timedelta, datetime
from scipy.integrate import odeint
from orbit_conversions import rv2coes, coes2rv
import planetary_data as pl


def tle_dataframe(tle_file:str)->pd.DataFrame:
    """Convert TLE details to dataframe format

    :param tle_file: three-line-element dataset filename
    :type tle_file: str
    :return: dataframe with TLE data
    :rtype: pd.DataFrame
    """
    tle = open(tle_file, 'r')  # open TLE file
    sat_name = []  # list containing satellite names
    line1 = []  # list containing first row of TLE
    line2 = []  # list containing second row of TLE

    ii = 1
    for line in tle:
        jj = ii
        if ii == 1:
            sat_name.append(line)
            jj = 2
        elif ii == 2:
            line1.append(line[:69])
            jj = 3
        elif ii == 3:
            line2.append(line[:69])
            jj = 1
        ii = jj

    # create a dataframe to gather data and fill in columns with TLE data
    dataframe = pd.DataFrame(columns=['Satellite_name', 'Line_1', 'Line_2'])
    dataframe.satellite_name = sat_name
    dataframe.line1 = line1
    dataframe.line2 = line2

    return dataframe


def extract_tle(line1,line2)->dict:
    """Extract parameters from two-line-element sets

    :param line1: First line in the TLE
    :type line1: _type_
    :param line2: Second line in the TLE
    :type line2: _type_
    :return: dictionary of satellite data, epoch details, mean elements
    :rtype: dict
    """
    satellite = Satrec.twoline2rv(line1, line2)
    if satellite.epochyr < 57:
        satellite.epochyr = satellite.epochyr + 2000
    else:
        satellite.epochyr = satellite.epochyr + 1900

    # TLE epoch datetime
    epoch = datetime(satellite.epochyr, 1, 1) + timedelta(satellite.epochdays - 1)

    # Extract mean elements from TLE
    i = float(line2[8:16])                         # inclination [deg]
    o = float(line2[17:25])                        # RAAN [deg]
    e = float(line2[26:33]) / 1e7                  # eccentricity [-]
    w = float(line2[34:42])                        # argument of perigee [deg]
    ma = float(line2[43:51])                       # mean anomaly [deg]
    n = float(line2[52:63]) * 2 * np.pi / 86400    # mean motion [rad/s]
    a = (pl.earth['mu'] / n ** 2) ** (1 / 3)       # semi-major axis [km]
    coes = [a, e, i, o, w, ma]

    # create dictionary of vimpel data
    tle_data = {
        'coes': coes,   # [rad] classical orbital elements
        'epoch': epoch,
        'satellite': satellite
    }
    return tle_data


def tle_to_state(coes:list):
    r, v = coes2rv(coes, deg=True, mean_anom=True)
    return r, v
    

def tle_propagation(tle_file:str, propagation_epoch:datetime):
    """Propagate TLE to desired epoch and extract osculating elements

    :param tle_file: name of TLE file
    :type tle_file: str
    :param propagation_epoch: epoch to propagate to
    :type propagation_epoch: datetime
    :raises RuntimeError: _description_
    :return: final position, velocity and osculating orbit elements
    :rtype: np.ndarray
    """
    df_tle = tle_dataframe(tle_file)  # dataframe of TLE data
    line1 = df_tle.line1[0]
    line2 = df_tle.line2[0]

    jd, fr = jday(propagation_epoch.year, propagation_epoch.month, propagation_epoch.day, propagation_epoch.hour, propagation_epoch.minute, propagation_epoch.second)
    tle_data = extract_tle(line1, line2)
    satellite = tle_data.satellite

    error_code, r, v = satellite.sgp4(jd, fr)  # propagate tle to given julian day
    if error_code != 0:
        raise RuntimeError(SGP4_ERRORS[error_code])
    # find osculating orbital elements at given julian day from r, v, mu
    coes = rv2coes(r, v)  # [km, -, rad, rad, rad, rad]

    return r, v, coes