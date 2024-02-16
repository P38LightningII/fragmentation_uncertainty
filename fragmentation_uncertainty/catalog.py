"""Module containing functions pertaining to TLE files
Functions include: extracting TLE file data and propagation of TLE files

================================================================================"""
# Written by: Arly Black

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday, SGP4_ERRORS
from datetime import timedelta, datetime
from scipy.integrate import odeint
from .orbit_conversions import rv2coes, coes2rv
import planetary_data as pl
import pdb


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
    dataframe = pd.DataFrame(columns=['satellite_name', 'line1', 'line2'])
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
    :return: dictionary of satellite data, epoch details, mean elements (angles in rad)
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
    d2r = np.pi/180
    i = float(line2[8:16]) * d2r                   # inclination [rad]
    o = float(line2[17:25]) * d2r                  # RAAN [rad]
    e = float(line2[26:33]) / 1e7                  # eccentricity [-]
    w = float(line2[34:42]) * d2r                  # argument of perigee [rad]
    ma = float(line2[43:51]) * d2r                 # mean anomaly [rad]
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


def extract_vimpel(vimpel_file:str, index=0)->dict:
    """Extract parameters from Vimpel space object data file

    :param vimpel_file: Vimpel dataset filename
    :type vimpel_file: str
    :param index: index of desired object in the file, defaults to 0
    :type index: int, optional
    :return: dictionary of vimpel data
    :rtype: dict
    """
    d2r = np.pi/180  # degree to radians conversion 
    # extract data from vimpel file
    df_vimpel = pd.read_csv(vimpel_file, header=None)   # create dataframe
    sma = df_vimpel[5][index]                           # semi-major axis [km]
    inc = df_vimpel[6][index] * d2r                     # inclination [rad]
    raan = df_vimpel[7][index] * d2r                    # right ascension [rad]
    ecc = df_vimpel[8][index]                           # eccentricity [-]
    arg_lat = df_vimpel[9][index] * d2r                 # argument of latitude [rad]
    arg_per = df_vimpel[10][index] * d2r                # argument of perigee [rad]
    amr = df_vimpel[11][index]                           # effective area-to-mass ratio [m2/kg]
    epoch = df_vimpel[3][index]                         # time corresponding to data [ddmmyyyy HHMMSS]
    true_anom = arg_lat - arg_per                       # true anomaly [rad]
    pos_std = df_vimpel[14][index]                      # position uncertainty in transverse direction [km]

    coes = sma, ecc, inc, raan, arg_per, true_anom      # collate the classical orbital elements

    # breakup epoch into components
    year = int(epoch[4:8])
    month = int(epoch[2:4])
    day = int(epoch[:2])
    hour = int(epoch[9:11])
    minute = int(epoch[11:13])
    second = int(epoch[13:])
    microsecond = 0
    if second == 60:
        second = 59
        microsecond = 999999
    epoch = datetime(year, month, day, hour, minute, second, microsecond)

    # create dictionary of vimpel data
    vimpel_data = {
        "coes": coes,       # [rad] classical orbital elements
        "epoch": epoch,     # [datetime]
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "min": minute,
        "sec": second,
        "mic": microsecond,
        "amr": amr,           # [m2/kg] area to mass ratio
        "std_r": pos_std    # [km] position uncertainty in transverse direction
    }
    return vimpel_data

