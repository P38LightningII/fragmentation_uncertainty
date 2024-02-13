"""Module containing functions pertaining to Vimpel files
Functions include: extracting vimpel file data and propagation of vimpel files

================================================================================"""
# Written by: Arly Black

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sgp4.api import jday
from datetime import datetime
from orbit_conversions import coes2rv, rv2coes
from ode import two_body_ode_with_perturbations


def extract_vimpel(vimpel_file:str, index=0)->dict:
    """Extract parameters from Vimpel space object data file

    :param vimpel_file: Vimpel dataset filename
    :type vimpel_file: str
    :param index: index of desired object in the file, defaults to 0
    :type index: int, optional
    :return: dictionary of vimpel data
    :rtype: dict
    """
    d2r = np.deg2rad  # degree to radians conversion 
    # extract data from vimpel file
    df_vimpel = pd.read_csv(vimpel_file, header=None)   # create dataframe
    sma = df_vimpel[5][index]                           # semi-major axis [km]
    inc = df_vimpel[6][index] * d2r                     # inclination [rad]
    raan = df_vimpel[7][index] * d2r                    # right ascension [rad]
    ecc = df_vimpel[8][index]                           # eccentricity [-]
    arg_lat = df_vimpel[9][index] * d2r                 # argument of latitude [rad]
    arg_per = df_vimpel[10][index] * d2r                # argument of perigee [rad]
    am = df_vimpel[11][index]                           # effective area-to-mass [m2/kg]
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

    # create dictionary of vimpel data
    vimpel_data = {
        "coes": coes,       # [rad] classical orbital elements
        "epoch": epoch,     # [ddmmyyyy HHMMSS]
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "min": minute,
        "sec": second,
        "mic": microsecond,
        "AM": am,           # [m2/kg] area to mass ratio
        "std_r": pos_std    # [km] position uncertainty in transverse direction
    }
    return vimpel_data


def vimpel_to_state(coes:list)->np.ndarray:
    """Find state of Vimpel object from classical orbital elements

    :param coes: classical orbital elements: [semi-major axis, eccentricity, inclination, 
                 right ascension of the ascending node, argument of perigee, true anomaly] (angles in rad)
    :type coes: list
    :return: [km, km/s] position and velocity vectors
    :rtype: np.ndarray
    """
    r, v = coes2rv(coes, deg=False, mean_anom=False)  # [km, km/s]
    return r, v


def vimpel_propagation(vimpel_file:str, tf:datetime, cdrag:float, cdiff:float, dt=3600, index=0):
    """ Propagation of a vimpel file object to desired final time tf
        * Requires a Vimpel data file as input

    :param vimpel_file: Vimpel dataset filename
    :type vimpel_file: str
    :param tf: epoch to propagate Vimpel object to
    :type tf: datetime
    :param cdrag: [-] drag coefficient
    :type cdrag: float
    :param cdiff: diffusion coefficient
    :type cdiff: float
    :param dt: [sec] propagation time increment, defaults to 3600 (once per hour)
    :type dt: int, optional
    :param index: index of desired object in the file, defaults to 0
    :type index: int, optional
    :return: position and velocity vectors after propagation, list of classical orbital elements
    :rtype: tuple
    """
   
    # extract parameters from vimpel file
    vimpel_data = extract_vimpel(vimpel_file, index)  # coes in [rad]
    r, v = vimpel_to_state(vimpel_data['coes'])  # [km, km/s]
    year = vimpel_data['year']
    month = vimpel_data['month']
    day = vimpel_data['day']
    hour = vimpel_data['hour']
    minute = vimpel_data['min']
    second = vimpel_data['sec']
    microsecond = vimpel_data['mic']
    area_to_mass = vimpel_data['AM']

    t_object = datetime(year, month, day, hour, minute, second, microsecond)   # epoch of object (datetime format)
    jd = jday(year, month, day, hour, minute, second)  # Julian date of object
    space_object = {  # object parameters
        'C_drag': cdrag,        # drag coefficient [-]
        'C_diff': cdiff,        # diffusion coefficient [-]
        'AMR': area_to_mass,    # body area to mass ratio [m2/kg]
        'BStar': None,          # Bstar value (from TLE) [1/m]
        'CdAM': None            # Cdrag * AMR [m2/kg]
    }

    "Propagate"
    tsince = tf - t_object                                    # difference between tf and the object epoch (datetime)
    tsince_sec = tsince.total_seconds()                       # difference between tf and the object epoch [sec]
    t = np.linspace(0, tsince_sec, num=abs(int(tsince_sec/dt)+1))  # propagation timesteps (default every hour)

    # initial state of object
    state0 = r.tolist() + v.tolist()  # initial state of object [km, km, km, km/s, km/s, km/s]
    
    # solve ODE to get final parent state
    y = odeint(two_body_ode_with_perturbations, state0, t, args=(jd, space_object), atol=1e-9, rtol=1e-9)
    statef = y[-1]
    r_prop = statef[:3]
    v_prop = statef[3:]
    coes_prop = rv2coes(r_prop, v_prop)  # angles in [rad]

    return r_prop, v_prop, coes_prop
