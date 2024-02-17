"""Module containing functions pertaining to Vimpel files
Functions include: extracting vimpel file data and propagation of vimpel files

================================================================================"""
# Written by: Arly Black

import numpy as np
from scipy.integrate import odeint
from sgp4.api import jday, SGP4_ERRORS
from datetime import datetime, timedelta
from typing import Optional, Union

from .catalog import extract_tle, tle_dataframe
from .orbit_conversions import coes2rv, rv2coes
from .ode import two_body_ode, two_body_ode_with_perturbations
import pdb


def two_body_propagation(t0:Optional[Union[float, datetime]], tf:Optional[Union[float, datetime]], state0:list, dt=3600)->tuple:
    """Two-body problem propagation

    :param t0: initial time [sec] or epoch [datetime]
    :type t0: datetime
    :param tf: final time [sec] or epoch [datetime]
    :type tf: datetime
    :param state0: [km, km/s] initial object position and velocity state 
    :type state0: list
    :param dt: [sec] propagation time increment, defaults to 3600 (once per hour)
    :type dt: int, optional
    :return: [km, km/s] position and velocity vectors after propagation, list of classical orbital elements (angles in rad)
    :rtype: tuple
    """
    tsince = tf - t0                                       # length of propagation [sec or datetime]
    if isinstance(tsince,timedelta):
        tsince = tsince.total_seconds()                    # convert datetime format to seconds
    t = np.linspace(0, tsince, num=abs(int(tsince/dt)+1))  # propagation timesteps (default every hour)
    
    # solve ODE to get final parent state
    y = odeint(two_body_ode, state0, t, atol=1e-9, rtol=1e-9)
    statef = y[-1]
    r = statef[:3]
    v = statef[3:]
    coes = rv2coes(r, v)  # angles in [rad]

    return r, v, coes


def two_body_propagation_with_perturbations(t0:datetime, tf:datetime, state0:list, object_parameters:dict, dt=3600)->tuple:
    """Two-body propagation with added perturbations of J2, drag, SRP, and three-body

    :param t0: initial epoch
    :type t0: datetime
    :param tf: final epoch
    :type tf: datetime
    :param state0: [km, km/s] initial object position and velocity state 
    :type state0: list
    :param object_parameters: parameters relevant to propagated object, 
                              including: drag coefficient ('C_drag'), diffusion coefficient ('C_diff'), area-to-mass ratio ('AMR'),
                              pseudo-ballistic bstar value ('BStar'), drag coeff * AMR ('CdAM') - last two only required if Cdrag and AMR are not available
    :type object_parameters: dict
    :param dt: [sec] propagation time increment, defaults to 3600 (once per hour)
    :type dt: int, optional
    :return: [km, km/s] position and velocity vectors after propagation, list of classical orbital elements (angles in rad)
    :rtype: tuple
    """
    jd = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)  # Julian date of object at t0
    tsince = tf - t0                                                     # difference between tf and the object epoch (datetime)
    tsince_sec = tsince.total_seconds()                                  # difference between tf and the object epoch [sec]
    t = np.linspace(0, tsince_sec, num=abs(int(tsince_sec/dt)+1))        # propagation timesteps (default every hour)
    
    # solve ODE to get final parent state
    y = odeint(two_body_ode_with_perturbations, state0, t, args=(jd, object_parameters), atol=1e-9, rtol=1e-9)
    statef = y[-1]
    r = statef[:3]
    v = statef[3:]
    coes = rv2coes(r, v)  # angles in [rad]

    return r, v, coes


def sgp4_propagation(tle_file:str, tf:datetime):
    """Propagate TLE to desired epoch and extract osculating elements

    :param tle_file: name of TLE file
    :type tle_file: str
    :param tf: final epoch (to propagate to)
    :type tf: datetime
    :raises RuntimeError: _description_
    :return: final position, velocity and osculating orbit elements
    :rtype: np.ndarray
    """
    df_tle = tle_dataframe(tle_file)  # dataframe of TLE data
    line1 = df_tle.line1[0]
    line2 = df_tle.line2[0]

    jd, fr = jday(tf.year, tf.month, tf.day, tf.hour, tf.minute, tf.second)
    tle_data = extract_tle(line1, line2)
    satellite = tle_data.satellite

    error_code, r, v = satellite.sgp4(jd, fr)  # propagate tle to given julian day
    if error_code != 0:
        raise RuntimeError(SGP4_ERRORS[error_code])
    # find osculating orbital elements at given julian day from r, v, mu
    coes = rv2coes(r, v)  # [km, -, rad, rad, rad, rad]

    return r, v, coes


class OrbitingBody:
    """ Class for defining characteristics of an orbiting body"""
    def __init__(self, **kwargs):
        # object properties (optional)
        self.C_drag = kwargs.get('cdrag', 2.2)  # [-] drag coefficient
        self.C_diff = kwargs.get('cdiff', 1)    # [-] diffusion coefficient 
        self.AMR = kwargs.get('amr', None)       # [m2/kg] body area to mass ratio
        self.BStar = kwargs.get('bstar', None)   # [1/m] B* parameter from TLE
        self.CdAM = kwargs.get('cd-am', None)     # [m2/kg] Cdrag*A/M


# def vimpel_propagation(vimpel_file:str, tf:datetime, cdrag:float, cdiff:float, dt=3600, index=0):
#     """ Propagation of a vimpel file object to desired final time tf
#         * Requires a Vimpel data file as input

#     :param vimpel_file: Vimpel dataset filename
#     :type vimpel_file: str
#     :param tf: epoch to propagate Vimpel object to
#     :type tf: datetime
#     :param cdrag: [-] drag coefficient
#     :type cdrag: float
#     :param cdiff: diffusion coefficient
#     :type cdiff: float
#     :param dt: [sec] propagation time increment, defaults to 3600 (once per hour)
#     :type dt: int, optional
#     :param index: index of desired object in the file, defaults to 0
#     :type index: int, optional
#     :return: position and velocity vectors after propagation, list of classical orbital elements
#     :rtype: tuple
#     """
   
#     # extract parameters from vimpel file
#     vimpel_data = extract_vimpel(vimpel_file, index)  # coes in [rad]
#     r, v = vimpel_to_state(vimpel_data['coes'])  # [km, km/s]
#     year = vimpel_data['year']
#     month = vimpel_data['month']
#     day = vimpel_data['day']
#     hour = vimpel_data['hour']
#     minute = vimpel_data['min']
#     second = vimpel_data['sec']
#     microsecond = vimpel_data['mic']
#     area_to_mass = vimpel_data['AM']

#     t_object = datetime(year, month, day, hour, minute, second, microsecond)   # epoch of object (datetime format)
#     jd = jday(year, month, day, hour, minute, second)  # Julian date of object
#     space_object = {  # object parameters
#         'C_drag': cdrag,        # drag coefficient [-]
#         'C_diff': cdiff,        # diffusion coefficient [-]
#         'AMR': area_to_mass,    # body area to mass ratio [m2/kg]
#         'BStar': None,          # Bstar value (from TLE) [1/m]
#         'CdAM': None            # Cdrag * AMR [m2/kg]
#     }

#     "Propagate"
#     tsince = tf - t0                                    # difference between tf and the object epoch (datetime)
#     tsince_sec = tsince.total_seconds()                       # difference between tf and the object epoch [sec]
#     t = np.linspace(0, tsince_sec, num=abs(int(tsince_sec/dt)+1))  # propagation timesteps (default every hour)

#     # initial state of object
#     state0 = r.tolist() + v.tolist()  # initial state of object [km, km, km, km/s, km/s, km/s]
    
#     # solve ODE to get final parent state
#     y = odeint(two_body_ode_with_perturbations, state0, t, args=(jd, space_object), atol=1e-9, rtol=1e-9)
#     statef = y[-1]
#     r_prop = statef[:3]
#     v_prop = statef[3:]
#     coes_prop = rv2coes(r_prop, v_prop)  # angles in [rad]

#     return r_prop, v_prop, coes_prop

