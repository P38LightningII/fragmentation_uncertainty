"""Apply unscented transform sigma point analysis to fragment Vimpel data and propagate to determine uncertainty

================================================================================"""
# Written by: Arly Black, Oct 2022
# Modified: Feb 2024

import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.integrate import odeint
from datetime import datetime
from sgp4.api import jday
import os

from vimpel_functions import extract_vimpel, vimpel_to_state, vimpel_propagation
from centaur_dict import Case09047B, Case14055B, Case18079B
from orbit_conversions import xyz_to_rdx, ico_to_rdx, eci_to_rdx
from ode import two_body_ode, two_body_ode_with_perturbations
import planetary_data as pl



def propagate_parent(filename, tf, cdrag, cdiff):
    # propagate parent spacecraft to breakup time
    r, v, coes = vimpel_propagation(filename, tf, cdrag, cdiff)  # angles in rad

    dcm_xyz2rdx = xyz_to_rdx(r, v) # transformation matrix from inertial to radial, downrange, crossrange
    v_rdx = np.matmul(dcm_xyz2rdx, v)
    v_radial = v_rdx[0]
    v_downrange = v_rdx[1]

    # propagate parent two-body orbit at time of breakup
    state0 = r.tolist() + v.tolist()  # initial state of parent [km, km/s]
    t = np.linspace(0, parent['period'])
    sol = odeint(two_body_ode, state0, t, atol=1e-9, rtol=1e-9)

    # parent properties at breakup
    parent = {
        'r': r,             # [km]
        'v': v,             # [km/s]
        'sma': coes[0],     # [km]
        'ecc': coes[1],     # [-]
        'inc': coes[2,],    # [rad]
        'period': 2 * np.pi / np.sqrt(pl.earth['mu'] / coes[0] ** 3),  # [sec]
        'perigee': coes[0] * (1 - coes[1]),         # [km]
        'apogee': coes[0] * (1 + coes[1]),          # [km]
        'altitude': norm(r) - pl.earth['radius'],   # [km]
        'v_radial': v_radial,                       # [km/s]
        'v_downrange': v_downrange,                 # [km/s]
        'orbit_coords': sol,  # orbit coordinates at breakup
    }
    return parent


def propagate_vimpel_objects(filename, tf, cdrag, cdiff, i):
     
    r, v, coes = vimpel_propagation(filename, tf, cdrag, cdiff, index=i)  # angles in rad

    # calculate period, perigee, and apogee
    period = 2 * np.pi / np.sqrt(pl.earth['mu'] / coes[0] ** 3)  # [sec]
    perigee = coes[0] * (1 - coes[1])  # [km] perigee
    apogee = coes[0] * (1 + coes[1])   # [km] apogee 

    # calculate radial, downrange, and crossrange velocity components
    # dv_radial, dv_downrange, dv_crossrange = get_velocity_perturbations(parent, sma, ecc, inc, ta, aop, v,
    #                                                                     method='Exact')
     
    # calculate angular distribution of fragments
    # dV = np.sqrt(dv_radial**2 + dv_downrange**2 + dv_crossrange**2)
    # angular_latitude, angular_longitude = get_angular_distribution(dv_radial, dv_downrange, dv_crossrange, dV)  # [deg]

    # return period, perigee, apogee, dv_radial, dv_downrange, dv_crossrange, angular_latitude, angular_longitude
    return period, perigee, apogee


def propagate_perturbed_vimpel_objects():


if __name__ == '__main__':
    " ----- INPUT ---------------------------- "
    case = Case09047B
    sigma_v = 0.001    # velocity noise [km/s]
    confidence = 0.95  # error ellipse confidence level

    cdrag = 2    # [-] drag coefficient (assuming all frags have same constant Cd)
    cdiff = 0.5  # [-] diffusion coefficient
    perts = 'J2, drag, SRP, 3B'  # perturbation options
    " ---------------------------------------- "

    filename = case['Vimpel file']
    file_path = os.path.join(os.environ["FRAG_SRC"], "catalog_files", filename)
    parent_path = os.path.join(os.environ["FRAG_SRC"], "catalog_files", case['Vimpel parent file'])

    # find parent properties at time of breakup
    parent = propagate_parent(filename, case['breakup time'], cdrag, cdiff)

    df_vimpel = pd.read_csv(file_path, header=None)
    num_frags = len(df_vimpel)  # number of objects in TLE

    # find the ECI state of each fragment at the given Vimpel epoch
    r_vimpel = []
    v_vimpel = []
    std_intrack = []
    for i in range(num_frags):
        vimpel_data = extract_vimpel(file_path, i)
        r, v = vimpel_to_state(vimpel_data['coes'])
        r_vimpel.append(r)  # [km] fragment position vector
        v_vimpel.append(v)  # [km/s] fragment velocity vector
        std_intrack.append(vimpel_data['std_r'])   # [km] fragment position uncertainty in transverse direction

    # obtain position uncertainty standard deviation information for each fragment - convert from ICO to RDX to ECI frame
    dcm_rdx2eci = xyz_to_rdx(parent['r'], parent['v]']).T
    dcm_ico2rdx = ico_to_rdx(parent['r'], parent['v]'])
    dcm_ico2eci = dcm_ico2rdx @ dcm_rdx2eci
    sig_avg = np.mean(std_intrack)
    std_r = []
    for i in range(num_frags):
        if std_intrack[i] == 0:
            std_intrack[i] = sig_avg
        std_ico = [std_intrack[i], std_intrack[i]/5.608, std_intrack[i]/5.663]  # based on Baietto thesis
        std_r.append(dcm_ico2eci @ std_ico)

    # propagate fragments with no uncertainty
        

    # propagate fragments with uncertainty