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
from orbit_conversions import xyz_to_rdx, ico_to_rdx
from ode import two_body_ode, two_body_ode_with_perturbations
import planetary_data as pl



def parent_propagation(filename, tf, cdrag, cdiff):
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

    df_vimpel = pd.read_csv(file_path, header=None)
    num_frags = len(df_vimpel)  # number of objects in TLE

    # find the Cartesian state of each fragment at the given Vimpel epoch
    r_vimpel = []
    v_vimpel = []
    sigma_intrack = []
    for i in range(num_frags):
        vimpel_data = extract_vimpel(file_path, i)
        r, v = vimpel_to_state(vimpel_data['coes'])
        r_vimpel.append(r)  # [km] fragment position vector
        v_vimpel.append(v)  # [km/s] fragment velocity vector
        sigma_intrack.append(vimpel_data['std_r'])   # [km] fragment position uncertainty in transverse direction

    # find parent properties at time of breakup
    parent = parent_propagation(filename, case['breakup time'], cdrag, cdiff)

    # obtain position uncertainty standard deviation information

    "Convert position uncertainty values from ICO to RDX to ECI frame"
    DCM_ICO2RDX = ICO_to_RDX(r_parent_breakup, v_parent_breakup)
    DCM_ECI2RDX = parent['rotMatrix']
    # DCM_ECI2RDX = ECI_to_RDX(raan_parent_breakup*deg2rad, inc_parent_breakup*deg2rad, aop_parent_breakup*deg2rad,
    #                          ta_parent_breakup*deg2rad)
    # DCM_RDX2ECI = np.array(DCM_ECI2RDX).T.tolist()
    DCM_RDX2ECI = DCM_ECI2RDX.T
    print(DCM_RDX2ECI)
    sigr_avg = np.mean(sigma_intrack)
    sigma_r = []
    for ii in range(num_frags):
        if sigma_intrack[ii] == 0:
            sigma_intrack[ii] = sigr_avg
        sigma_ICO = [sigma_intrack[ii], sigma_intrack[ii]/5.608, sigma_intrack[ii]/5.663]  # based on Baietto thesis
        sigma_RDX = (np.matmul(DCM_ICO2RDX, sigma_ICO))
        sigma_r.append(np.matmul(DCM_RDX2ECI, sigma_RDX))
        # print(ii)
        # print(sigma_ICO)
        # print(sigma_RDX)