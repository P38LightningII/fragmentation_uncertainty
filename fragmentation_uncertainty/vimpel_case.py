"""Apply unscented transform sigma point analysis to fragment Vimpel data and propagate to determine uncertainty

================================================================================"""
# Written by: Arly Black, Oct 2022
# Modified: Feb 2024

import numpy as np
from numpy.linalg import norm
import pandas as pd
import os
from multiprocessing import Pool
import functools

from .catalog import extract_vimpel
from .centaur_dict import Case09047B, Case14055B, Case18079B
from .orbit_conversions import xyz_to_rdx, ico_to_rdx, eci_to_rdx, coes2rv, xyz_to_ico
from .unscented_transform import generate_sigma_points, get_covariance_from_noise
from .vimpel_functions import propagate_parent, vimpel_parameters
import planetary_data as pl
import pdb


def fragment_state(filename, num_frags):
    r_vimpel = []
    v_vimpel = []
    std_intrack = []
    for i in range(num_frags):
        vimpel_data = extract_vimpel(filename, i)
        r, v = coes2rv(vimpel_data['coes'], deg=False, mean_anom=False)
        r_vimpel.append(r)  # [km] fragment position vector
        v_vimpel.append(v)  # [km/s] fragment velocity vector
        std_intrack.append(vimpel_data['std_r'])   # [km] fragment position uncertainty in transverse direction

    return r_vimpel, v_vimpel, std_intrack


def position_uncertainty(parent, std_intrack, num_frags):
    # obtain position uncertainty standard deviation information for each fragment - convert from ICO to ECI frame
    dcm_ico2eci = xyz_to_ico(parent['r'], parent['v']).T  # ^I[C]^X

    sig_avg = np.mean(std_intrack)
    std_eci = []
    for i in range(num_frags):
        if std_intrack[i] == 0:
            std_intrack[i] = sig_avg
        std_ico = [std_intrack[i], std_intrack[i]/5.608, std_intrack[i]/5.663]  # based on Baietto thesis
        std_eci.append(np.abs(dcm_ico2eci @ std_ico))

    return std_eci



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
    parent = propagate_parent(parent_path, case['breakup time'], cdrag, cdiff)

    df_vimpel = pd.read_csv(file_path, header=None)
    num_frags = len(df_vimpel)  # number of objects in TLE

    # obtain ECI states of each fragment at the given Vimpel epoch
    r_i, v_i, std_intrack_ico = fragment_state(file_path, num_frags)

    # obtain position uncertainty standard deviation information for each fragment - convert from ICO to ECI frame
    std_r = position_uncertainty(parent, std_intrack_ico, num_frags)
    
    # propagate fragments with no uncertainty
    # Multiprocessing
    for k in range(1):
        period, perigee, apogee, dv_radial, dv_downrange, dv_crossrange, angular_latitude, angular_longitude = vimpel_parameters(file_path, case['breakup time'], cdrag, cdiff, parent, k)
    # pool = Pool(8)  # specify number of cores for multi-processing
    # fragment_outputs = functools.partial(vimpel_parameters,file_path, case['breakup time'], cdrag, cdiff, parent)
    # output = pool.map(fragment_outputs, range(num_frags))   
    # pool.close() 

    # generate sigma points for each fragment
    mean = np.hstack((r_i, v_i))
    covariance = get_covariance_from_noise(num_frags,std_r,sigma_v,file_format='Vimpel')
    pdb.set_trace()
    s_num, sigma_points, weight_mean, weight_cov = generate_sigma_points(num_frags, len(mean[0]), mean, covariance)
    pdb.set_trace()

    # propagate fragments with uncertainty