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
import matplotlib.pyplot as plt

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
        pdb.set_trace()

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


def plot_fragments_no_uncertainty(output, ri):
    period = output[0]
    perigee = output[1]
    apogee = output[2]
    dvr = output[3]
    dvd = output[4]
    dvx = output[5]
    lat = output[6] * 180/np.pi
    lon = output[7] * 180/np.pi
    rf1 = output[8]
    rf2 = output[9]
    rf3 = output[10]

    # location 
    ax = plt.figure().add_subplot(projection="3d")
    plt.plot(rf1,rf2,rf3,'.')
    plt.plot(ri[0],ri[1],ri[2],'.')
    ax.set_aspect("equal")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Gabbard plot
    fig = plt.figure()
    ap = plt.scatter(np.array(period)/60, np.array(apogee), s=2)
    pe = plt.scatter(np.array(period)/60, np.array(perigee), s=2)
    plt.xlabel('Period [min]')
    plt.ylabel('Altitude [km]')
    plt.grid(True)
    plt.legend((ap, pe),
                ('Apogee', 'Perigee'),
                scatterpoints=1,
                loc='upper left',
                fontsize=8)

    # Plot radial vs downrange velocity perturbations
    fig_rd_orig = plt.figure()
    ax_rd_orig = fig_rd_orig.add_subplot(111)
    plt.scatter(dvr, dvd, s=1.5)
    ax_rd_orig.set(xlabel=r'Radial velocity perturbation, $dv_r$ [km/s]',
                    ylabel=r'Downrange velocity perturbation, $dv_d$ [km/s]')
    plt.grid(True)
    ax_rd_orig.axis('equal')

    # Plot radial vs cross-range velocity perturbations
    fig_rx_orig = plt.figure()
    ax_rx_orig = fig_rx_orig.add_subplot(111)
    plt.scatter(dvr, dvx, s=1.5)
    ax_rx_orig.set(xlabel=r'Radial velocity perturbation, $dv_r$ [km/s]',
                    ylabel=r'Cross-range velocity perturbation, $dv_x$ [km/s]')
    plt.grid(True)
    ax_rx_orig.axis('equal')

    # Plot downrange vs cross-range velocity perturbations
    fig_dx_orig = plt.figure()
    ax_dx_orig = fig_dx_orig.add_subplot(111)
    plt.scatter(dvd, dvx, s=1.5)
    ax_dx_orig.set(xlabel=r'Downrange velocity perturbation, $dv_d$ [km/s]',
                    ylabel=r'Cross-range velocity perturbation, $dv_x$ [km/s]')
    plt.grid(True)
    ax_dx_orig.axis('equal')

    # Plot angular distribution
    fig_ang_orig = plt.figure()
    ax_ang_orig = fig_ang_orig.add_subplot(111)
    plt.scatter(lon, lat, s=1.5)
    ax_ang_orig.set(xlabel=r'Longitude, [deg]',
                    ylabel=r'Latitude [deg]')
    ax_ang_orig.set_ylim([-90, 90])
    # ax_ang_orig.set_xlim([-180, 180])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    " ----- INPUT ---------------------------- "
    case = Case14055B
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
    pdb.set_trace()

    # obtain position uncertainty standard deviation information for each fragment - convert from ICO to ECI frame
    std_r = position_uncertainty(parent, std_intrack_ico, num_frags)
    
    # propagate fragments with no uncertainty
    # Multiprocessing
    # for k in range(1):
    #     period, perigee, apogee, dv_radial, dv_downrange, dv_crossrange, angular_latitude, angular_longitude = vimpel_parameters(file_path, case['breakup time'], cdrag, cdiff, parent, k)
    pool = Pool(8)  # specify number of cores for multi-processing
    fragment_outputs = functools.partial(vimpel_parameters,file_path, case['breakup time'], cdrag, cdiff, parent)
    output = pool.map(fragment_outputs, range(num_frags))   
    pool.close() 

    # plot original fragments
    plot_fragments_no_uncertainty(np.array(output).T, np.array(r_i).T)

    # generate sigma points for each fragment
    mean = np.hstack((r_i, v_i))
    covariance = get_covariance_from_noise(num_frags,std_r,sigma_v,file_format='Vimpel')
    s_num, sigma_points, weight_mean, weight_cov = generate_sigma_points(num_frags, len(mean[0]), mean, covariance)
    pdb.set_trace()

    # propagate fragments with uncertainty