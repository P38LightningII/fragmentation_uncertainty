
import numpy as np
from numpy.linalg import norm
import cmath  # enables calculation of complex numbers
import planetary_data as pl


def velocity_perturbations(parent, coes, v, method='Exact'):
    """Calculate fragment radial, downrange, and crossrange velocity components"""

    if method == 'DCM':
        " --DCM Method-- "
        v_frag_rdx = np.matmul(parent['rotMatrix'], v)
        v_parent_rdx = np.matmul(parent['rotMatrix'], parent['v'])
        dv = v_frag_rdx - v_parent_rdx
        dvr_i = dv[0]
        dvd_i = dv[1]
        dvx_i = dv[2]
    elif method == 'Exact':
        "--Exact Solution Method (Tan & Reynolds)--"
        # radial velocity perturbation [km/s]
        # np.sqrt results in nan (and thus an error) for a negative number, while cmath delivers complex values
        dvr_i = cmath.sqrt(pl.earth['mu'] * (2 / norm(parent["r"]) - 1 / coes[0]) -
                           pl.earth['mu'] / (norm(parent["r"]) ** 2) * coes[0] * (1 - coes[1] ** 2)) - parent["v_radial"]
        if coes[5] % (2*np.pi) > np.pi:
            dvr_i = -cmath.sqrt(pl.earth['mu'] * (2 / norm(parent["r"]) - 1 / coes[0]) -
                                pl.earth['mu'] / (norm(parent["r"]) ** 2) * coes[0] * (1 - coes[1] ** 2)) - parent["v_radial"]
        # if not dvr_i.imag == 0.0:  # set complex valued results to 0
        #     dvr_i = 0
        dvr_i = dvr_i.real  # only save real component of dvr

        # calculate plane-change angle [rad]
        # According to Tan & Reynolds, +xi when i_frag > i_parent on northbound orbits (opposite on southbound)
        xi = np.arccos((np.cos(parent["inc"]) * np.cos(coes[2]) +
                        cmath.sqrt(np.cos(parent["latitude"]) ** 2 - np.cos(parent["inc"]) ** 2) *
                        cmath.sqrt(np.cos(parent["latitude"]) ** 2 - np.cos(coes[2]) ** 2)) / np.cos(
                        parent["latitude"]) ** 2)  # [rad]
        xi = xi.real
        if (coes[4] + coes[5]) % (2*np.pi) <= np.pi:  # ascending (northbound)
            if coes[2] < parent["inc"]:
                xi = -xi
        elif (coes[4] + coes[5]) % (2*np.pi) > np.pi:  # descending (southbound)
            if coes[2] > parent["inc"]:
                xi = -xi

        # downrange velocity perturbation [km/s]
        dvd_i = np.cos(xi) / norm(parent["r"]) * np.sqrt(pl.earth['mu'] * coes[0] * (1 - coes[1] ** 2)) - parent["v_downrange"]
        if np.isnan(dvd_i):
            dvd_i = 0
        # dvd_i = dvd_i.real  # only save real component of dvr

        # cross-range velocity perturbation [km/s]
        dvx_i = np.sin(xi) / norm(parent["r"]) * np.sqrt(pl.earth['mu'] * coes[0] * (1 - coes[1] ** 2))
        if np.isnan(dvx_i):
            dvx_i = 0
        # dvx_i = dvx_i.real  # only save real component of dvr
    else:
        raise Exception("Not a valid velocity perturbation calculation method")

    # return dv_radial, dv_downrange, dv_crossrange
    return dvr_i, dvd_i, dvx_i


def angular_distribution(dv_r, dv_d, dv_x, dv):
    """calculate angular distribution of fragments"""
    latitude = np.arcsin(dv_r / dv)  # [rad]
    latitude = np.rad2deg(latitude)  # [deg]

    # longitude = math.atan(dv_x / dv_d) + n * np.pi  # [rad]
    longitude = np.arctan2(dv_x, dv_d)  # [rad]
    longitude = np.rad2deg(longitude) % 360  # [deg] must be between 0 and 360 deg

    return latitude, longitude  # [deg]