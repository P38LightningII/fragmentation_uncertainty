
import numpy as np
from numpy.linalg import norm
from datetime import datetime
from scipy.integrate import odeint
from pymap3d import eci2geodetic

from .catalog import extract_vimpel
from .propagation import two_body_propagation_with_perturbations, two_body_propagation #, OrbitingBody
from .ode import two_body_ode, two_body_ode_with_perturbations
from .orbit_conversions import xyz_to_rdx, ico_to_rdx, eci_to_rdx, coes2rv
from .perturbation_functions import velocity_perturbations, angular_distribution
import planetary_data as pl
import pdb


def propagation_inputs(filename:str, cdrag:float, cdiff:float, index=0)->tuple:
    """Obtain required inputs for propagation

    :param filename: name of file containing Vimpel data (with .txt extension)
    :type filename: str
    :param cdrag: [-] drag coefficient
    :type cdrag: float
    :param cdiff: [-] diffusion coefficient
    :type cdiff: float
    :param index: index of desired object in the file, defaults to 0
    :type index: int, optional
    :return: initial state and object parameters
    :rtype: tuple
    """
    # extract parameters from vimpel file
    vimpel_data = extract_vimpel(filename, index)  # coes in [rad]

    # calculate initial object state
    r0, v0 = coes2rv(vimpel_data['coes'], deg=False, mean_anom=False)  # [km, km/s]
    state0 = r0.tolist() + v0.tolist()  # [km, km/s]

    # establish object parameters
    object_parameters = {
        'C_drag': cdrag,
        'C_diff': cdiff,
        'AMR': vimpel_data['amr'],
    }
    # object_parameters = OrbitingBody(cdrag=cdrag, cdiff=cdiff, amr=vimpel_data['amr'])

    return vimpel_data, state0, object_parameters


def propagate_parent(filename:str, tf:datetime, cdrag:float, cdiff:float, parent_index=0)->dict:
    """Propagate parent spacecraft to time of breakup event

    :param filename: name of Vimpel file containing parent object data (with .txt extension)
    :type filename: str
    :param tf: breakup epoch
    :type tf: datetime
    :param cdrag: [-] drag coefficient
    :type cdrag: float
    :param cdiff: [-] diffusion coefficient
    :type cdiff: float
    :param parent_index: index of the parent object in the file, defaults to 0
    :type parent_index: int, optional
    :return: dictionary of parent object parameters
    :rtype: dict
    """
    # propagate parent to time of breakup and extract cartesian and keplerian state data
    vimpel_data, state0, object_parameters = propagation_inputs(filename, cdrag, cdiff, parent_index)
    rf, vf, coes = two_body_propagation(vimpel_data['epoch'], tf, state0)
    # rf, vf, coes = two_body_propagation_with_perturbations(vimpel_data['epoch'], tf, state0, object_parameters)  ## USE THIS!!

    period = 2 * np.pi / np.sqrt(pl.earth['mu'] / coes[0] ** 3)

    # obtain velocity components in RDX coordinates
    dcm_xyz2rdx = xyz_to_rdx(rf, vf) # transformation matrix from inertial to radial, downrange, crossrange
    v_rdx = dcm_xyz2rdx @ vf
    v_radial = v_rdx[0]
    v_downrange = v_rdx[1]

    latitude, longitude, altitude = eci2geodetic(rf[0] * 1000, rf[1] * 1000, rf[2] * 1000, tf)  # [deg, deg, m]

    # propagate parent orbit over one period
    state0 = rf.tolist() + vf.tolist()  # initial state of parent [km, km/s]
    t = np.linspace(0, period)
    sol = odeint(two_body_ode, state0, t, atol=1e-9, rtol=1e-9)

    # parent properties at breakup
    parent = {
        'r': rf,           # [km]
        'v': vf,           # [km/s]
        'sma': coes[0],    # [km]
        'ecc': coes[1],    # [-]
        'inc': coes[2],    # [rad]
        'period': period,  # [sec]
        'perigee': coes[0] * (1 - coes[1]),         # [km]
        'apogee': coes[0] * (1 + coes[1]),          # [km]
        'altitude': norm(rf) - pl.earth['radius'],   # [km]
        'v_radial': v_radial,                       # [km/s]
        'v_downrange': v_downrange,                 # [km/s]
        'latitude': latitude * np.pi/180,           # [rad]
        'orbit_coords': sol,  # orbit coordinates at breakup
    }
    return parent


def vimpel_parameters(filename, tf, cdrag, cdiff, parent, i):

    vimpel_data, state0, object_parameters = propagation_inputs(filename, cdrag, cdiff, i)
    rf, vf, coes = two_body_propagation(vimpel_data['epoch'], tf, state0)
    # rf, vf, coes = two_body_propagation_with_perturbations(vimpel_data['epoch'], tf, state0, object_parameters)

    # calculate period, perigee, and apogee
    period = 2 * np.pi / np.sqrt(pl.earth['mu'] / coes[0] ** 3)  # [sec]
    perigee = coes[0] * (1 - coes[1])  # [km] perigee
    apogee = coes[0] * (1 + coes[1])   # [km] apogee 

    # calculate radial, downrange, and crossrange velocity components
    dv_radial, dv_downrange, dv_crossrange = velocity_perturbations(parent, coes, vf, method='Exact')  # [km/s]
     
    # calculate angular distribution of fragments
    dV = np.sqrt(dv_radial**2 + dv_downrange**2 + dv_crossrange**2)  # [km/s]
    latitude, longitude = angular_distribution(dv_radial, dv_downrange, dv_crossrange, dV)  # [deg]
    d2r = np.pi/180

    return period, perigee, apogee, dv_radial, dv_downrange, dv_crossrange, latitude*d2r, longitude*d2r


def perturbed_vimpel_parameters(filename, tf, cdrag, cdiff, parent, i, perturbed_state, nsigma):

    # initialize arrays
    state_j = []
    coes_j = []
    # state_j = [np.empty([0,1])]
    # coes_j = np.empty([nsigma,1])
    period_j = np.empty([nsigma,1])
    perigee_j = np.empty([nsigma,1])
    apogee_j = np.empty([nsigma,1])
    dv_j = np.empty([nsigma,1])
    geographic_coords_j = np.empty([nsigma,1])
    # cycle through each sigma point
    for j in range(nsigma): 
        vimpel_data, _, object_parameters = propagation_inputs(filename, cdrag, cdiff, i)
        rf, vf, coes = two_body_propagation(vimpel_data['epoch'], tf, perturbed_state[i][j])
        # rf, vf, coes = two_body_propagation_with_perturbations(vimpel_data['epoch'], tf, perturbed_state[i][j], object_parameters)

        # save fragment parameters (for every sigma point)
        state_j.append(np.hstack((rf,vf)))
        coes_j.append(np.array(coes))
        # state_j = np.vstack((state_j, np.hstack((rf,vf))))
        # coes_j = np.vstack((coes_j, np.array(coes)))

        # calculate period, perigee, and apogee
        period_j = np.vstack((period_j, 2 * np.pi / np.sqrt(pl.earth['mu'] / coes[0] ** 3)))    # [sec] period
        perigee_j = np.vstack((perigee_j, coes[0] * (1 - coes[1])  - pl.earth['radius']))       # [km] perigee altitude
        apogee_j = np.vstack((apogee_j, coes[0] * (1 + coes[1])  - pl.earth['radius']))         # [km] apogee altitude

        # Calculate fragment radial, downrange, and crossrange velocity components
        dv_r, dv_d, dv_x = velocity_perturbations(parent, coes, vf, method='Exact')  # [km/s]
        dv_j = np.vstack((dv_j, np.hstack((dv_r, dv_d, dv_x))))  # [km/s]

        # calculate angular distribution of fragments
        dV = np.sqrt(dv_r ** 2 + dv_d ** 2 + dv_x ** 2)  # [km/s]
        latitude, longitude = angular_distribution(dv_r, dv_d, dv_x, dV)  # [deg]
        d2r = np.pi/180
        geographic_coords_j = np.vstack((geographic_coords_j, np.hstack((longitude*d2r,latitude*d2r)))) # [rad]