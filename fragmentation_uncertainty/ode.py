"""Ordinary differential equations for propagation

================================================================================"""

import numpy as np
from numpy.linalg import norm
import planetary_data as pl
from orbital_perturbations import satellite_dynamics


def two_body_ode(state, t):
    """Two Body Propagator"""
    r = state[:3]
    dvdt = -pl.earth['mu'] / norm(r)**3 * r
    state_derivative = [state[3], state[4], state[5], dvdt[0], dvdt[1], dvdt[2]]

    return state_derivative


def two_body_ode_with_perturbations(state:list, t:np.ndarray, jd:tuple, space_obj):
    """Propagator"""
    jd = jd[0] + jd[1]
    jd = jd + t / 86400  # updated Julian date [days]
    state_derivative = satellite_dynamics(t, state, jd, space_obj, perturbations='J2, drag, SRP, 3B', degree=4, order=2)