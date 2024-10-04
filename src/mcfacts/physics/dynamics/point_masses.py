"""Module for handling simple GR orbital evolution (Peters 1964)

Contain functions for orbital evolution and converting between
    units of r_g and SI units

This module is its own module because it does not import other parts
    of mcfacts, in order to avoid circular imports
"""
######## Imports ########

import numpy as np
import scipy

from astropy import units as astropy_units
from astropy import constants as astropy_constants

######## Functions ########

def time_of_orbital_shrinkage(mass_1, mass_2, sep_initial, sep_final):
    """Calculate the GW time for orbital shrinkage

    Calculate the time it takes for two orbiting masses
        to shrink from an initial separation to a final separation
        (Peters)

    Parameters
    ----------
    mass_1 : astropy quantity (npts,)
        A mass array with astropy units
    mass_2 : astropy quantity (npts,)
        Another mass array with astropy units
    sep_initial : astropy quantity (npts,)
        Initial separation of two bodies with astropy units
    sep_final : astropy quantity (npts,)
        Final separation of two bodies with astropy units

    Returns
    -------
    time_of_shrinkage : astropy quantity (npts,)
        time of orbital shrinkage (seconds)
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s').value
    G = astropy_constants.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_initial = sep_initial.to('m').value
    sep_final = sep_final.to('m').value
    # Set up the constant as a single float
    const = ((64 / 5) * G**3) * (c**-5)
    # Calculate the beta array
    beta_arr = const * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate the time
    time_of_shrinkage = (sep_initial ** 4 - sep_final ** 4) / 4 / beta_arr
    # Assign units
    time_of_shrinkage = time_of_shrinkage * astropy_units.s
    return time_of_shrinkage

def orbital_separation_evolve(mass_1, mass_2, sep_initial, evolve_time):
    """Calculate the final separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy quantity (npts,)
        A mass array with astropy units
    mass_2 : astropy quantity (npts,)
        Another mass array with astropy units
    sep_initial : astropy quantity (npts,)
        Initial separation of two bodies with astropy units
    evolve_time : astropy quantity (npts,)
        Time to evolve GW orbit

    Returns
    -------
    sep_final : astropy quantity (npts,)
        Final separation of two bodies with astropy units
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s').value
    G = astropy_constants.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_initial = sep_initial.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const = ((64 / 5) * G**3) * (c**-5)
    # Calculate the beta array
    beta_arr = const * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_initial**4) - (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_final = np.zeros_like(sep_initial)
    sep_final[quantity > 0] = np.sqrt(np.sqrt(quantity[quantity > 0]))
    return sep_final * astropy_units.m

def orbital_separation_evolve_reverse(mass_1, mass_2, sep_final, evolve_time):
    """Calculate the initial separation of an evolved orbit

    Parameters
    ----------
    mass_1 : astropy quantity (npts,)
        A mass array with astropy units
    mass_2 : astropy quantity (npts,)
        Another mass array with astropy units
    sep_final : astropy quantity (npts,)
        Final separation of two bodies with astropy units
    evolve_time : astropy quantity (npts,)
        Time to evolve GW orbit

    Returns
    -------
    sep_initial : astropy quantity (npts,)
        Initial separation of two bodies with astropy units
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s').value
    G = astropy_constants.G.to('m^3/(kg s^2)').value
    # Assert SI units
    mass_1 = mass_1.to('kg').value
    mass_2 = mass_2.to('kg').value
    sep_final = sep_final.to('m').value
    evolve_time = evolve_time.to('s').value
    # Set up the constant as a single float
    const = ((64 / 5) * G**3) * (c**-5)
    # Calculate the beta array
    beta_arr = const * mass_1 * mass_2 * (mass_1 + mass_2)
    # Calculate an intermediate quantity
    quantity = (sep_final**4) + (4 * beta_arr * evolve_time)
    # Calculate final separation
    sep_initial = np.sqrt(np.sqrt(quantity))
    return sep_initial * astropy_units.m

def si_from_r_g(smbh_mass, distance_rg):
    """Calculate the SI distance from r_g
     
    Parameters
    ----------
    smbh_mass : float
        SMBH mass in units of solMass
    distance_rg : array_like
        Distances in r_g

    Returns
    -------
    distance : array_like
        Distance in SI, with astropy units
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s')
    G = astropy_constants.G.to('m^3/(kg s^2)')
    # Assign units to smbh mass
    if hasattr(smbh_mass, 'unit'):
        smbh_mass = smbh_mass.to('solMass')
    else:
        smbh_mass = smbh_mass * astropy_units.solMass
    # convert smbh mass to kg
    smbh_mass = smbh_mass.to('kg')
    # Calculate r_g in SI
    r_g = G*smbh_mass/(c**2)
    # Calculate distance
    distance = distance_rg * r_g
    return distance

def r_g_from_units(smbh_mass, distance):
    """Calculate the SI distance from r_g
     
    Parameters
    ----------
    smbh_mass : float
        SMBH mass in units of solMass
    distance_rg : astropy quantity
        Distances in astropy units

    Returns
    -------
    distance_rg : array_like
        Distance in r_g
    """
    # Calculate c and G in SI
    c = astropy_constants.c.to('m/s')
    G = astropy_constants.G.to('m^3/(kg s^2)')
    # Assign units to smbh mass
    if hasattr(smbh_mass, 'unit'):
        smbh_mass = smbh_mass.to('solMass')
    else:
        smbh_mass = smbh_mass * astropy_units.solMass
    # convert smbh mass to kg
    smbh_mass = smbh_mass.to('kg')
    # Calculate r_g in SI
    r_g = G*smbh_mass/(c**2)
    # Calculate distance
    distance_rg = distance / r_g
    return distance_rg
