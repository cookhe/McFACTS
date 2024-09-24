"""
Capture Inclination Dampening
===========

This module provides a function for calculating change of an orbiter's inclination angle.
"""
import numpy as np
import scipy
import mcfacts.constants as mc_const


def orb_inc_damping(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses, disk_bh_retro_orbs_ecc,
                    disk_bh_retro_orbs_inc, disk_bh_retro_arg_periapse, timestep_duration_yr, disk_surf_density_func):
    """Calculates how fast the inclination angle of an arbitrary single orbiter changes due to dynamical friction.
     
    Appropriate for BH, NS, maaaybe WD?--check using Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL).
    
    Notes
    -----
    
    It returns the new locations of the retrograde
    orbiters after 1 timestep_duration_yr. Note we have assumed the masses of the orbiters are
    negligible compared to the SMBH (<1% should be fine).

    Unlike all the other orbital variables (semi-major axis, ecc, semi-latus rectum)
    the timescale won't necessarily do anything super horrible for inc = pi, since it only
    depends on the inclination angle itself, sin(inc)... however, it could do something 
    horrible at inc=pi for some values of omega; and it WILL go to zero at inc=0, which 
    could easily cause problems...

    Also, I think this function will still work fine if you feed it prograde bh
    just change the variable name in the function call... (this is not true for migration)
    Testing implies that inc=0 or pi is actually ok, at least for omega=0 or pi

    Parameters
    ----------
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    disk_bh_retro_orbs_a : float array
        locations of retrograde singleton BH at start of a timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
        (actually we assume this is the same as the semi-major axis)
    disk_bh_retro_masses : float array
        masses of retrograde singleton BH at start of a timestep in units of solar masses
    disk_bh_retro_orbs_ecc : float array
        orbital eccentricities of retrograde singleton BH at start of a timestep.
    disk_bh_retro_orbs_inc : float array
        orbital inclinations of retrograde singleton BH at start of a timestep.
    disk_bh_retro_arg_periapse : float array
        argument of periapse of retrograde singleton BH at start of a timestep.
    timestep_duration_yr : float
        size of a timestep in years
    disk_surf_density_func : function
        method provides the AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    disk_bh_retro_orbs_ecc_new : float array
        orbital inclinations of retrograde singletons BH at end of a timestep.
    """

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    smbh_mass = smbh_mass * mc_const.mass_per_msun  # kg
    semi_maj_axis = disk_bh_retro_orbs_a * scipy.constants.G * smbh_mass \
                    / (scipy.constants.c) ** 2  # m
    retro_mass = disk_bh_retro_masses * mc_const.mass_per_msun  # kg
    omega = disk_bh_retro_arg_periapse  # radians
    ecc = disk_bh_retro_orbs_ecc  # unitless
    inc = disk_bh_retro_orbs_inc  # radians
    timestep_duration_yr = timestep_duration_yr * scipy.constants.Julian_year  # sec

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt(semi_maj_axis ** 3 / (scipy.constants.G * smbh_mass))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0 - ecc ** 2)
    # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + ecc ** 2 + 2.0 * ecc * np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc ** 2 - 2.0 * ecc * np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc * np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc * np.cos(omega))
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0 / eta_plus ** 15) + np.sqrt(1.0 / eta_minus ** 15))
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus / eta_plus ** 2 + sigma_minus / eta_minus ** 2)
    # WZL Eqn 71
    #   NOTE: preserved disk_bh_retro_orbs_a in r_g to feed to disk_surf_density_func function
    #   tau in units of sec
    tau_i_dyn = np.sqrt(2.0) * inc * (delta - np.cos(inc)) ** 1.5 \
                * smbh_mass ** 2 * period / (
                            retro_mass * disk_surf_density_func(disk_bh_retro_orbs_a) * np.pi * semi_lat_rec ** 2) \
                / kappa

    # assume the fractional change in inclination is the fraction
    #   of tau_i_dyn represented by one timestep_duration_yr
    frac_change = timestep_duration_yr / tau_i_dyn

    # if the timescale for change of inclination is larger than the timestep_duration_yr
    #    send the new inclination to zero
    frac_change[frac_change > 1.0] = 1.0

    disk_bh_retro_orbs_ecc_new = inc * (1.0 - frac_change)

    return disk_bh_retro_orbs_ecc_new
