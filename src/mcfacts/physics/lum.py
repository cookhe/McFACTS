"""
Module for calculating luminosities produced by merger remnant interacting with gas via ram-pressure stripping or jet formation.
"""

import numpy as np

from astropy import units as u
from astropy import constants as const

from mcfacts.physics.point_masses import si_from_r_g


def shock_luminosity(smbh_mass,
        mass_final,
        bin_orb_a,
        disk_aspect_ratio,
        disk_density,
        vk):
    """
    Estimate the shock luminosity from the interaction between a merger remnant 
    and gas within its Hill sphere.

    Based on McKernan et al. (2019) (arXiv:1907.03746v2), this function computes:
    - The Hill radius of the remnant system.
    - The gas volume inside the Hill sphere, accounting for the vertical extent of the disk.
    - The mass of gas available for interaction.
    - The shock energy and characteristic timescale over which energy is radiated.

    The shock luminosity is given by:
        L_shock ≈ E / t,
    where
        E = 1e47 erg * (M_gas / M_sun) * (v_kick / 200 km/s)^2
        t ~ R_Hill / v_kick

    Parameters:
    ----------
    smbh_mass : float
        Mass of the supermassive black hole (in solar masses).
    mass_final : numpy.ndarray
        Final mass of the binary black hole remnant (in solar masses).
    bin_orb_a : numpy.ndarray
        Orbital separation between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_aspect_ratio : callable
        Function that returns the aspect ratio (height/radius) of the disk at a given radius.
    disk_density : callable
        Function that returns the gas density at a given radius (in kg m^-3).
    vk : numpy.ndarray
        Kick velocity imparted to the remnant (in km/s).

    Returns:
    -------
    Lshock : float
        Estimated shock luminosity (in erg/s).
    """
    r_hill_rg = bin_orb_a * ((mass_final / smbh_mass) / 3)**(1/3) 
    r_hill_m = si_from_r_g(smbh_mass, r_hill_rg)

    r_hill_cm = r_hill_m.cgs.value

    disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a
    disk_height_m = si_from_r_g(smbh_mass, disk_height_rg)
    disk_height_cm = disk_height_m.cgs.value

    v_hill = (4 / 3) * np.pi * r_hill_cm**3  
    v_hill_gas = abs(v_hill - (2 / 3) * np.pi * ((r_hill_cm - disk_height_cm)**2) * (3 * r_hill_cm - (r_hill_cm - disk_height_cm)))
    
    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)
    disk_density_cgs = disk_density_si.cgs

    disk_density_cgs = disk_density_cgs.value
    msolar = const.M_sun.cgs.value

    r_hill_mass = (disk_density_cgs * v_hill_gas) / msolar

    v_kick_scale = 200. * (u.km / u.s)
    v_kick_scale = v_kick_scale.value
    E = 10**46 * (r_hill_mass / 1) * (vk / v_kick_scale)**2  # Energy of the shock
    time = 31556952.0 * ((r_hill_rg / 3) / (vk / v_kick_scale))  # Timescale for energy dissipation
    Lshock = E / time  # Shock luminosity

    assert np.all(Lshock > 0), \
        "Lshock has values <= 0"

    return Lshock


def gas_capture_rate(
    bin_orb_a,
    disk_aspect_ratio,
    disk_density,  # g/cm^3
    mass_final,       # M_sun
    smbh_mass      # M_sun
):
    """
    Compute the gas capture rate for a stellar-mass black hole in an AGN disk.
    
    Returns:
        float: Capture rate in solar masses per year.
    """
    # Base capture rate coefficient in M_sun/year
    fc = 10
    r_hill_rg = bin_orb_a * ((mass_final / smbh_mass) / 3)**(1/3) 
    r_hill_m = si_from_r_g(smbh_mass, r_hill_rg)

    r_hill_cm = r_hill_m.cgs.value 

    disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a
    disk_height_m = si_from_r_g(smbh_mass, disk_height_rg)
    disk_height_pc = disk_height_m.to('pc').value

    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)
    disk_density_cgs = disk_density_si.cgs.value

    bin_orb_a_pc = si_from_r_g(smbh_mass, bin_orb_a).to('pc').value

    base_rate = 3e-3    # Base capture rate coefficient in M_sun/year
    rate = base_rate * (fc/10) * (disk_height_pc / 0.003)**2 * (bin_orb_a_pc / 1.0)**(-1.5) 
    rate *= (disk_density_cgs / 1e-17) * (mass_final / 10)**2 * (smbh_mass / 1e6)**(-1.5)
    return rate


def jet_luminosity(mass_final,
        bin_orb_a,
        disk_density,
        disk_aspect_ratio,
        smbh_mass,
        spin_final,
        vk):
    """
    Estimate the jet luminosity produced by Bondi-Hoyle-Lyttleton (BHL) accretion.

    Based on Graham et al. (2020), the luminosity goes as:
        L_BHL ≈ 2.5e45 erg/s * (η / 0.1) * (M / 100 M_sun)^2 * (v / 200 km/s)^-3 * (rho / 1e-9 g/cm^3)

    Parameters:
    ----------
    mass_final : numpy.ndarray
        mass of remnant post-merger (mass loss accounted for via Tichy & Maronetti 08)
    bin_orb_a : numpy.ndarray
        Orbital separation between the SMBH and the binary at the time of merger (in gravitational radii).
    disk_density : callable
        Function that returns the gas density at a given radius (in kg m^-3).
    vk : numpy.ndarray
        Kick velocity imparted to the remnant (in km/s).

    Returns:
    -------
    LBHL : numpy.ndarray
        Estimated jet (Bondi-Hoyle) luminosity (in erg/s).
    """

    disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)

    disk_density_cgs = disk_density_si.to(u.g / u.cm**3)
    disk_density_cgs = disk_density_cgs.value
    # eta depends on spin t.f. isco, assuming eddington accretion... .06-.42 and so 0.1 is a good O approx.
    # but.. bondi is greater mass accretion rate, t.f. L per mass acrreted will be less becayse so much shit 
    # is trying to get in in so little space for light to escape
    v_kick_scale = 200. * (u.km / u.s)
    v_kick_scale = v_kick_scale.value

    #mcap = gas_capture_rate(bin_orb_a, disk_aspect_ratio, disk_density, mass_final, smbh_mass)
    eta = spin_final**2

    #LBHL = 1e42 * (mcap / 3e-4) * (eta / 0.5) * (0.1 / 0.1)
    LBHL = 2.5e45 * (eta / 0.1) * (mass_final / 100)**2 * (vk / v_kick_scale)**-3 * (disk_density_cgs / 10e-10)  # Jet luminosity

    assert np.all(LBHL > 0), \
        "LBHL has values <= 0"

    return LBHL