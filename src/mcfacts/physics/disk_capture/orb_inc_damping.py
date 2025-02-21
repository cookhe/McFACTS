import numpy as np

import scipy

def bin_orb_inc_damping(mass_smbh,binary_bh_array,retro_arg_periapse,timestep,disk_surf_model):
    """This function calculates how fast the inclination angle of a BBH
    changes due to dynamical friction (appropriate for BH, NS, maaaybe WD?--check) using
    Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL). It returns the new locations of the retrograde
    orbiters after 1 timestep. Note we have assumed the masses of the orbiters are
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
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    retrograde_bh_locations : float array
        locations of retrograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
        (actually we assume this is the same as the semi-major axis)
    retrograde_bh_masses : float array
        mass of retrograde singleton BH at start of timestep in units of solar masses
    retrograde_bh_orb_ecc : float array
        orbital eccentricity of retrograde singleton BH at start of timestep.
    retrograde_bh_orb_inc : float array
        orbital inclination of retrograde singleton BH at start of timestep.
    retro_arg_periapse : float array
        argument of periapse of retrograde singleton BH at start of timestep.
    timestep : float
        size of timestep in years
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    new_orb_inc : float array
        orbital inclinations of retrograde singletons BH at end of timestep
    """
    # This should probably be set somewhere for the whole code? But...
    KgPerMsun = 1.99e30

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    mass_smbh = mass_smbh * KgPerMsun  # kg
    semi_maj_axis = retrograde_bh_locations * scipy.constants.G * mass_smbh \
                    / (scipy.constants.c)**2  # m
    retro_mass = retrograde_bh_masses * KgPerMsun  # kg
    omega = retro_arg_periapse  # radians
    ecc = retrograde_bh_orb_ecc  # unitless
    inc = retrograde_bh_orb_inc  # radians
    timestep = timestep * scipy.constants.Julian_year # sec

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt(semi_maj_axis**3/(scipy.constants.G * mass_smbh))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0-ecc**2)
        # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + ecc**2 + 2.0*ecc*np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc**2 - 2.0*ecc*np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc*np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc*np.cos(omega))
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0/eta_plus**15) + np.sqrt(1.0/eta_minus**15))
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus/eta_plus**2 + sigma_minus/eta_minus**2)
    # WZL Eqn 71
    #   NOTE: preserved retrograde_bh_locations in r_g to feed to disk_surf_model function
    #   tau in units of sec
    tau_i_dyn = np.sqrt(2.0) * inc * (delta - np.cos(inc))**1.5 \
                * mass_smbh**2 * period / (retro_mass*disk_surf_model(retrograde_bh_locations)*np.pi*semi_lat_rec**2) \
                / kappa
    
    # assume the fractional change in inclination is the fraction
    #   of tau_i_dyn represented by one timestep
    frac_change = timestep / tau_i_dyn

    # if the timescale for change of inclination is larger than the timestep
    #    send the new inclination to zero
    frac_change[frac_change>1.0] = 1.0

    new_orb_inc = inc * (1.0 - frac_change)

    return new_orb_inc    

def orb_inc_damping(mass_smbh,retrograde_bh_locations,retrograde_bh_masses,retrograde_bh_orb_ecc,retrograde_bh_orb_inc,retro_arg_periapse,timestep,disk_surf_model):
    """This function calculates how fast the inclination angle of an arbitrary single orbiter
    changes due to dynamical friction (appropriate for BH, NS, maaaybe WD?--check) using
    Wang, Zhu & Lin 2024, MNRAS, 528, 4958 (WZL). It returns the new locations of the retrograde
    orbiters after 1 timestep. Note we have assumed the masses of the orbiters are
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
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    retrograde_bh_locations : float array
        locations of retrograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
        (actually we assume this is the same as the semi-major axis)
    retrograde_bh_masses : float array
        mass of retrograde singleton BH at start of timestep in units of solar masses
    retrograde_bh_orb_ecc : float array
        orbital eccentricity of retrograde singleton BH at start of timestep.
    retrograde_bh_orb_inc : float array
        orbital inclination of retrograde singleton BH at start of timestep.
    retro_arg_periapse : float array
        argument of periapse of retrograde singleton BH at start of timestep.
    timestep : float
        size of timestep in years
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g

    Returns
    -------
    new_orb_inc : float array
        orbital inclinations of retrograde singletons BH at end of timestep
    """
    # This should probably be set somewhere for the whole code? But...
    KgPerMsun = 1.99e30

    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    mass_smbh = mass_smbh * KgPerMsun  # kg
    semi_maj_axis = retrograde_bh_locations * scipy.constants.G * mass_smbh \
                    / (scipy.constants.c)**2  # m
    retro_mass = retrograde_bh_masses * KgPerMsun  # kg
    omega = retro_arg_periapse  # radians
    ecc = retrograde_bh_orb_ecc  # unitless
    inc = retrograde_bh_orb_inc  # radians
    timestep = timestep * scipy.constants.Julian_year # sec

    # period in units of sec
    period = 2.0 * np.pi * np.sqrt(semi_maj_axis**3/(scipy.constants.G * mass_smbh))
    # semi-latus rectum in units of meters
    semi_lat_rec = semi_maj_axis * (1.0-ecc**2)
        # WZL Eqn 7 (sigma+/-)
    sigma_plus = np.sqrt(1.0 + ecc**2 + 2.0*ecc*np.cos(omega))
    sigma_minus = np.sqrt(1.0 + ecc**2 - 2.0*ecc*np.cos(omega))
    # WZL Eqn 8 (eta+/-)
    eta_plus = np.sqrt(1.0 + ecc*np.cos(omega))
    eta_minus = np.sqrt(1.0 - ecc*np.cos(omega))
    # WZL Eqn 62
    kappa = 0.5 * (np.sqrt(1.0/eta_plus**15) + np.sqrt(1.0/eta_minus**15))
    # WZL Eqn 30
    delta = 0.5 * (sigma_plus/eta_plus**2 + sigma_minus/eta_minus**2)
    # WZL Eqn 71
    #   NOTE: preserved retrograde_bh_locations in r_g to feed to disk_surf_model function
    #   tau in units of sec
    tau_i_dyn = np.sqrt(2.0) * inc * (delta - np.cos(inc))**1.5 \
                * mass_smbh**2 * period / (retro_mass*disk_surf_model(retrograde_bh_locations)*np.pi*semi_lat_rec**2) \
                / kappa
    
    # assume the fractional change in inclination is the fraction
    #   of tau_i_dyn represented by one timestep
    frac_change = timestep / tau_i_dyn

    # if the timescale for change of inclination is larger than the timestep
    #    send the new inclination to zero
    frac_change[frac_change>1.0] = 1.0

    new_orb_inc = inc * (1.0 - frac_change)

    return new_orb_inc    