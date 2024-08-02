import numpy as np


def change_mass(prograde_stars_masses, frac_Eddington_ratio, mass_growth_Edd_rate, timestep):
    """Given initial black hole masses at start of timestep, add mass according to
        chosen stellar mass accretion prescription

    Parameters
    ----------
    prograde_stars_masses : float array
        initial masses of stars in prograde orbits around SMBH, units of solar masses
    frac_Eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stars
          in units of Eddington accretion rate. 1.0=embedded star accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    mass_growth_Edd_rate : float
        fractional rate of mass growth AT Eddington accretion rate per year (2.3e-8)
    timestep : float
        length of timestep in units of years

    Returns
    -------
    bh_new_masses : float array
        masses of black holes after accreting at prescribed rate for one timestep
    """
    # Mass grows exponentially for length of timestep:
    stars_new_masses = prograde_stars_masses*np.exp(mass_growth_Edd_rate*frac_Eddington_ratio*timestep)

    return stars_new_masses
