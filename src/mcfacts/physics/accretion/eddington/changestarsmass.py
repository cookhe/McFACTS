import numpy as np


def change_mass(disk_star_pro_masses,
                disk_star_eddington_ratio,
                mdisk_star_eddington_mass_growth_rate,
                timestep_duration_yr):
    """Given initial star masses at start of timestep, add mass according to
        chosen stellar mass accretion prescription

    Parameters
    ----------
    disk_star_pro_masses : float array
        initial masses of stars in prograde orbits around SMBH, units of solar masses
    disk_star_eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stars
          in units of Eddington accretion rate. 1.0=embedded star accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    mdisk_star_eddington_mass_growth_rate : float
        fractional rate of mass growth AT Eddington accretion rate per year (2.3e-8)
    timestep_duration_yr : float
        length of timestep in units of years

    Returns
    -------
    disk_star_pro_new_masses : float array
        masses of stars after accreting at prescribed rate for one timestep
    """
    # Mass grows exponentially for length of timestep:
    disk_star_pro_new_masses = disk_star_pro_masses*np.exp(mdisk_star_eddington_mass_growth_rate*disk_star_eddington_ratio*timestep_duration_yr)

    return disk_star_pro_new_masses
