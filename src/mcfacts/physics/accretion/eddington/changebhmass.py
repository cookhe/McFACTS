import numpy as np


def change_mass(disk_bh_pro_masses, disk_bh_eddington_ratio, disk_bh_eddington_mass_growth_rate, timestep_duration_yr):
    """Given initial black hole masses at start of timestep, add mass according to
        chosen BH mass accretion prescription

    Parameters
    ----------
    disk_bh_pro_masses : float array
        initial masses of black holes in prograde orbits around SMBH, units of solar masses
    disk_bh_eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass 
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    mdisk_bh_eddington_mass_growth_rate : float
        fractional rate of mass growth AT Eddington accretion rate per year (2.3e-8)
    timestep_duration_yr : float
        length of timestep in units of years

    Returns
    -------
    disk_bh_pro_new_masses : float array
        masses of black holes after accreting at prescribed rate for one timestep
    """
    # Mass grows exponentially for length of timestep:
    disk_bh_pro_new_masses = disk_bh_pro_masses*np.exp(disk_bh_eddington_mass_growth_rate*disk_bh_eddington_ratio*timestep_duration_yr)

    return disk_bh_pro_new_masses
