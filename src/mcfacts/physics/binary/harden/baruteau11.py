import numpy as np


def bin_harden_baruteau(blackholes_binary, smbh_mass, timestep_duration_yr,
                        time_gw_normalization, time_passed):
    """
    Harden black hole binaries using Baruteau+11 prescription

    Use Baruteau+11 prescription to harden a pre-existing binary.
    For every 1000 orbits of binary around its center of mass, the
    separation (between binary components) is halved.

    Parameters
    ----------
    binary_bh_array : ndarray
        Array of binary black holes in the disk.
    smbh_mass : ndarray
        Mass of supermassive black hole.
    timestep_duration_yr : float
        Length of timestep of the simulation in years.
    time_gw_normalization : float
        A normalization for GW decay timescale, set by `smbh_mass` & normalized for
        a binary total mass of 10 solar masses.
    bin_index : int
        Count of number of binaries
    time_passed : float
        Time elapsed since beginning of simulation.

    Returns
    -------
    ndarray
        Updated array of black hole binaries in the disk.
    """

    # 1. Find active binaries
    # 2. Find number of binary orbits around its center of mass within the timestep
    # 3. For every 10^3 orbits, halve the binary separation.


    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)[0]

    # If all binaries have merged then nothing to do
    if (idx_non_mergers.shape[0] == 0):
        return blackholes_binary

    # Set up variables
    mass_binary = blackholes_binary.mass_1[idx_non_mergers] + blackholes_binary.mass_2[idx_non_mergers]
    mass_reduced = (blackholes_binary.mass_1[idx_non_mergers] * blackholes_binary.mass_2[idx_non_mergers]) / mass_binary
    bin_sep = blackholes_binary.bin_sep[idx_non_mergers]
    bin_orb_ecc = blackholes_binary.bin_ecc[idx_non_mergers]

    # Find eccentricity factor (1-e_b^2)^7/2
    ecc_factor_1 = np.power(1 - np.power(bin_orb_ecc, 2), 3.5)
    # and eccentricity factor [1+(73/24)e_b^2+(37/96)e_b^4]
    ecc_factor_2 = 1 + ((73/24) * np.power(bin_orb_ecc, 2)) + ((37/96) * np.power(bin_orb_ecc, 4))
    # overall ecc factor = ecc_factor_1/ecc_factor_2
    ecc_factor = ecc_factor_1/ecc_factor_2

    # Binary period = 2pi*sqrt((delta_r)^3/GM_bin)
    # or T_orb = 10^7s*(1r_g/m_smmbh=10^8Msun)^(3/2) *(M_bin/10Msun)^(-1/2) = 0.32yrs
    bin_period = 0.32 * np.power(bin_sep, 1.5) * np.power(smbh_mass/1.e8, 1.5) * np.power(mass_binary/10.0, -0.5)

    # Find how many binary orbits in timestep. Binary separation is halved for every 10^3 orbits.
    num_orbits_in_timestep = np.zeros(len(bin_period))
    num_orbits_in_timestep[bin_period > 0] = timestep_duration_yr / bin_period[bin_period > 0]
    scaled_num_orbits = num_orbits_in_timestep / 1000.0

    # Timescale for binary merger via GW emission alone, scaled to bin parameters
    time_to_merger_gw = time_gw_normalization*((bin_sep)**(4.0))*((mass_binary/10.0)**(-2))*((mass_reduced/2.5)**(-1.0))*ecc_factor
    # Finite check
    assert np.isfinite(time_to_merger_gw).all(),\
        "Finite check failure: time_to_merger_gw"
    blackholes_binary.time_to_merger_gw[idx_non_mergers] = time_to_merger_gw

    # Binary will not merge in this timestep
    # new bin_sep according to Baruteu+11 prescription
    bin_sep[time_to_merger_gw > timestep_duration_yr] = bin_sep[time_to_merger_gw > timestep_duration_yr] * np.power(0.5, scaled_num_orbits[time_to_merger_gw > timestep_duration_yr])
    blackholes_binary.bin_sep[idx_non_mergers[time_to_merger_gw > timestep_duration_yr]] = bin_sep[time_to_merger_gw > timestep_duration_yr]
    # Finite check
    assert np.isfinite(blackholes_binary.bin_sep).all(),\
        "Finite check failure: blackholes_binary.bin_sep"

    # Otherwise binary will merge in this timestep
    # Update flag_merging to -2 and time_merged to current time
    blackholes_binary.flag_merging[idx_non_mergers[time_to_merger_gw <= timestep_duration_yr]] = np.full(np.sum(time_to_merger_gw <= timestep_duration_yr), -2)
    blackholes_binary.time_merged[idx_non_mergers[time_to_merger_gw <= timestep_duration_yr]] = np.full(np.sum(time_to_merger_gw <= timestep_duration_yr), time_passed)
    # Finite check
    assert np.isfinite(blackholes_binary.flag_merging).all(),\
        "Finite check failure: blackholes_binary.flag_merging"
    # Finite check
    assert np.isfinite(blackholes_binary.time_merged).all(),\
        "Finite check failure: blackholes_binary.time_merged"

    return (blackholes_binary)
