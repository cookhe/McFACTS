import numpy as np
from mcfacts.objects.agnobject import obj_to_binary_bh_array


def bin_harden_baruteau(binary_bh_array, smbh_mass, timestep_duration_yr, time_gw_normalization, bin_index, time_passed):
    """Harden black hole binaries using Baruteau+11 prescription

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

    # 1. Run over active binaries
    # 2. Find number of binary orbits around its center of mass within the timestep
    # 3. For every 10^3 orbits, halve the binary separation.
    bindex = int(bin_index)
    for j in range(0, bindex):
        if binary_bh_array[11,j] < 0:
            # do nothing - merger happened!
            continue
        else:
            temp_bh_mass_1 = binary_bh_array[2,j]
            temp_bh_mass_2 = binary_bh_array[3,j]
            temp_bin_separation = binary_bh_array[8,j]
            temp_bin_ecc_around_com = binary_bh_array[13,j]
            temp_bin_mass = temp_bh_mass_1 + temp_bh_mass_2
            temp_bin_reduced_mass = (temp_bh_mass_1*temp_bh_mass_2)/temp_bin_mass
            #Find eccentricity factor (1-e_b^2)^7/2
            ecc_factor_1 = (1 - (temp_bin_ecc_around_com**(2.0)))**(3.5)
            # and eccentricity factor [1+(73/24)e_b^2+(37/96)e_b^4]
            ecc_factor_2 = [1+((73/24)*(temp_bin_ecc_around_com**2.0))+
                            ((37/96)*(temp_bin_ecc_around_com**4.0))]
            #overall ecc factor = ecc_factor_1/ecc_factor_2
            ecc_factor = ecc_factor_1/ecc_factor_2
            # Binary period = 2pi*sqrt((delta_r)^3/GM_bin)
            # or T_orb = 10^7s*(1r_g/m_smmbh=10^8Msun)^(3/2) *(M_bin/10Msun)^(-1/2) = 0.32yrs
            temp_bin_period = 0.32*((temp_bin_separation)**(1.5))*((smbh_mass/1.e8)**(1.5))*(temp_bin_mass/10.0)**(-0.5)
            #Find how many binary orbits in timestep. Binary separation is halved for every 10^3 orbits.
            if temp_bin_period > 0:
                temp_num_orbits_in_timestep = timestep_duration_yr/temp_bin_period
            else:
                temp_num_orbits_in_timestep = 0

            scaled_num_orbits=temp_num_orbits_in_timestep/1000.0
            #Timescale for binary merger via GW emission alone, scaled to bin parameters
            temp_bin_t_gw = time_gw_normalization*((temp_bin_separation)**(4.0))*((temp_bin_mass/10.0)**(-2))*((temp_bin_reduced_mass/2.5)**(-1.0))*ecc_factor
            binary_bh_array[10,j]=temp_bin_t_gw

            if temp_bin_t_gw > timestep_duration_yr:
                #Binary will not merge in this timestep.
                #Write new bin_separation according to Baruteau+11 prescription
                new_temp_bin_separation = temp_bin_separation*((0.5)**(scaled_num_orbits))
                binary_bh_array[8,j] = new_temp_bin_separation
            else:
                #Binary will merge in this timestep.
                #Return a merger in bin_array! A negative flag on this line indicates merger.
                binary_bh_array[11,j] = -2
                binary_bh_array[12,j] = time_passed

    return binary_bh_array


def bin_harden_baruteau_obj(blackholes_binary, smbh_mass, timestep_duration_yr, time_gw_normalization, time_passed):

    binary_bh_array = obj_to_binary_bh_array(blackholes_binary)

    bin_index = blackholes_binary.num

    binary_bh_array = bin_harden_baruteau(binary_bh_array, smbh_mass, timestep_duration_yr,
                                          time_gw_normalization, bin_index, time_passed)

    blackholes_binary.time_to_merger_gw = binary_bh_array[10, :]
    blackholes_binary.bin_sep = binary_bh_array[8, :]
    blackholes_binary.flag_merging = binary_bh_array[11, :]
    blackholes_binary.time_merged = binary_bh_array[12, :]

    return (blackholes_binary)