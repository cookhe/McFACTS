"""
Module for calculating the timescale of migrations.
"""

import numpy as np
import scipy


def type1_migration_old(smbh_mass, disk_bh_orb_a_pro, disk_bh_mass_pro, disk_surf_density_func, disk_aspect_ratio_func, timestep_duration_yr, disk_feedback_ratio_func, disk_radius_trap, disk_bh_orb_ecc_pro, disk_bh_pro_orb_ecc_crit,disk_radius_outer):
    """Calculates how far an object migrates in an AGN gas disk in a single timestep

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    disk_bh_orb_a_pro : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_mass_pro : numpy.ndarray
        Masses [M_sun] of prograde singleton BH at start of timestep with :obj:`float` type
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_bh_orb_ecc_pro : numpy.ndarray
        Orbital ecc [unitless]of prograde singleton BH at start of timestep :math:`\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    disk_bh_pro_orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)

    Returns
    -------
    bh_new_locations : float array
        Semi-major axes [r_{g,SMBH}] of prograde singleton BH at end of timestep
    """
    # get surface density function, or process if just a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(disk_bh_orb_a_pro)
    # get aspect ratio function, or process if just a float
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(disk_bh_orb_a_pro)

    # Migration can only occur for sufficiently damped orbital ecc. If orb ecc <= e_crit, then migration.
    # Otherwise, no change in semi-major axis. Wait till orb ecc damped to <=e_crit.
    # Only show BH with orb ecc <=e_crit
    #BUG below line is original
    #disk_bh_orb_ecc_pro = np.ma.masked_where(disk_bh_orb_ecc_pro > disk_bh_pro_orb_ecc_crit, disk_bh_orb_ecc_pro)
    disk_bh_mig = np.ma.masked_where(disk_bh_orb_ecc_pro > disk_bh_pro_orb_ecc_crit, disk_bh_orb_ecc_pro)
    # Those BH with orb ecc > e_crit
    disk_bh_not_mig = np.ma.masked_where(disk_bh_orb_ecc_pro <= disk_bh_pro_orb_ecc_crit, disk_bh_orb_ecc_pro)
    # Indices of BH with <=critical ecc
    disk_bh_crit_ecc_pro_indices = np.ma.nonzero(disk_bh_mig)
    # Indicies of BH with > critical ecc
    disk_bh_crit_ecc_not_mig = np.ma.nonzero(disk_bh_not_mig)

    # Migration only if there are BH with e<=e_crit
    # if np.size(crit_ecc_prograde_indices) > 0:
    # compute migration timescale for each orbiter in seconds
    # eqn from Paardekooper 2014, rewritten for R in terms of r_g of SMBH = GM_SMBH/c^2
    # tau = (pi/2) h^2/(q_d*q) * (1/Omega)
    #   where h is aspect ratio, q is m/M_SMBH, q_d = pi R^2 disk_surface_density/M_SMBH
    #   and Omega is the Keplerian orbital frequency around the SMBH
    # here smbh_mass/disk_bh_mass_pro are both in M_sun, so units cancel
    # c, G and disk_surface_density in SI units
    tau = ((disk_aspect_ratio**2)* scipy.constants.c/(3.0*scipy.constants.G) * (smbh_mass/disk_bh_mass_pro) / disk_surface_density) / np.sqrt(disk_bh_orb_a_pro)
    # ratio of timestep to tau_mig (timestep in years so convert)
    dt = timestep_duration_yr * scipy.constants.year / tau
    # migration distance is original locations times fraction of tau_mig elapsed
    disk_bh_dist_mig = disk_bh_orb_a_pro * dt
    # Mask migration distance with zeros if orb ecc >= e_crit.
    disk_bh_dist_mig[disk_bh_crit_ecc_not_mig] = 0.

    # Feedback provides a universal modification of migration distance
    # If feedback off, then feedback_ratio= ones and migration is unchanged
    # Construct empty array same size as prograde_bh_locations

    disk_bh_pro_a_new = np.zeros_like(disk_bh_orb_a_pro)

    # Find indices of objects where feedback ratio <1; these still migrate inwards, but more slowly
    # feedback ratio is a tuple, so need [0] part not [1] part (ie indices not details of array)
    disk_bh_mig_inward_mod = np.where(disk_feedback_ratio_func < 1)[0]
    disk_bh_mig_inward_all = disk_bh_orb_a_pro[disk_bh_mig_inward_mod]

    # Given a population migrating inwards
    if disk_bh_mig_inward_mod.size > 0:
        for i in range(0, disk_bh_mig_inward_mod.size):
            # Among all inwards migrators, find location in disk & compare to trap radius
            disk_bh_mig_inward_index = disk_bh_mig_inward_mod[i]

            # If outside trap, migrates inwards
            if disk_bh_mig_inward_all[i] > disk_radius_trap:
                disk_bh_pro_a_new[disk_bh_mig_inward_index] = disk_bh_orb_a_pro[disk_bh_mig_inward_index] - (disk_bh_dist_mig[disk_bh_mig_inward_index]*(1-disk_feedback_ratio_func[disk_bh_mig_inward_index]))
                # If inward migration takes object inside trap, fix at trap.
                if disk_bh_pro_a_new[disk_bh_mig_inward_index] <= disk_radius_trap:
                    disk_bh_pro_a_new[disk_bh_mig_inward_index] = disk_radius_trap
            # If inside trap, migrate outwards
            elif disk_bh_mig_inward_all[i] < disk_radius_trap:
                disk_bh_pro_a_new[disk_bh_mig_inward_index] = disk_bh_orb_a_pro[disk_bh_mig_inward_index] + (disk_bh_dist_mig[disk_bh_mig_inward_index]*(1-disk_feedback_ratio_func[disk_bh_mig_inward_index]))
                #If outward migration takes object outside trap, fix at trap.
                if disk_bh_pro_a_new[disk_bh_mig_inward_index] >= disk_radius_trap:
                    disk_bh_pro_a_new[disk_bh_mig_inward_index] = disk_radius_trap
            # If at trap, stays there
            elif disk_bh_mig_inward_all[i] == disk_radius_trap:
                disk_bh_pro_a_new[disk_bh_mig_inward_index] = disk_bh_orb_a_pro[disk_bh_mig_inward_index]
            # Something wrong has happened
            else:
                raise RuntimeError("Forbidden case")

    # Find indices of objects where feedback ratio >1; these migrate outwards.
    # In Sirko & Goodman (2003) disk model this is well outside migration trap region.
    disk_bh_mig_outward_mod = np.where(disk_feedback_ratio_func >1)[0]

    if disk_bh_mig_outward_mod.size > 0:
        disk_bh_pro_a_new[disk_bh_mig_outward_mod] = disk_bh_orb_a_pro[disk_bh_mig_outward_mod] +(disk_bh_dist_mig[disk_bh_mig_outward_mod]*(disk_feedback_ratio_func[disk_bh_mig_outward_mod]-1))
        # catch to keep stuff from leaving the outer radius of the disk
        disk_bh_pro_a_new[disk_bh_mig_outward_mod[np.where(disk_bh_pro_a_new[disk_bh_mig_outward_mod] > disk_radius_outer)]] = disk_radius_outer

    # Find indices where feedback ratio is identically 1; shouldn't happen (edge case) if feedback on, but == 1 if feedback off.
    disk_bh_mig_unchanged = np.where(disk_feedback_ratio_func == 1)[0]
    if disk_bh_mig_unchanged.size > 0:
        # If BH location > trap radius, migrate inwards
        for i in range(0, disk_bh_mig_unchanged.size):
            disk_bh_mig_unchanged_index = disk_bh_mig_unchanged[i]
            if disk_bh_orb_a_pro[disk_bh_mig_unchanged_index] > disk_radius_trap:
                disk_bh_pro_a_new[disk_bh_mig_unchanged_index] = disk_bh_orb_a_pro[disk_bh_mig_unchanged_index] - disk_bh_dist_mig[disk_bh_mig_unchanged_index]
            # if new location is <= trap radius, set location to trap radius
                if disk_bh_pro_a_new[disk_bh_mig_unchanged_index] <= disk_radius_trap:
                    disk_bh_pro_a_new[disk_bh_mig_unchanged_index] = disk_radius_trap

            # If BH location < trap radius, migrate outwards
            if disk_bh_orb_a_pro[disk_bh_mig_unchanged_index] < disk_radius_trap:
                disk_bh_pro_a_new[disk_bh_mig_unchanged_index] = disk_bh_orb_a_pro[disk_bh_mig_unchanged_index] + disk_bh_dist_mig[disk_bh_mig_unchanged_index]
                # if new location is >= trap radius, set location to trap radius
                if disk_bh_pro_a_new[disk_bh_mig_unchanged_index] >= disk_radius_trap:
                    disk_bh_pro_a_new[disk_bh_mig_unchanged_index] = disk_radius_trap
    # print("bh new locations",np.sort(bh_new_locations))
    # print('migration distance2',migration_distance, prograde_bh_orb_ecc)
    # new locations are original ones - distance traveled
    # bh_new_locations = prograde_bh_locations - migration_distance
    # Assert that things are not allowed to migrate out of the disk.
    mask_disk_radius_outer = disk_radius_outer < disk_bh_pro_a_new
    disk_bh_pro_a_new[mask_disk_radius_outer] = disk_radius_outer
    if np.sum(disk_bh_pro_a_new == 0) > 0:
        empty_mask = disk_bh_pro_a_new == 0
        print("empty_mask:",np.where(empty_mask))
        print("smbh_mass:",smbh_mass)
        print("disk_radius_trap:",disk_radius_trap)
        print("disk_radius_outer:",disk_radius_outer)
        print("disk_bh_pro_orb_ecc_crit:",disk_bh_pro_orb_ecc_crit)
        print("disk_bh_orb_ecc_pro:",disk_bh_orb_ecc_pro[empty_mask])
        print("disk_bh_orb_a_pro:", disk_bh_orb_a_pro[empty_mask])
        print("disk_bh_mass_pro:", disk_bh_mass_pro[empty_mask])
        # Toss the binaries
        disk_bh_pro_a_new[empty_mask] = 0.
        raise RuntimeError("disk_bh_pro_a_new was not set properly; a case was missed")
    return disk_bh_pro_a_new


def type1_migration(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                    disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                    disk_radius_trap, disk_radius_outer, timestep_duration_yr):
    """Calculates how far an object migrates in an AGN gas disk in a single timestep

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] wrt to SMBH of objects at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    masses : numpy.ndarray
        Masses [M_sun] of objects at start of timestep with :obj:`float` type
    orbs_ecc : numpy.ndarray
        Orbital ecc [unitless] wrt to SMBH of objects at start of timestep :math:`\\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    orbs_a : float array
        Semi-major axes [r_{g,SMBH}] of objects at end of timestep
    """

    # Migration only occurs for sufficiently damped orbital ecc. If orb_ecc <= ecc_crit, then migrate.
    # Otherwise no change in semi-major axis (orb_a).
    # Get indices of objects with orb_ecc <= ecc_crit so we can only update orb_a for those.
    migration_indices = np.asarray(orbs_ecc <= orb_ecc_crit).nonzero()[0]
    # If nothing will migrate then end the function
    if migration_indices.shape == (0,):
        # BUG it shouldn't work like this
        orbs_a[orbs_a > disk_radius_outer] = disk_radius_outer
        return (orbs_a)

    # If things will migrate then copy over the orb_a of objects that will migrate
    new_orbs_a = orbs_a[migration_indices].copy()

    # Get surface density function or process if just a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(orbs_a)[migration_indices]
    # Get aspect ratio function or process if just a float
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(orbs_a)[migration_indices]

    disk_feedback_ratio = disk_feedback_ratio_func[migration_indices]

    # Compute migration timescale for each orbiter in seconds
    # Eqn from Paardekooper 2014, rewritten for R in terms of r_g of SMBH = GM_SMBH/c^2
    # tau = (pi/2) h^2/(q_d*q) * (1/Omega)
    #   where h is aspect ratio, q is m/M_SMBH, q_d = pi R^2 disk_surface_density/M_SMBH
    #   and Omega is the Keplerian orbital frequency around the SMBH
    # Here smbh_mass/disk_bh_mass_pro are both in M_sun, so units cancel
    # c, G and disk_surface_density in SI units
    tau = ((disk_aspect_ratio ** 2.0) * scipy.constants.c / (3.0 * scipy.constants.G) * (smbh_mass/masses[migration_indices]) / disk_surface_density) / np.sqrt(new_orbs_a)
    # ratio of timestep to tau_mig (timestep in years so convert)
    dt = timestep_duration_yr * scipy.constants.year / tau
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = new_orbs_a.copy() * dt

    # Get masks for if objects are inside or outside the trap radius
    mask_out_trap = new_orbs_a > disk_radius_trap
    mask_in_trap = new_orbs_a < disk_radius_trap

    # Get mask for objects where feedback_ratio <1; these still migrate inwards, but more slowly
    mask_mig_in = disk_feedback_ratio < 1
    if (np.sum(mask_mig_in) > 0):
        # If outside trap migrate inwards
        temp_orbs_a = new_orbs_a[mask_mig_in & mask_out_trap] - migration_distance[mask_mig_in & mask_out_trap] * (1 - disk_feedback_ratio[mask_mig_in & mask_out_trap])
        # If migration takes object inside trap, fix at trap
        temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap
        new_orbs_a[mask_mig_in & mask_out_trap] = temp_orbs_a

        # If inside trap, migrate outwards
        temp_orbs_a = new_orbs_a[mask_mig_in & mask_in_trap] + migration_distance[mask_mig_in & mask_in_trap] * (1 - disk_feedback_ratio[mask_mig_in & mask_in_trap])
        # If migration takes object outside trap, fix at trap
        temp_orbs_a[temp_orbs_a >= disk_radius_trap] = disk_radius_trap
        new_orbs_a[mask_mig_in & mask_in_trap] = temp_orbs_a

    # Get mask for objects where feedback_ratio > 1: these migrate outwards
    mask_mig_out = disk_feedback_ratio > 1
    if (np.sum(mask_mig_out) > 0):
        new_orbs_a[mask_mig_out] = new_orbs_a[mask_mig_out] + migration_distance[mask_mig_out] * (disk_feedback_ratio[mask_mig_out] - 1)

    # Get mask for objects where feedback_ratio == 1. Shouldn't happen if feedback = 1 (on), but will happen if feedback = 0 (off)
    mask_mig_stay = disk_feedback_ratio == 1
    if (np.sum(mask_mig_stay) > 0):
        # If outside trap migrate inwards
        temp_orbs_a = new_orbs_a[mask_mig_stay & mask_out_trap] - migration_distance[mask_mig_stay & mask_out_trap]
        # If migration takes object inside trap, fix at trap
        temp_orbs_a[temp_orbs_a <= disk_radius_trap] = disk_radius_trap
        new_orbs_a[mask_mig_stay & mask_out_trap] = temp_orbs_a

        # If inside trap migrate outwards
        temp_orbs_a = new_orbs_a[mask_mig_stay & mask_in_trap] + migration_distance[mask_mig_stay & mask_in_trap]
        # If migration takes object outside trap, fix at trap
        temp_orbs_a[temp_orbs_a >= disk_radius_trap] = disk_radius_trap
        new_orbs_a[mask_mig_stay & mask_in_trap] = temp_orbs_a

    # Assert that things cannot migrate out of the disk
    new_orbs_a[new_orbs_a > disk_radius_outer] = disk_radius_outer

    # Update orbs_a
    orbs_a[migration_indices] = new_orbs_a
    orbs_a[orbs_a > disk_radius_outer] = disk_radius_outer  # BUG should not work like this
    return (orbs_a)


def type1_migration_single(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                           disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                           disk_radius_trap, disk_radius_outer, timestep_duration_yr):
    """Wrapper function for type1_migration for single objects in the disk.

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] wrt to SMBH of objects at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    masses : numpy.ndarray
        Masses [M_sun] of objects at start of timestep with :obj:`float` type
    orbs_ecc : numpy.ndarray
        Orbital ecc [unitless] wrt to SMBH of objects at start of timestep :math:`\\mathtt{disk_radius_trap}. Floor in orbital ecc given by e_crit.
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    new_orbs_a : float array
        Semi-major axes [r_{g,SMBH}] of objects at end of timestep
    """

    new_orbs_a = type1_migration(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                                 disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                                 disk_radius_trap, disk_radius_outer, timestep_duration_yr)

    return (new_orbs_a)


def type1_migration_binary(smbh_mass, blackholes_binary, orb_ecc_crit,
                           disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                           disk_radius_trap, disk_radius_outer, timestep_duration_yr):
    """Wrapper function for type1_migration for binaries in the disk.

    Assumes a gas disk surface density and aspect ratio profile, for objects of specified masses and
    starting locations, and returns their new locations after migration over one timestep.

    This function replaces dr_migration which did not include smbh mass and was unreadable.

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    blackholes_binary : AGNBinaryBlackHole
        Binary black hole parameters, including mass_1, mass_2, bin_orb_a, and bin_orb_ecc
    orb_ecc_crit : float
        Critical value of orbital eccentricity [unitless] below which we assume Type 1 migration must occur. Do not damp orb ecc below this (e_crit=0.01 is default)
    disk_surf_density_func : function
        Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        Returns AGN gas disk aspect ratio [unitless] given a distance [r_{g,SMBH}] from the SMBH
        can accept a simple float (constant), but this is deprecated
    disk_feedback_ratio_func : function
        Ratio of heating/migration torque [unitless]. If ratio <1, migration inwards, but slows by factor tau_mig/(1-R)
        if ratio >1, migration outwards on timescale tau_mig/(R-1)
    disk_radius_trap : float
        Radius [r_{g,SMBH}] of disk migration trap
    disk_radius_outer : float
        Radius [r_{g,SMBH}] of outer edge of disk
    timestep_duration_yr : float
        Length of timestep [yr]

    Returns
    -------
    orbs_a : float array
        Semi-major axes [r_{g,SMBH}] of objects at end of timestep
    """

    orbs_a = blackholes_binary.bin_orb_a
    masses = blackholes_binary.mass_1 + blackholes_binary.mass_2
    orbs_ecc = blackholes_binary.bin_orb_ecc

    blackholes_binary.bin_orb_a = type1_migration(smbh_mass, orbs_a, masses, orbs_ecc, orb_ecc_crit,
                                                  disk_surf_density_func, disk_aspect_ratio_func, disk_feedback_ratio_func,
                                                  disk_radius_trap, disk_radius_outer, timestep_duration_yr)

    return (blackholes_binary)