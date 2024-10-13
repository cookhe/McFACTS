"""
Module for evolving the state of a binary.
"""
import numpy as np
import scipy
from mcfacts.objects.agnobject import obj_to_binary_bh_array


def change_bin_mass(blackholes_binary, disk_bh_eddington_ratio,
                    disk_bh_eddington_mass_growth_rate, timestep_duration_yr):
    """
    Given initial binary black hole masses at timestep start, add mass according to
    chosen BH mass accretion prescription

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        binary black holes in prograde orbits around SMBH
    disk_bh_eddington_ratio : float
        user chosen input set by input file; accretion rate of fully embedded stellar
        mass black hole in units of Eddington accretion rate. 1.0=embedded BH accreting
        at Eddington. Super-Eddington accretion rates are permitted.
    disk_bh_eddington_mass_growth : float
        fractional rate of mass growth AT Eddington accretion rate per year (2.3e-8)
    timestep_duration_yr : float
        length of timestep in units of years

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        updated binary black holes after accreting mass at prescribed rate for one timestep
    """

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)

    # If all BH have merged then nothing to do
    if (idx_non_mergers[0].shape[0] == 0):
        return (blackholes_binary)

    mass_growth_factor = np.exp(disk_bh_eddington_mass_growth_rate * disk_bh_eddington_ratio * timestep_duration_yr)

    mass_1_before = blackholes_binary.mass_1[idx_non_mergers]
    mass_2_before = blackholes_binary.mass_2[idx_non_mergers]

    blackholes_binary.mass_1[idx_non_mergers] = mass_1_before * mass_growth_factor
    blackholes_binary.mass_2[idx_non_mergers] = mass_2_before * mass_growth_factor

    return (blackholes_binary)


def change_bin_spin_magnitudes(blackholes_binary, disk_bh_eddington_ratio,
                               disk_bh_torque_condition, timestep_duration_yr):
    """
    Given initial binary black hole spins at start of timestep_duration_yr, add spin according to
        chosen BH torque prescription. If spin is greater than max allowed spin, spin is set to
        max value.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        binary black holes in prograde orbits around SMBH
    disk_bh_eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass 
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.    
    disk_bh_torque_condition : float
        fraction of initial mass required to be accreted before BH spin is torqued
        fully into alignment with the AGN disk. We don't know for sure but
        Bogdanovic et al. (2007) says between 0.01=1% and 0.1=10% is what is required.
    timestep_duration_yr : float
        length of timestep in units of years. Default is 10^4yr

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Updated blackholes_binary after spin up of BH at prescribed rate for one timestep_duration_yr
    """

    disk_bh_eddington_ratio_normalized = disk_bh_eddington_ratio/1.0  # does nothing?
    timestep_duration_yr_normalized = timestep_duration_yr/1.e4  # yrs to yr/10k?
    disk_bh_torque_condition_normalized = disk_bh_torque_condition/0.1  # what does this do?

    # Set max allowed spin
    max_allowed_spin = 0.98

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)

    # If all BH have merged then nothing to do
    if (idx_non_mergers[0].shape[0] == 0):
        return (blackholes_binary)

    spin_change_factor = 4.4e-3 * disk_bh_eddington_ratio_normalized * disk_bh_torque_condition_normalized * timestep_duration_yr_normalized

    spin_1_before = blackholes_binary.spin_1[idx_non_mergers]
    spin_2_before = blackholes_binary.spin_2[idx_non_mergers]

    spin_1_after = spin_1_before + spin_change_factor
    spin_2_after = spin_2_before + spin_change_factor

    spin_1_after[spin_1_after > max_allowed_spin] = np.full(np.sum(spin_1_after > max_allowed_spin), max_allowed_spin)
    spin_2_after[spin_2_after > max_allowed_spin] = np.full(np.sum(spin_2_after > max_allowed_spin), max_allowed_spin)

    blackholes_binary.spin_1[idx_non_mergers] = spin_1_after
    blackholes_binary.spin_2[idx_non_mergers] = spin_2_after

    return (blackholes_binary)


def change_bin_spin_angles(blackholes_binary, disk_bh_eddington_ratio,
                           disk_bh_torque_condition, spin_minimum_resolution,
                           timestep_duration_yr):
    """
    Given initial binary black hole spin angles at start of timestep, subtract spin angle
    according to chosen BH torque prescription. If spin angle is less than spin minimum
    resolution, spin angle is set to 0.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        binary black holes in prograde orbits around the SMBH
    disk_bh_eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass 
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.    
    disk_bh_torque_condition : float
        fraction of initial mass required to be accreted before BH spin is torqued
        fully into alignment with the AGN disk. We don't know for sure but
        Bogdanovic et al. (2007) says between 0.01=1% and 0.1=10% is what is required.
    timestep_duration_yr : float
        length of timestep in units of years. Default is 10^4yr

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        Updated blackholes_binary after spin up of BH at prescribed rate for one timestep_duration_yr
    """
    disk_bh_eddington_ratio_normalized = disk_bh_eddington_ratio/1.0  # does nothing?
    timestep_duration_yr_normalized = timestep_duration_yr/1.e4  # yrs to yr/10k?
    disk_bh_torque_condition_normalized = disk_bh_torque_condition/0.1  # what does this do?

    # Only interested in BH that have not merged
    idx_non_mergers = np.where(blackholes_binary.flag_merging >= 0)

    # If all BH have merged then nothing to do
    if (idx_non_mergers[0].shape[0] == 0):
        return (blackholes_binary)

    spin_angle_change_factor = 6.98e-3 * disk_bh_eddington_ratio_normalized * disk_bh_torque_condition_normalized * timestep_duration_yr_normalized

    spin_angle_1_before = blackholes_binary.spin_angle_1[idx_non_mergers]
    spin_angle_2_before = blackholes_binary.spin_angle_2[idx_non_mergers]

    spin_angle_1_after = spin_angle_1_before - spin_angle_change_factor
    spin_angle_2_after = spin_angle_2_before - spin_angle_change_factor

    spin_angle_1_after[spin_angle_1_after < spin_minimum_resolution] = np.zeros(np.sum(spin_angle_1_after < spin_minimum_resolution))
    spin_angle_2_after[spin_angle_2_after < spin_minimum_resolution] = np.zeros(np.sum(spin_angle_2_after < spin_minimum_resolution))

    blackholes_binary.spin_angle_1[idx_non_mergers] = spin_angle_1_after
    blackholes_binary.spin_angle_2[idx_non_mergers] = spin_angle_2_after

    return (blackholes_binary)


def bin_com_feedback_hankla(blackholes_binary, disk_surface_density,
                            disk_opacity_func, disk_bh_eddington_ratio,
                            disk_alpha_viscosity, disk_radius_outer,
                            thermal_feedback_max):
    """
    This feedback model uses Eqn. 28 in Hankla, Jiang & Armitage (2020)
    which yields the ratio of heating torque to migration torque.
    Heating torque is directed outwards. 
    So, Ratio <1, slows the inward migration of an object. Ratio>1 sends the object migrating outwards.
    The direction & magnitude of migration (effected by feedback) will be executed in type1.py.

    The ratio of torque due to heating to Type 1 migration torque is calculated as
    R   = Gamma_heat/Gamma_mig 
        ~ 0.07 (speed of light/ Keplerian vel.)(Eddington ratio)(1/optical depth)(1/alpha)^3/2
    where Eddington ratio can be >=1 or <1 as needed,
    optical depth (tau) = Sigma* kappa
    alpha = disk_alpha_viscosity (e.g. alpha = 0.01 in Sirko & Goodman 2003)
    kappa = 10^0.76 cm^2 g^-1=5.75 cm^2/g = 0.575 m^2/kg for most of Sirko & Goodman disk model (see Fig. 1 & sec 2)
    but e.g. electron scattering opacity is 0.4 cm^2/g
    So tau = Sigma*0.575 where Sigma is in kg/m^2.
    Since v_kep = c/sqrt(a(r_g)) then
    R   ~ 0.07 (a(r_g))^{1/2}(Edd_ratio) (1/tau) (1/alpha)^3/2
    So if assume a=10^3r_g, Sigma=7.e6kg/m^2, alpha=0.01, tau=0.575*Sigma (SG03 disk model), Edd_ratio=1, 
    R   ~5.5e-4 (a/10^3r_g)^(1/2) (Sigma/7.e6) v.small modification to in-migration at a=10^3r_g
        ~0.243 (R/10^4r_g)^(1/2) (Sigma/5.e5)  comparable.
        >1 (a/2x10^4r_g)^(1/2)(Sigma/) migration is *outward* at >=20,000r_g in SG03
        >10 (a/7x10^4r_g)^(1/2)(Sigma/) migration outwards starts to runaway in SG03

    Parameters
    ----------

    blackholes_binary : AGNBinaryBlackHole
        binary black holes
    disk_surface_density : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g (r_g=GM_SMBH/c^2)
        can accept a simple float (constant), but this is deprecated
    disk_opacity_func : lambda
        Opacity as a function of radius
    disk_bh_eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass 
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    disk_alpha_viscosity : float
        disk viscosity parameter
    disk_radius_outer : float
        final element of disk_model_radius_array (units of r_g)
    thermal_feedback_max : float
        Maximum allowed value for the ratio of radiative feedback torque
        to Type 1 migration torque.

    Returns
    -------
    ratio_feedback_to_mig : float array
        ratio of feedback torque to migration torque for each entry in prograde_bh_locations
    """

    # Making sure that surface density is a float or a function (from old function)
    if not isinstance(disk_surface_density, float):
        disk_surface_density_at_location = disk_surface_density(blackholes_binary.bin_orb_a)
    else:
        raise AttributeError("disk_surface_density is a float")

    # Define kappa (or set up a function to call). 
    disk_opacity = disk_opacity_func(blackholes_binary.bin_orb_a)

    ratio_heat_mig_torques_bin_com = 0.07 * (1 / disk_opacity) * np.power(disk_alpha_viscosity, -1.5) * disk_bh_eddington_ratio * np.sqrt(blackholes_binary.bin_orb_a) / disk_surface_density_at_location

    # set ratio = 1 (no migration) for binaries at or beyond the disk outer radius
    ratio_heat_mig_torques_bin_com[blackholes_binary.bin_orb_a > disk_radius_outer] = 1.

    # apply the cap to the feedback ratio
    ratio_heat_mig_torques_bin_com[np.where(ratio_heat_mig_torques_bin_com > thermal_feedback_max)] = thermal_feedback_max

    return (ratio_heat_mig_torques_bin_com)


def bin_migration(smbh_mass, disk_bin_bhbh_pro_array, disk_surf_model, disk_aspect_ratio_model, timestep_duration_yr, feedback_ratio, disk_radius_trap, disk_bh_pro_orb_ecc_crit, disk_radius_outer):
    """
    This function calculates how far the center of mass of a binary migrates in an AGN gas disk in a time
    of length timestep_duration_yr, assuming a gas disk surface density and aspect ratio profile, for
    objects of specified masses and starting locations, and returns their new locations
    after migration over one timestep_duration_yr. Uses standard Type I migration prescription,
    modified by Hankla+22 feedback model if included.
    This is an exact copy of mcfacts.physics.migration.type1.type1

    Parameters
    ----------
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    disk_bin_bhbh_pro_array : float array
        Full binary array.
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    timestep_duration_yr : float
        size of timestep_duration_yr in years
    feedback_ratio : float
        effect of feedback on Type I migration torque if feedback switch on
    disk_radius_trap : float
        location of migration trap in units of r_g. From Bellovary+16, should be 700r_g for Sirko & Goodman '03, 245r_g for Thompson et al. '05
    disk_bh_pro_orb_ecc_crit : float
        User defined critical orbital eccentricity for pro BH, below which BH are considered circularized

    Returns
    -------
    disk_bin_bhbh_pro_array : float array
        Returns modified disk_bin_bhbh_pro_array with updated center of masses of the binary bhbh.
    """

    # locations of center of mass of bhbh binaries
    bin_com = disk_bin_bhbh_pro_array[9,:]
    # masses of each bhbh binary
    bin_mass = disk_bin_bhbh_pro_array[2,:] + disk_bin_bhbh_pro_array[3,:]
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_model, float):
        disk_surface_density = disk_surf_model
    else:
        disk_surface_density = disk_surf_model(bin_com)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_model, float):
        disk_aspect_ratio = disk_aspect_ratio_model
    else:
        disk_aspect_ratio = disk_aspect_ratio_model(bin_com)

    # This is an exact copy of mcfacts.physics.migration.type1.type1.
    tau_mig = ((disk_aspect_ratio**2)* scipy.constants.c/(3.0*scipy.constants.G) * (smbh_mass/bin_mass) / disk_surface_density) / np.sqrt(bin_com)
    # ratio of timestep_duration_yr to tau_mig (timestep_duration_yr in years so convert)
    dt = timestep_duration_yr * scipy.constants.year / tau_mig
    # migration distance is original locations times fraction of tau_mig elapsed
    migration_distance = bin_com * dt

    disk_bin_bhbh_pro_orbs_a = np.zeros_like(bin_com)

    # Find indices of objects where feedback ratio <1; these still migrate inwards, but more slowly
    index_inwards_modified = np.where(feedback_ratio < 1)[0]
    index_inwards_size = index_inwards_modified.size
    all_inwards_migrators = bin_com[index_inwards_modified]

    #Given a population migrating inwards
    if index_inwards_size > 0:
        for i in range(0,index_inwards_size):
                # Among all inwards migrators, find location in disk & compare to trap radius
                critical_distance = all_inwards_migrators[i]
                actual_index = index_inwards_modified[i]
                #If outside trap, migrates inwards
                if critical_distance > disk_radius_trap:
                    disk_bin_bhbh_pro_orbs_a[actual_index] = bin_com[actual_index] - (migration_distance[actual_index]*(1-feedback_ratio[actual_index]))
                    #If inward migration takes object inside trap, fix at trap.
                    if disk_bin_bhbh_pro_orbs_a[actual_index] <= disk_radius_trap:
                        disk_bin_bhbh_pro_orbs_a[actual_index] = disk_radius_trap
                #If inside trap, migrates out
                if critical_distance < disk_radius_trap:
                    disk_bin_bhbh_pro_orbs_a[actual_index] = bin_com[actual_index] + (migration_distance[actual_index]*(1-feedback_ratio[actual_index]))
                    if disk_bin_bhbh_pro_orbs_a[actual_index] >= disk_radius_trap:
                        disk_bin_bhbh_pro_orbs_a[actual_index] = disk_radius_trap
                #If at trap, stays there
                if critical_distance == disk_radius_trap:
                    disk_bin_bhbh_pro_orbs_a[actual_index] = bin_com[actual_index]

    # Find indices of objects where feedback ratio >1; these migrate outwards.
    index_outwards_modified = np.where(feedback_ratio >1)[0]

    if index_outwards_modified.size > 0:
        disk_bin_bhbh_pro_orbs_a[index_outwards_modified] = bin_com[index_outwards_modified] +(migration_distance[index_outwards_modified]*(feedback_ratio[index_outwards_modified]-1))
        # catch to keep stuff from leaving the outer radius of the disk!
        disk_bin_bhbh_pro_orbs_a[index_outwards_modified[np.where(disk_bin_bhbh_pro_orbs_a[index_outwards_modified] > disk_radius_outer)]] = disk_radius_outer
    
    #Find indices where feedback ratio is identically 1; shouldn't happen (edge case) if feedback on, but == 1 if feedback off.
    index_unchanged = np.where(feedback_ratio == 1)[0]
    if index_unchanged.size > 0:
    # If BH location > trap radius, migrate inwards
        for i in range(0,index_unchanged.size):
            locn_index = index_unchanged[i]
            if bin_com[locn_index] > disk_radius_trap:
                disk_bin_bhbh_pro_orbs_a[locn_index] = bin_com[locn_index] - migration_distance[locn_index]
            # if new location is <= trap radius, set location to trap radius
                if disk_bin_bhbh_pro_orbs_a[locn_index] <= disk_radius_trap:
                    disk_bin_bhbh_pro_orbs_a[locn_index] = disk_radius_trap

        # If BH location < trap radius, migrate outwards
            if bin_com[locn_index] < disk_radius_trap:
                disk_bin_bhbh_pro_orbs_a[locn_index] = bin_com[locn_index] + migration_distance[locn_index]
                #if new location is >= trap radius, set location to trap radius
                if disk_bin_bhbh_pro_orbs_a[locn_index] >= disk_radius_trap:
                    disk_bin_bhbh_pro_orbs_a[locn_index] = disk_radius_trap

    # Finite check
    assert np.isfinite(disk_bin_bhbh_pro_orbs_a).all(),\
        "Finite check failed for disk_bin_bhbh_pro_orbs_a"
    # Zero check
    assert (disk_bin_bhbh_pro_orbs_a != 0.).all(),\
        "Some disk_bin_bhbh_pro_orbs_a are zero"
    # Distance travelled per binary is old location of com minus new location of com. Is +ive(-ive) if migrating in(out)
    dist_travelled = disk_bin_bhbh_pro_array[9,:] - disk_bin_bhbh_pro_orbs_a

    num_of_bins = np.count_nonzero(disk_bin_bhbh_pro_array[2,:])

    for i in range(num_of_bins):
        # If circularized then migrate
        if disk_bin_bhbh_pro_array[18,i] <= disk_bh_pro_orb_ecc_crit:
            disk_bin_bhbh_pro_array[9,i] = disk_bin_bhbh_pro_orbs_a[i]
        # If not circularized, no migration
        if disk_bin_bhbh_pro_array[18,i] > disk_bh_pro_orb_ecc_crit:
            pass

    # Finite check
    assert np.isfinite(disk_bin_bhbh_pro_array[18,:]).all(),\
        "Fintie check failure: disk_bin_bhbh_pro_array"
    # Assert that things are not allowed to migrate out of the disk.
    mask_disk_radius_outer = disk_radius_outer < disk_bin_bhbh_pro_array
    disk_bin_bhbh_pro_array[mask_disk_radius_outer] = disk_radius_outer
    return disk_bin_bhbh_pro_array


def bin_migration_obj(smbh_mass, blackholes_binary, disk_surf_model, disk_aspect_ratio_model,
                      timestep_duration_yr, feedback_ratio, disk_radius_trap,
                      disk_bh_pro_orb_ecc_crit, disk_radius_outer):

    disk_bin_bhbh_pro_array = obj_to_binary_bh_array(blackholes_binary)

    disk_bin_bhbh_pro_array = bin_migration(smbh_mass, disk_bin_bhbh_pro_array, disk_surf_model,
                                            disk_aspect_ratio_model, timestep_duration_yr,
                                            feedback_ratio, disk_radius_trap, disk_bh_pro_orb_ecc_crit,
                                            disk_radius_outer)

    blackholes_binary.bin_orb_a = disk_bin_bhbh_pro_array[9, :]

    return (blackholes_binary)


def bin_ionization_check(blackholes_binary, smbh_mass):
    """
    This function tests whether a binary has been softened beyond some limit.
    Returns ID numbers of binaries to be ionized.
    The limit is set to some fraction of the binary Hill sphere, frac_R_hill

    Default is frac_R_hill = 1.0 (ie binary is ionized at the Hill sphere). 
    Change frac_R_hill if you're testing binary formation at >R_hill.

    R_hill = a_com*(M_bin/3M_smbh)^1/3

    where a_com is the radial disk location of the binary center of mass,
    M_bin = M_1 + M_2 is the binary mass
    M_smbh is the SMBH mass (given by smbh_mass) 

    Condition is:
    if bin_separation > frac_R_hill*R_hill:
        Ionize binary.
        Remove binary from blackholes_binary!
        Add two new singletons to the singleton arrays.

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole 
        Full binary array.
    smbh_mass : float
        mass of supermassive black hole in units of solar masses

    Returns
    -------
    bh_id_nums : numpy array
        ID numbers of binaries to be removed from binary array
    """

    # Remove returning -1 if that's not how it's supposed to work
    # Define ionization threshold as a fraction of Hill sphere radius
    # Default is 1.0, change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    # bin_orb_a is in units of r_g of the SMBH = GM_smbh/c^2
    mass_ratio = blackholes_binary.mass_total/smbh_mass
    hill_sphere = blackholes_binary.bin_orb_a * np.power(mass_ratio / 3, 1. / 3.)

    bh_id_nums = blackholes_binary.id_num[np.where(blackholes_binary.bin_sep > (frac_rhill*hill_sphere))[0]]

    return (bh_id_nums)


def bin_contact_check(blackholes_binary, smbh_mass):
    """
    This function tests to see if the binary separation has shrunk so that the binary is touching!

    Touching condition is where binary separation is <= R_schw(M_1) + R_schw(M_2)
                                                      = 2(R_g(M_1) + R_g(M_2))
                                                      = 2G(M_1+M_2) / c^{2}

    Since binary separation is in units of r_g (GM_smbh/c^2) then condition is simply:
        binary_separation <= 2M_bin/M_smbh
    
    Parameters
    ---------- 
    blackholes_binary : float array 
        Full binary array.
    smbh_mass : float
        mass of supermassive black hole in units of solar masses

    Returns
    -------
    blackholes_binary : float array 
        Returns modified blackholes_binary with updated bin_sep and flag_merging.
    """

    mass_binary = blackholes_binary.mass_1 + blackholes_binary.mass_2

    # We assume bh are not spinning when in contact. TODO: Consider spin in future.
    contact_condition = 2 * (mass_binary / smbh_mass)
    mask_condition = (blackholes_binary.bin_sep <= contact_condition)

    # If binary separation <= contact condition, set binary separation to contact condition
    blackholes_binary.bin_sep[mask_condition] = contact_condition[mask_condition]
    blackholes_binary.flag_merging[mask_condition] = np.full(np.sum(mask_condition), -2)

    return (blackholes_binary)


def bin_reality_check(blackholes_binary):
    """ This function tests to see if the binary is real. If location = 0 or mass = 0 *and* any other element is NON-ZERO then discard this binary element.
        Returns ID numbers of fake binaries.

        Parameters
        ----------
        blackholes_binary : AGNBinaryBlackHole 
            binary black holes.

        Returns
        -------
        disk_bin_bhbh_pro_array : float array 
            Returns modified disk_bin_bhbh_pro_array with updated GW properties (strain,freq) bhbh.
        """
    bh_bin_id_num_fakes = np.array([])

    mass_1_id_num = blackholes_binary.id_num[blackholes_binary.mass_1 == 0]
    mass_2_id_num = blackholes_binary.id_num[blackholes_binary.mass_2 == 0]
    orb_a_1_id_num = blackholes_binary.id_num[blackholes_binary.orb_a_1 == 0]
    orb_a_2_id_num = blackholes_binary.id_num[blackholes_binary.orb_a_2 == 0]

    id_nums = np.concatenate([mass_1_id_num, mass_2_id_num,
                             orb_a_1_id_num, orb_a_2_id_num])

    if id_nums.size > 0:
        return (id_nums)
    else:
        return (bh_bin_id_num_fakes)


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
