"""This module provides methods for calculating the orbital and binary eccentricities."""

import numpy as np
from mcfacts.objects.agnobject import obj_to_binary_bh_array


def orbital_ecc_damping(smbh_mass, disk_bh_pro_orbs_a, disk_bh_pro_orbs_masses, disk_surf_density_func,
                        disk_aspect_ratio_func, disk_bh_pro_orbs_ecc, timestep_duration_yr, disk_bh_pro_orb_ecc_crit):
    """This method returns an array of BH orbital eccentricities damped according to a prescription.
    
    Using Tanaka & Ward (2004)  t_damp = M^3/2 h^4 / (2^1/2 m Sigma a^1/2 G ) 
    where M is the central mass, h is the disk aspect ratio (H/a), m is the orbiter mass, 
    Sigma is the disk surface density, a is the semi-major axis, G is the universal gravitational constant.
     
    From McKernan & Ford (2023) eqn 4. we can parameterize t_damp as 
    t_damp ~ 0.1Myr (q/10^-7)^-1 (h/0.03)^4 (Sigma/10^5 kg m^-2)^-1 (a/10^4r_g)^-1/2

    Notes
    -----
    For eccentricity e<2h 
    e(t)=e0*exp(-t/t_damp)......(1)
    
    So 
    in 0.1 damping time, e(t_damp)=0.90*e0
    in 1 damping time,  e(t_damp)=0.37*e0
    in 2 damping times, e(t_damp)=0.135*e0
    in 3 damping times, e(t_damp)=0.05*e0

    For now assume e<2h condition. To do: Add condition below (if ecc>2h..)

    For eccentricity e>2h eqn. 9 in McKernan & Ford (2023), based on Horn et al. (2012) the scaling time is now t_ecc.
    t_ecc = (t_damp/0.78)*[1 - (0.14*(e/h)^2) + (0.06*(e/h)^3)] ......(2)
    which in the limit of e>0.1 for most disk models becomes
    t_ecc ~ (t_damp/0.78)*[1 + (0.06*(e/h)^3)] 

    Parameters
    ----------
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    disk_bh_pro_orbs_a : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    disk_bh_pro_orbs_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_density_func : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_bh_pro_orbs_ecc : float array
        orbital eccentricity of singleton BH     
    timestep_duration_yr : float
        size of timestep in years
    disk_bh_pro_orb_ecc_crit: float
        critical eccentricity of prograde BH
        
    Returns
    -------
    bh_new_orb_ecc : float array
        updated orbital eccentricities damped by AGN gas
    """

    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(disk_bh_pro_orbs_a)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(disk_bh_pro_orbs_a)

    # Set up new_disk_bh_pro_orbs_ecc
    new_disk_bh_pro_orbs_ecc = np.empty_like(disk_bh_pro_orbs_ecc)

    # Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7
    mass_ratio = disk_bh_pro_orbs_masses / smbh_mass

    normalized_mass_ratio = mass_ratio / 10 ** (-7)
    normalized_bh_locations = disk_bh_pro_orbs_a / 1.e4
    normalized_disk_surf_density_func = disk_surface_density / 1.e5
    normalized_aspect_ratio = disk_aspect_ratio / 0.03

    # Assume all incoming eccentricities are prograde (for now)
    prograde_disk_bh_pro_orbs_ecc = disk_bh_pro_orbs_ecc

    # Calculate (e/h) ratio for all prograde BH for use in eqn. 2 above
    e_h_ratio = prograde_disk_bh_pro_orbs_ecc / disk_aspect_ratio

    # Modest orb eccentricities: e < 2h (experience simple exponential damping): mask entries > 2*aspect_ratio;
    # only show BH with e<2h
    prograde_bh_modest_ecc = np.ma.masked_where(
        prograde_disk_bh_pro_orbs_ecc > 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a),
        prograde_disk_bh_pro_orbs_ecc)

    # Large orb eccentricities: e > 2h (experience more complicated damping)
    prograde_bh_large_ecc = np.ma.masked_where(
        prograde_disk_bh_pro_orbs_ecc < 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a),
        prograde_disk_bh_pro_orbs_ecc)

    # Indices of orb eccentricities where e<2h
    modest_ecc_prograde_indices = np.ma.nonzero(prograde_bh_modest_ecc)

    # Indices of orb eccentricities where e>2h
    large_ecc_prograde_indices = np.ma.nonzero(prograde_bh_large_ecc)

    # print('modest ecc indices', modest_ecc_prograde_indices)
    # print('large ecc indices', large_ecc_prograde_indices)
    # Calculate the 1-d array of damping times at all locations since we need t_damp for both modest & large ecc
    # (see eqns above)
    t_damp = 1.e5 * (1.0 / normalized_mass_ratio) * (normalized_aspect_ratio ** 4) * (
            1.0 / normalized_disk_surf_density_func) * (1.0 / np.sqrt(normalized_bh_locations))

    # timescale ratio for modest ecc damping
    modest_timescale_ratio = timestep_duration_yr / t_damp

    # timescale for large ecc damping from eqn. 2 above
    t_ecc = (t_damp / 0.78) * (1 - (0.14 * (e_h_ratio) ** (2.0)) + (0.06 * (e_h_ratio) ** (3.0)))
    large_timescale_ratio = timestep_duration_yr / t_ecc
    # print("t_damp",timestep_duration_yr/t_damp)
    # print("t_ecc",timestep_duration_yr/t_ecc)

    # print("timescale_ratio",timescale_ratio)
    new_disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] * np.exp(
        -modest_timescale_ratio[modest_ecc_prograde_indices])
    new_disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] * np.exp(
        -large_timescale_ratio[large_ecc_prograde_indices])

    new_disk_bh_pro_orbs_ecc = np.where(new_disk_bh_pro_orbs_ecc < disk_bh_pro_orb_ecc_crit,
                                        disk_bh_pro_orb_ecc_crit, new_disk_bh_pro_orbs_ecc)

    # print("Old ecc, New ecc",disk_bh_pro_orbs_ecc,new_disk_bh_pro_orbs_ecc)
    return new_disk_bh_pro_orbs_ecc


def orbital_bin_ecc_damping(smbh_mass, bin_array, disk_surf_density_func, disk_aspect_ratio_func, timestep_duration_yr,
                            disk_bh_pro_orb_ecc_crit):
    """"Return bin_array orbital eccentricities damped according to a prescription.

    Use same mechanisms as for prograde singleton BH. 

    E.g. Tanaka & Ward (2004)  t_damp = M^3/2 h^4 / (2^1/2 m Sigma a^1/2 G )
    where M is the central mass, h is the disk aspect ratio (H/a), m is the orbiter mass,
    Sigma is the disk surface density, a is the semi-major axis, G is the universal gravitational constant.
    From McKernan & Ford (2023) eqn 4. we can parameterize t_damp as
    t_damp ~ 0.1Myr (q/10^-7)^-1 (h/0.03)^4 (Sigma/10^5 kg m^-2)^-1 (a/10^4r_g)^-1/2

    For eccentricity e<2h
        e(t)=e0*exp(-t/t_damp)......(1)
        So 
        in 0.1 damping time, e(t_damp)=0.90*e0
        in 1 damping time,  e(t_damp)=0.37*e0
        in 2 damping times, e(t_damp)=0.135*e0
        in 3 damping times, e(t_damp)=0.05*e0

    For now assume e<2h condition. To do: Add condition below (if ecc>2h..)

    For eccentricity e>2h eqn. 9 in McKernan & Ford (2023), based on Horn et al. (2012) the scaling time is now t_ecc.
        t_ecc = (t_damp/0.78)*[1 - (0.14*(e/h)^2) + (0.06*(e/h)^3)] ......(2)
        which in the limit of e>0.1 for most disk models becomes
        t_ecc ~ (t_damp/0.78)*[1 + (0.06*(e/h)^3)] 

        Parameters
    ----------
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    bin_array : float array
        binary array. Including bin_array[18,:]= bin orb ecc, bin_array[9,:] = bin center of mass etc.
    disk_surf_density_func : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated   
    timestep_duration_yr : float
        size of timestep in years
        
    Returns
    -------
    bin_array : float array
        updated binary orbital eccentricities damped by AGN gas
    """

    # First find num of bins by counting non zero M_1s.
    num_of_bins = np.count_nonzero(bin_array[2, :])
    bin_coms = np.zeros(num_of_bins)
    bin_orb_ecc = np.zeros(num_of_bins)
    bin_total_mass = np.zeros(num_of_bins)

    # Read in binary center of mass location, binary orb. ecc around SMBH and binary total mass:
    for i in range(num_of_bins):
        bin_coms[i] = bin_array[9, i]
        bin_orb_ecc[i] = bin_array[18, i]
        bin_total_mass[i] = bin_array[2, i] + bin_array[3, i]

    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(bin_coms)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(bin_coms)

    # Set up new_bin_orb_ecc
    new_bin_orb_ecc = np.empty_like(bin_orb_ecc)

    # Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7
    mass_ratio = bin_total_mass / smbh_mass

    normalized_mass_ratio = mass_ratio / 10 ** (-7)
    normalized_bh_locations = bin_coms / 1.e4
    normalized_disk_surf_density_func = disk_surface_density / 1.e5
    normalized_aspect_ratio = disk_aspect_ratio / 0.03

    # Calculate the damping time for all bins
    t_damp = 1.e5 * (1.0 / normalized_mass_ratio) * (normalized_aspect_ratio ** 4) * (
            1.0 / normalized_disk_surf_density_func) * (1.0 / np.sqrt(normalized_bh_locations))

    # Calculate (e/h) ratio for all prograde BH for use in eqn. 2 above
    e_h_ratio = bin_orb_ecc / disk_aspect_ratio

    # Calculate damping time for large orbital eccentricity binaries
    t_ecc = (t_damp / 0.78) * (1 - (0.14 * (e_h_ratio) ** (2.0)) + (0.06 * (e_h_ratio) ** (3.0)))

    for i in range(num_of_bins):
        # If bin orb ecc <= disk_bh_pro_orb_ecc_crit, do nothing (no damping needed)
        if bin_orb_ecc[i] <= disk_bh_pro_orb_ecc_crit:
            new_bin_orb_ecc[i] = bin_orb_ecc[i]
        # If bin orb ecc > disk_bh_pro_orb_ecc_crit, but <2*h then damp modest orb eccentricity
        if bin_orb_ecc[i] > disk_bh_pro_orb_ecc_crit and bin_orb_ecc[i] < 2.0 * disk_aspect_ratio[i]:
            modest_timescale_ratio = timestep_duration_yr / t_damp[i]
            new_bin_orb_ecc[i] = bin_orb_ecc[i] * np.exp(-modest_timescale_ratio)
            # print("NEW modest bin orb ecc old, new ",bin_orb_ecc[i], new_bin_orb_ecc[i])
        # If bin orb ecc > 2*h then damp large orb eccentricity
        if bin_orb_ecc[i] > disk_bh_pro_orb_ecc_crit and bin_orb_ecc[i] > 2.0 * disk_aspect_ratio[i]:
            large_timescale_ratio = timestep_duration_yr / t_ecc[i]
            new_bin_orb_ecc[i] = bin_orb_ecc[i] * np.exp(-large_timescale_ratio)
            # print("NEW large bin orb ecc old,new", bin_orb_ecc[i],new_bin_orb_ecc[i])

    # Write new values of bin orbital eccentricity to bin_array
    for j in range(num_of_bins):
        if new_bin_orb_ecc[j] < disk_bh_pro_orb_ecc_crit:
            new_bin_orb_ecc[j] = disk_bh_pro_orb_ecc_crit
        bin_array[18, j] = new_bin_orb_ecc[j]

    return bin_array


def orbital_bin_ecc_damping_obj(smbh_mass, blackholes_binary, disk_surf_density_func, disk_aspect_ratio_func, timestep_duration_yr,
                                disk_bh_pro_orb_ecc_crit):

    bin_array = obj_to_binary_bh_array(blackholes_binary)

    bin_array = orbital_bin_ecc_damping(smbh_mass, bin_array, disk_surf_density_func, disk_aspect_ratio_func, timestep_duration_yr,
                                        disk_bh_pro_orb_ecc_crit)

    blackholes_binary.bin_orb_ecc = bin_array[18, :]

    return (blackholes_binary)


def bin_ecc_damping(smbh_mass, disk_bh_pro_orbs_a, disk_bh_pro_orbs_masses, disk_surf_density_func,
                    disk_aspect_ratio_func,
                    disk_bh_pro_orbs_ecc, timestep_duration_yr, disk_bh_pro_orb_ecc_crit):
    """"Return binary array with modified eccentricities according to Calcino et al. (2023), arXiv:2312.13727

    Notes
    -----
    dot{e}_b increases in magnitude as e_b is larger.
    dot{e}_b is always negative for prograde bins (ie e_b is damped for prograde bins)
    dot{e}_b is always positive for retrograde bins (ie e_b is pumped for retrograde bins)

    For retrograde bins:
        dot{e_b} ~ 1 in units of dot{M_bondi}/M_bondi for e_b~ 0.5, dot_{e_b}>1 for e_b >0.5, <1 for e_b<0.5
    For prograde bins:
        dot{e_b} ~ -0.25 in the same units for e_b~0.5, dot{e_b} ~ -0.28 at e_b=0.7, dot{e_b}~ -0.04 at e_b=0.1
    and
        dot{m_bin} is ~0.05 M_bondi in these sims.
    with (from their eqn. 19)
        M_bondi/M_edd ~ 5e5 (R_H/H)^3  (rho/10^-14 g/cc) (M_smbh/10^6M_sun)^-1/2 (R0/0.1pc)^3/2 (e/0.1)
        where R_H=Hill sphere radius, H = disk scale height, rho = disk midplane density, 
        R0=location of binary, e=acc. efficiency onto SMBH (L=e*dot{M}c^2)
    Convert to 10^8Msun, *1/10    

    Use Tanaka & Ward (2004)  t_damp = M^3/2 h^4 / (2^1/2 m Sigma a^1/2 G )
    where M is the central mass, h is the disk aspect ratio (H/a), m is the orbiter mass,
    Sigma is the disk surface density, a is the semi-major axis, G is the universal gravitational constant.

    From McKernan & Ford (2023) eqn 4. we can parameterize t_damp as
        t_damp ~ 0.1Myr (q/10^-7)^-1 (h/0.03)^4 (Sigma/10^5 kg m^-2)^-1 (a/10^4r_g)^-1/2

    For eccentricity e<2h
        e(t)=e0*exp(-t/t_damp)......(1)
        So 
            in 0.1 damping time, e(t_damp)=0.90*e0
            in 1 damping time,  e(t_damp)=0.37*e0
            in 2 damping times, e(t_damp)=0.135*e0
            in 3 damping times, e(t_damp)=0.05*e0

    For now assume e<2h condition. To do: Add condition below (if ecc>2h..)

    For eccentricity e>2h eqn. 9 in McKernan & Ford (2023), based on Horn et al. (2012) the scaling time is now t_ecc.
        t_ecc = (t_damp/0.78)*[1 - (0.14*(e/h)^2) + (0.06*(e/h)^3)] ......(2)
        which in the limit of e>0.1 for most disk models becomes
        t_ecc ~ (t_damp/0.78)*[1 + (0.06*(e/h)^3)] 

    Parameters
    ----------
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    disk_bh_pro_orbs_a : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    disk_bh_pro_orbs_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_density_func : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_func : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_bh_pro_orbs_ecc : float array
        orbital eccentricity of singleton BH     
    timestep_duration_yr : float
        size of timestep in years
        
    Returns
    -------
    bh_new_orb_ecc : float array
        updated orbital eccentricities damped by AGN gas
    """
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_density_func, float):
        disk_surface_density = disk_surf_density_func
    else:
        disk_surface_density = disk_surf_density_func(disk_bh_pro_orbs_a)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_func, float):
        disk_aspect_ratio = disk_aspect_ratio_func
    else:
        disk_aspect_ratio = disk_aspect_ratio_func(disk_bh_pro_orbs_a)
    # Set up new_disk_bh_pro_orbs_ecc
    new_disk_bh_pro_orbs_ecc = np.empty_like(disk_bh_pro_orbs_ecc)

    # Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7 
    mass_ratio = disk_bh_pro_orbs_masses / smbh_mass

    normalized_mass_ratio = mass_ratio / 10 ** (-7)
    normalized_bh_locations = disk_bh_pro_orbs_a / 1.e4
    normalized_disk_surf_density_func = disk_surface_density / 1.e5
    normalized_aspect_ratio = disk_aspect_ratio / 0.03

    # Assume all incoming eccentricities are prograde (for now)
    prograde_disk_bh_pro_orbs_ecc = disk_bh_pro_orbs_ecc

    # Calculate (e/h) ratio for all prograde BH for use in eqn. 2 above
    e_h_ratio = prograde_disk_bh_pro_orbs_ecc / disk_aspect_ratio

    # Modest orb eccentricities: e < 2h (experience simple exponential damping): mask entries > 2*aspect_ratio;
    # only show BH with e<2h
    prograde_bh_modest_ecc = np.ma.masked_where(
        prograde_disk_bh_pro_orbs_ecc > 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a),
        prograde_disk_bh_pro_orbs_ecc)

    # Large orb eccentricities: e > 2h (experience more complicated damping)
    prograde_bh_large_ecc = np.ma.masked_where(
        prograde_disk_bh_pro_orbs_ecc < 2.0 * disk_aspect_ratio_func(disk_bh_pro_orbs_a),
        prograde_disk_bh_pro_orbs_ecc)

    # Indices of orb eccentricities where e<2h
    modest_ecc_prograde_indices = np.ma.nonzero(prograde_bh_modest_ecc)

    # Indices of orb eccentricities where e>2h
    large_ecc_prograde_indices = np.ma.nonzero(prograde_bh_large_ecc)

    # print('modest ecc indices', modest_ecc_prograde_indices)
    # print('large ecc indices', large_ecc_prograde_indices)
    # Calculate the 1-d array of damping times at all locations since we need t_damp for both modest & large ecc
    # (see eqns above)
    t_damp = 1.e5 * (1.0 / normalized_mass_ratio) * (normalized_aspect_ratio ** 4) * (
            1.0 / normalized_disk_surf_density_func) * (1.0 / np.sqrt(normalized_bh_locations))

    # timescale ratio for modest ecc damping
    modest_timescale_ratio = timestep_duration_yr / t_damp

    # timescale for large ecc damping from eqn. 2 above
    t_ecc = (t_damp / 0.78) * (1 - (0.14 * (e_h_ratio) ** (2.0)) + (0.06 * (e_h_ratio) ** (3.0)))
    large_timescale_ratio = timestep_duration_yr / t_ecc
    # print("t_damp",timestep_duration_yr/t_damp)
    # print("t_ecc",timestep_duration_yr/t_ecc)

    # print("timescale_ratio",timescale_ratio)
    new_disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[modest_ecc_prograde_indices] * np.exp(
        -modest_timescale_ratio[modest_ecc_prograde_indices])
    new_disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] = disk_bh_pro_orbs_ecc[large_ecc_prograde_indices] * np.exp(
        -large_timescale_ratio[large_ecc_prograde_indices])
    new_disk_bh_pro_orbs_ecc = np.where(new_disk_bh_pro_orbs_ecc < disk_bh_pro_orb_ecc_crit, disk_bh_pro_orb_ecc_crit,
                                        new_disk_bh_pro_orbs_ecc)

    # print("Old ecc, New ecc",disk_bh_pro_orbs_ecc,new_disk_bh_pro_orbs_ecc)
    return new_disk_bh_pro_orbs_ecc
