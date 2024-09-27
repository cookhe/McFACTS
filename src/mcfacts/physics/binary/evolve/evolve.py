import numpy as np
import scipy
from mcfacts.objects.agnobject import obj_to_binary_bh_array
from astropy import constants as const
from astropy import units as u
from astropy.units import cds


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


def com_feedback_hankla(blackholes_binary, disk_surface_density, disk_opacity_func, disk_bh_eddington_ratio, disk_alpha_viscosity, disk_radius_outer):
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

    TO (MAYBE) DO: kappa default as an input? Or kappa table? Or kappa user set?

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

    ratio_heat_mig_torques_bin_com[blackholes_binary.bin_orb_a > disk_radius_outer] = np.ones(np.sum(blackholes_binary.bin_orb_a > disk_radius_outer))

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

    disk_bin_bhbh_pro_orbs_a = np.empty_like(bin_com)

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
        disk_bin_bhbh_pro_orbs_a[np.where(disk_bin_bhbh_pro_orbs_a[index_outwards_modified] > disk_radius_outer)] = disk_radius_outer
    
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


def gw_strain_freq(mass_1, mass_2, obj_sep, timestep_duration_yr, old_gw_freq, smbh_mass, agn_redshift, flag_include_old_gw_freq=1):
    """
    This function takes in two masses, their separation, the previous frequency, and the redshift and
    calculates the new GW strain (unitless) and frequency (Hz).

    Parameters
    ----------
    mass_1 : numpy array
        mass of object 1
    mass_2 : numpy array
        mass of object 2
    obj_sep : numpy array
        separation between both objects, in R_g of the SMBH = GM_smbh/c^2
    timestep_duration_yr : float, or -1 if not given
        current timestep in years
    old_gw_freq : numpy array, or -1 if not given
        previous GW frequency
    smbh_mass : float
        mass of the SMBH in Msun
    agn_redshift : float
        redshift of the SMBH
    flag_include_old_gw_freq : boolean
        flag indicating if old_gw_freq should be included in calculations
        0 if no, 1 if yes
    """

    redshift_d_obs_dict = {0.1: 421*u.Mpc,
                           0.5: 1909*u.Mpc}

    timestep_units = (timestep_duration_yr*u.year).to(u.second)

    # 1rg =1AU=1.5e11m for 1e8Msun
    rg = 1.5e11*(smbh_mass/1.e8)*u.meter
    mass_1 = (mass_1 * cds.Msun).to(u.kg)
    mass_2 = (mass_2 * cds.Msun).to(u.kg)
    mass_total = mass_1 + mass_2
    bin_sep = obj_sep * rg

    mass_chirp = np.power(mass_1 * mass_2, 3./5.) / np.power(mass_total, 1./5.)
    rg_chirp = ((const.G * mass_chirp) / np.power(const.c, 2)).to(u.meter)

    # If separation is less than rg_chirp then cap separation at rg_chirp.
    bin_sep[bin_sep < rg_chirp] = rg_chirp[bin_sep < rg_chirp]

    nu_gw = (1.0/np.pi)*np.sqrt(mass_total * const.G /
                               (bin_sep**(3.0)))
    nu_gw = nu_gw.to(u.Hz)

    # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
    # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
    # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
    d_obs = redshift_d_obs_dict[agn_redshift].to(u.meter)
    strain = (4/d_obs) * rg_chirp * np.power(np.pi * nu_gw * rg_chirp / const.c, 2./3.)
    
    # But power builds up in band over multiple cycles! 
    # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
    strain_factor = np.ones(len(nu_gw))

    # char amplitude = strain_factor*h0
    #                = sqrt(N/8)*h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
    strain_factor[nu_gw < (1e-6)*u.Hz] = np.sqrt(nu_gw[nu_gw < (1e-6)*u.Hz]*np.pi*(1e7)/8)

    # For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt).
    # char amplitude = strain_factor*h0
    #                = sqrt(freq^2/(dfreq/dt)/8)*h0
    if (flag_include_old_gw_freq == 1):
        delta_nu = np.abs(old_gw_freq - nu_gw)
        delta_nu_delta_timestep = delta_nu/timestep_units
        nu_squared = (nu_gw*nu_gw)
        strain_factor[nu_gw > (1e-6)*u.Hz] = np.sqrt((nu_squared[nu_gw > (1e-6)*u.Hz] / delta_nu_delta_timestep[nu_gw > (1e-6)*u.Hz])/8.)
    # Condition from evolve_gw
    elif (flag_include_old_gw_freq == 0):
        strain_factor[nu_gw > (1e-6)*u.Hz] = np.full(np.sum(nu_gw > (1e-6)*u.Hz), 4.e3)
    char_strain = strain_factor*strain

    return (char_strain.value, nu_gw.value)


def evolve_gw(blackholes_binary, smbh_mass, agn_redshift):

    char_strain, nu_gw = gw_strain_freq(mass_1=blackholes_binary.mass_1,
                                        mass_2=blackholes_binary.mass_2,
                                        obj_sep=blackholes_binary.bin_sep,
                                        timestep_duration_yr=-1,
                                        old_gw_freq=-1,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=0)

    # Update binaries
    blackholes_binary.gw_freq = nu_gw
    blackholes_binary.gw_strain = char_strain

    return (blackholes_binary)


def bbh_gw_params(blackholes_binary, bh_binary_id_num_gw, smbh_mass, timestep_duration_yr, old_bbh_freq, agn_redshift):
    """
    This function evaluates the binary gravitational wave frequency and strain at the end of each timestep_duration_yr.
    Set up binary GW frequency nu_gw = 1/pi *sqrt(GM_bin/a_bin^3). Set up binary strain of GW
    h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
    where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)


    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole 
        Full binary array.
    bh_binary_id_num_gw : numpy array
        ID numbers of binaries with separations below min_bbh_gw_separation
    smbh_mass : float
        Mass of the SMBH
    timestep_duration_yr : float
        timestep in years
    old_bbh_freq : numpy array
        Previous gw_freq
    agn_redshift : float
        Redshift of the AGN, used to set d_obs

    Returns
    -------
    char_strain : numpy array
        characteristic strain, unitless
    nu_gw : numpy array
        GW frequency in Hz
    """

    num_tracked = bh_binary_id_num_gw.size

    old_bbh_freq = old_bbh_freq * u.Hz

    while (num_tracked > len(old_bbh_freq)):
        old_bbh_freq = np.append(old_bbh_freq, (9.e-7) * u.Hz)

    while (num_tracked < len(old_bbh_freq)):
        old_bbh_freq = np.delete(old_bbh_freq, 0)

    char_strain, nu_gw = gw_strain_freq(mass_1=blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_1"),
                                        mass_2=blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_2"),
                                        obj_sep=blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_sep"),
                                        timestep_duration_yr=timestep_duration_yr,
                                        old_gw_freq=old_bbh_freq,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=1)

    return (char_strain, nu_gw)


def evolve_emri_gw(blackholes_inner_disk, timestep_duration_yr, old_gw_freq, smbh_mass, agn_redshift):
    """
    This function evaluates the EMRI gravitational wave frequency and strain at the end of each timestep_duration_yr

    Parameters
    ----------
    blackholes_inner_disk : AGNBlackHole
        black holes in the inner disk
    timestep_duration_yr : float
        timestep in years
    old_gw_freq : numpy array
        previous GW frequency in Hz
    smbh_mass : float
        mass of the SMBH in Msun
    agn_redshift : float
        redshift of the AGN
    """

    old_gw_freq = old_gw_freq * u.Hz

    # If number of EMRIs has grown since last timestep_duration_yr, add a new component to old_gw_freq to carry out dnu/dt calculation
    while (blackholes_inner_disk.num < len(old_gw_freq)):
        old_gw_freq = np.delete(old_gw_freq, 0)
    while blackholes_inner_disk.num > len(old_gw_freq):
        old_gw_freq = np.append(old_gw_freq, (9.e-7) * u.Hz)

    char_strain, nu_gw = gw_strain_freq(mass_1=smbh_mass,
                                        mass_2=blackholes_inner_disk.mass,
                                        obj_sep=blackholes_inner_disk.orb_a,
                                        timestep_duration_yr=timestep_duration_yr,
                                        old_gw_freq=old_gw_freq,
                                        smbh_mass=smbh_mass,
                                        agn_redshift=agn_redshift,
                                        flag_include_old_gw_freq=1)

    return (char_strain, nu_gw)


def ionization_check(blackholes_binary, smbh_mass):
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


def contact_check(blackholes_binary, smbh_mass):
    """
    This function tests to see if the binary separation has shrunk so that the binary is touching!
    Touching condition is where binary separation is <= R_g(M_chirp).
    Since binary separation is in units of r_g (GM_smbh/c^2) then condition is simply:
        binary_separation < M_chirp/M_smbh
    
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
    mass_chirp = np.power(blackholes_binary.mass_1 * blackholes_binary.mass_2, 3. / 5.) / np.power(mass_binary, 1. / 5.)

    # Condition is if binary separation < R_g(M_chirp). 
    # Binary separation is in units of r_g(M_smbh) so 
    # condition is separation < R_g(M_chirp)/R_g(M_smbh) =M_chirp/M_smbh
    # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
    # M1,M2, M_smbh are all in units of M_sun

    contact_condition = mass_chirp / smbh_mass

    mask_condition = (blackholes_binary.bin_sep < contact_condition)

    # If binary separation < merge condition, set binary separation to merge condition
    blackholes_binary.bin_sep[mask_condition] = contact_condition[mask_condition]
    blackholes_binary.flag_merging[mask_condition] = np.full(np.sum(mask_condition), -2)

    return (blackholes_binary)


def reality_check(blackholes_binary):
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
