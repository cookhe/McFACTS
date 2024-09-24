import numpy as np
import scipy
from mcfacts.objects.agnobject import obj_to_binary_bh_array
from astropy import constants as const
from astropy import units as u
from astropy.units import cds


def change_bin_mass(disk_bin_bhbh_pro_array, disk_bh_eddington_ratio, disk_bh_eddington_mass_growth_rate, timestep_duration_yr, bin_index):
    """Given initial binary black hole masses at start of timestep, add mass according to
        chosen BH mass accretion prescription

    Parameters
    ----------
    disk_bin_bhbh_pro_array : float array
        Array of binary black holes in prograde orbits around SMBH
    disk_bh_eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    disk_bh_eddington_mass_growth_rate : float
        fractional rate of mass growth AT Eddington accretion rate per year (2.3e-8)
    timestep_duration_yr : float
        length of timestep in units of years
    bin_index : int
        number of binaries in array

    Returns
    -------
    disk_bin_bhbh_pro_array : float array
        Updated disk_bin_bhbh_pro_array after accreting mass at prescribed rate for one timestep

    """

    bindex = int(bin_index)

    for j in range(0, bindex):
            if disk_bin_bhbh_pro_array[11,j] < 0:
                #do nothing -merger happened!
                pass
            else:
                temp_bh_mass_1 = disk_bin_bhbh_pro_array[2,j]
                temp_bh_mass_2 = disk_bin_bhbh_pro_array[3,j]
                mass_growth_factor = np.exp(disk_bh_eddington_mass_growth_rate*disk_bh_eddington_ratio*timestep_duration_yr)
                new_bh_mass_1 = temp_bh_mass_1*mass_growth_factor
                new_bh_mass_2 = temp_bh_mass_2*mass_growth_factor

                disk_bin_bhbh_pro_array[2,j] = new_bh_mass_1
                disk_bin_bhbh_pro_array[3,j] = new_bh_mass_2

    return disk_bin_bhbh_pro_array


def change_bin_mass_obj(blackholes_binary, disk_bh_eddington_ratio, disk_bh_eddington_mass_growth_rate,
                        timestep_duration_yr):

    disk_bin_bhbh_pro_array = obj_to_binary_bh_array(blackholes_binary)

    bin_index = blackholes_binary.num

    disk_bin_bhbh_pro_array = change_bin_mass(disk_bin_bhbh_pro_array, disk_bh_eddington_ratio,
                                              disk_bh_eddington_mass_growth_rate, timestep_duration_yr, bin_index)

    blackholes_binary.mass_1 = disk_bin_bhbh_pro_array[2, :]
    blackholes_binary.mass_2 = disk_bin_bhbh_pro_array[3, :]

    return (blackholes_binary)


def change_bin_spin_magnitudes(disk_bin_bhbh_pro_array, disk_bh_eddington_ratio, disk_bh_torque_condition, timestep_duration_yr, bin_index):
    """Given initial binary black hole spins at start of timestep_duration_yr, add spin according to
        chosen BH torque prescription

    Parameters
    ----------
    disk_bin_bhbh_pro_array : float array
        Array of binary black holes in prograde orbits around SMBH
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
    bin_index : int
        number of binaries in array

    Returns
    -------
    disk_bin_bhbh_pro_array : float array
        Updated disk_bin_bhbh_pro_array after spin up of BH at prescribed rate for one timestep_duration_yr

    """

    normalized_Eddington_ratio = disk_bh_eddington_ratio/1.0
    normalized_timestep = timestep_duration_yr/1.e4
    normalized_spin_torque_condition = disk_bh_torque_condition/0.1

    #max allowed spin
    max_allowed_spin=0.98
    bindex = int(bin_index)

    for j in range(0, bindex):
            if disk_bin_bhbh_pro_array[11,j] < 0:
                #do nothing -merger happened!
                pass
            else:
                temp_bh_spin_1 = disk_bin_bhbh_pro_array[4,j]
                temp_bh_spin_2 = disk_bin_bhbh_pro_array[5,j]
                spin_change_factor = 4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep
                new_bh_spin_1 = temp_bh_spin_1 + spin_change_factor
                new_bh_spin_2 = temp_bh_spin_2 + spin_change_factor
                if new_bh_spin_1 > max_allowed_spin:
                    new_bh_spin_1 = max_allowed_spin
                if new_bh_spin_2 > max_allowed_spin:
                    new_bh_spin_2 = max_allowed_spin

                disk_bin_bhbh_pro_array[4,j] = new_bh_spin_1
                disk_bin_bhbh_pro_array[5,j] = new_bh_spin_2

    return disk_bin_bhbh_pro_array


def change_bin_spin_magnitudes_obj(blackholes_binary, disk_bh_eddington_ratio, disk_bh_torque_condition,
                                   timestep_duration_yr):

    disk_bin_bhbh_pro_array = obj_to_binary_bh_array(blackholes_binary)

    bin_index = blackholes_binary.num

    disk_bin_bhbh_pro_array = change_bin_spin_magnitudes(disk_bin_bhbh_pro_array, disk_bh_eddington_ratio,
                                                         disk_bh_torque_condition, timestep_duration_yr, bin_index)

    blackholes_binary.spin_1 = disk_bin_bhbh_pro_array[4, :]
    blackholes_binary.spin_2 = disk_bin_bhbh_pro_array[5, :]

    return (blackholes_binary)


def change_bin_spin_angles(disk_bin_bhbh_pro_array, disk_bh_eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep_duration_yr, bin_index):
    #Calculate change in spin angle due to accretion during timestep_duration_yr
    normalized_Eddington_ratio = disk_bh_eddington_ratio/1.0
    normalized_timestep = timestep_duration_yr/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1

    #Extract the binary locations and spin magnitudes
    bindex = int(bin_index)
    # Run over active binaries (j is jth binary; i is the ith property of the jth binary, e.g. mass1,mass 2 etc)

    for j in range(0, bindex):
            if disk_bin_bhbh_pro_array[11,j] < 0:
                #do nothing -merger happened!
                pass
            else:
            #for i in range(0, integer_nbinprop):
                temp_bh_spin_angle_1 = disk_bin_bhbh_pro_array[6,j]
                temp_bh_spin_angle_2 = disk_bin_bhbh_pro_array[7,j]
                #bh_new_spin_angles[prograde_orb_ang_mom_indices]=bh_new_spin_angles[prograde_orb_ang_mom_indices]-(6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
                spin_angle_change_factor = (6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
                new_bh_spin_angle_1 = temp_bh_spin_angle_1 - spin_angle_change_factor
                new_bh_spin_angle_2 = temp_bh_spin_angle_2 - spin_angle_change_factor
                if new_bh_spin_angle_1 < spin_minimum_resolution:
                    new_bh_spin_angle_1 = 0.0
                if new_bh_spin_angle_2 < spin_minimum_resolution:
                    new_bh_spin_angle_2 = 0.0
                disk_bin_bhbh_pro_array[6,j] = new_bh_spin_angle_1
                disk_bin_bhbh_pro_array[7,j] = new_bh_spin_angle_2
                #print("SPIN ANGLE EVOLVES, old1,old2, new1,new2",temp_bh_spin_angle_1,temp_bh_spin_angle_2,new_bh_spin_angle_1,new_bh_spin_angle_2)
    return disk_bin_bhbh_pro_array


def change_bin_spin_angles_obj(blackholes_binary, disk_bh_eddington_ratio, spin_torque_condition,
                               spin_minimum_resolution, timestep_duration_yr):

    disk_bin_bhbh_pro_array = obj_to_binary_bh_array(blackholes_binary)

    bin_index = blackholes_binary.num

    disk_bin_bhbh_pro_array = change_bin_spin_angles(disk_bin_bhbh_pro_array, disk_bh_eddington_ratio,
                                                     spin_torque_condition, spin_minimum_resolution,
                                                     timestep_duration_yr, bin_index)

    blackholes_binary.spin_angle_1 = disk_bin_bhbh_pro_array[6, :]
    blackholes_binary.spin_angle_2 = disk_bin_bhbh_pro_array[7, :]

    return (blackholes_binary)


def com_feedback_hankla(disk_bin_bhbh_pro_array, disk_surf_func, disk_opacity_func, disk_bh_eddington_ratio, disk_alpha_viscosity, disk_radius_outer):
    """_summary_
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
    alpha = disk viscosity parameter (e.g. alpha = 0.01 in Sirko & Goodman 2003)
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
    disk_bin_bhbh_pro_array : float array
        binary array. Row 9 is center of mass of binary BH at start of timestep_duration_yr
        in units of gravitational radii (r_g=GM_SMBH/c^2)
    disk_surf_func : lambda
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_opacity_func : lambda
        Opacity as a function of radius
    disk_bh_eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    disk_alpha_viscosity : float
        Disk viscosity parameter (e.g. alpha = 0.1 in Sirko & Goodman 2003).
    disk_radius_outer : float
            final element of disk_model_radius_array (units of r_g)

    Returns
    -------
    ratio_feedback_migration_torque_bin_com : float array
        ratio of feedback torque to migration torque for each entry in prograde_bh_locations
    """
    #Extract the binary locations and masses
    temp_bin_com_locations = disk_bin_bhbh_pro_array[9,:]
    # get surface density function
    disk_surface_density = disk_surf_func(temp_bin_com_locations)

    #Define kappa (or set up a function to call).
    disk_opacity = disk_opacity_func(temp_bin_com_locations)

    ratio_feedback_migration_torque_bin_com = 0.07 * (1/disk_opacity) * ((disk_alpha_viscosity)**(-1.5)) * \
                                              disk_bh_eddington_ratio*np.sqrt(temp_bin_com_locations) / \
                                              disk_surface_density
    
    # set ratio = 1 (no migration) for binaries beyond the disk outer radius
    ratio_feedback_migration_torque_bin_com[np.where(temp_bin_com_locations > disk_radius_outer)] = 1

    return ratio_feedback_migration_torque_bin_com


def com_feedback_hankla_obj(blackholes_binary, disk_surf_func, disk_opacity_func, disk_bh_eddington_ratio, disk_alpha_viscosity, disk_radius_outer):

    disk_bin_bhbh_pro_array = obj_to_binary_bh_array(blackholes_binary)

    ratio_feedback_migration_torque_bin_com = com_feedback_hankla(disk_bin_bhbh_pro_array, disk_surf_func,
                                                                  disk_opacity_func, disk_bh_eddington_ratio,
                                                                  disk_alpha_viscosity, disk_radius_outer)

    return (ratio_feedback_migration_torque_bin_com)


def bin_migration(smbh_mass, disk_bin_bhbh_pro_array, disk_surf_model, disk_aspect_ratio_model, timestep_duration_yr, feedback_ratio, disk_radius_trap, disk_bh_pro_orb_ecc_crit, disk_radius_outer):
    """This function calculates how far the center of mass of a binary migrates in an AGN gas disk in a time
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

    #Distance travelled per binary is old location of com minus new location of com. Is +ive(-ive) if migrating in(out)
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
    mask_disk_radius_outer = disk_radius_outer > disk_bin_bhbh_pro_array
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

    blackholes_binary.bin_orb_a = disk_bin_bhbh_pro_array[9,:]

    return (blackholes_binary)



def evolve_gw(disk_bin_bhbh_pro_array, bin_index, smbh_mass, agn_redshift):
    """This function evaluates the binary gravitational wave frequency and strain at the end of each timestep_duration_yr.
    Set up binary GW frequency nu_gw = 1/pi *sqrt(GM_bin/a_bin^3). Set up binary strain of GW
    h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
    where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
    Assume binary is located at z=0.1=422Mpc for now.

    Parameters
    ----------
    disk_bin_bhbh_pro_array : float array
        Full binary array.
    bin_index : int
        number of binaries in array
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    agn_redshift : float
        redshift of the AGN, used to set d_obs

    Returns
    -------
    disk_bin_bhbh_pro_array : float array
        Returns modified disk_bin_bhbh_pro_array with updated GW properties (strain,freq) bhbh.

    """
    
    redshift_d_obs_dict = {0.1: 421,
                           0.5: 1909}
    
    for j in range(0,bin_index):
        temp_mass_1 = disk_bin_bhbh_pro_array[2,j]
        temp_mass_2 = disk_bin_bhbh_pro_array[3,j]
        temp_bin_mass = temp_mass_1 + temp_mass_2
        temp_bin_separation = disk_bin_bhbh_pro_array[8,j]
        #1rg =1AU=1.5e11m for 1e8Msun
        rg = 1.5e11*(smbh_mass/1.e8)
        m_sun = 2.0e30
        temp_mass_1_kg = m_sun*temp_mass_1
        temp_mass_2_kg = m_sun*temp_mass_2
        temp_bin_mass_kg = m_sun*temp_bin_mass
        temp_bin_separation_meters = temp_bin_separation*rg

        m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/(temp_bin_mass_kg**(1/5))
        rg_chirp = (scipy.constants.G * m_chirp)/(scipy.constants.c**(2.0))
        # If separation is less than rg_chirp then cap separation at rg_chirp.
        if temp_bin_separation_meters < rg_chirp:
            temp_bin_separation_meters = rg_chirp

        nu_gw = (1.0/scipy.constants.pi)*np.sqrt(temp_bin_mass_kg*scipy.constants.G/(temp_bin_separation_meters**(3.0)))
        disk_bin_bhbh_pro_array[19,j] = nu_gw

        # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
        # 1Mpc = 3.1e22m.
        Mpc = 3.1e22
        # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
        # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
        #d_obs = 421*Mpc
        redshift = redshift_d_obs_dict[agn_redshift]
        d_obs = (redshift*u.Mpc).to(u.meter).value  #1909*Mpc

        strain = (4/d_obs)*rg_chirp*(np.pi*nu_gw*rg_chirp/scipy.constants.c)**(2/3)
        # But power builds up in band over multiple cycles!
        # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
        strain_factor = 1
        if nu_gw < 10**(-6):
            strain_factor = np.sqrt(nu_gw*np.pi*(10**7)/8)

        if nu_gw > 10**(-6):
            strain_factor = 4.e3
        # char amplitude = sqrt(N/8)h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
        # So in LISA band
        #For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt)
        disk_bin_bhbh_pro_array[20,j] = strain_factor*strain

    return disk_bin_bhbh_pro_array


def evolve_gw_obj(blackholes_binary, smbh_mass, agn_redshift):

    #redshift_d_obs_dict = {0.1: 421*u.Mpc,
    #                       0.5: 1909*u.Mpc}

    disk_bin_bhbh_pro_array = obj_to_binary_bh_array(blackholes_binary)

    bin_index = blackholes_binary.num

    disk_bin_bhbh_pro_array = evolve_gw(disk_bin_bhbh_pro_array, bin_index, smbh_mass, agn_redshift)

    blackholes_binary.gw_freq = disk_bin_bhbh_pro_array[19, :]
    blackholes_binary.gw_strain = disk_bin_bhbh_pro_array[20, :]

    return (blackholes_binary)


def bbh_gw_params_old(disk_bin_bhbh_pro_array, bbh_gw_indices, smbh_mass, timestep_duration_yr, old_bbh_freq, agn_redshift):
    """This function evaluates the binary gravitational wave frequency and strain at the end of each timestep_duration_yr
    Set up binary GW frequency nu_gw = 1/pi *sqrt(GM_bin/a_bin^3).
    Set up binary strain of h0 = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
    where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
    Assume binary is located at z=0.1=422Mpc for now.
    For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc and 1Mpc = 3.1e22m.
    From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
    (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
    But power builds up in band over multiple cycles!
    So characteristic strain amplitude measured by e.g. LISA is given by
            h_char^2 = N/8*h_0^2
    where N is number of cycles per year & divide by 8 to average over viewing angles.
    So, h_char is given by
    char amplitude = strain_factor*h0
                   = sqrt(N/8)*h_0
    and for a source that is approximately constant frequency over a year, N=freq*1yr.
    For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt). In that case:
    char amplitude = strain_factor*h0
                   = sqrt(freq^2/(dfreq/dt)/8)*h_0

    Parameters
    ----------
    disk_bin_bhbh_pro_array : float array
        Full binary array.
    bbh_gw_indices : int
        indices of bhbh pro binaries in array that have bin separations < min_bbh_gw_sep (=2.0r_g,SMBH by default).
        These BHBH are at small enough separations that they are in the LISA GW band.
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    timestep_duration_yr : float
        size of timestep_duration_yr in years
    old_bbh_freq : float 
        gw freq of this BHBH pro binary on previous timestep, so characteristic strain can be calculated (function of rate of change over timestep)
    agn_redshift : float
        redshift of the AGN, used to set d_obs      

    Returns
    -------
    disk_bin_bhbh_pro_array : float array
        Returns modified disk_bin_bhbh_pro_array with updated GW properties (strain,freq) bhbh.
    """

    redshift_d_obs_dict = {0.1: 421,
                           0.5: 1909}
    
    year =3.15e7
    timestep_secs = (timestep_duration_yr*u.year).to(u.second).value #timestep_duration_yr*year
    # If there are BBH that meet the condition (ie if bbh_gw_indices exists, is not empty)
    if (len(bbh_gw_indices) > 0):
        num_tracked = np.size(bbh_gw_indices)
        char_strain=np.zeros(num_tracked)
        nu_gw=np.zeros(num_tracked)
        # If number of BBH tracked has grown since last timestep, add a new component to old_gw_freq to carry out dnu/dt calculation
        while num_tracked > len(old_bbh_freq):
            old_bbh_freq = np.append(old_bbh_freq,9.e-7)
        # If number of BBH tracked has shrunk. Reduce old_bbh_freq to match size of num_tracked.
        while num_tracked < len(old_bbh_freq):
            old_bbh_freq = np.delete(old_bbh_freq,0)

        #for j in range(0,num_tracked):
        for j,bindex in enumerate(bbh_gw_indices[0]):
            temp_mass_1 = disk_bin_bhbh_pro_array[2,bindex]
            temp_mass_2 = disk_bin_bhbh_pro_array[3,bindex]
            temp_bin_mass = temp_mass_1 + temp_mass_2
            temp_bin_separation = disk_bin_bhbh_pro_array[8,bindex]
            #1rg =1AU=1.5e11m for 1e8Msun
            rg = 1.5e11*(smbh_mass/1.e8)
            m_sun = 2.0e30
            temp_mass_1_kg = ((temp_mass_1*cds.Msun).to(u.kg)).value  #m_sun*temp_mass_1
            temp_mass_2_kg = ((temp_mass_2*cds.Msun).to(u.kg)).value #m_sun*temp_mass_2
            temp_bin_mass_kg = ((temp_bin_mass*cds.Msun).to(u.kg)).value #m_sun*temp_bin_mass
            temp_bin_separation_meters = temp_bin_separation*rg

            m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/(temp_bin_mass_kg**(1/5))
            rg_chirp = (scipy.constants.G * m_chirp)/(scipy.constants.c**(2.0))
            # If separation is less than rg_chirp then cap separation at rg_chirp.
            if temp_bin_separation_meters < rg_chirp:
                temp_bin_separation_meters = rg_chirp

            nu_gw[j] = (1.0/scipy.constants.pi)*np.sqrt(temp_bin_mass_kg*scipy.constants.G/(temp_bin_separation_meters**(3.0)))
            # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc 
            # 1Mpc = 3.1e22m. 
            Mpc = 3.1e22
            # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
            # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
            redshift = redshift_d_obs_dict[agn_redshift]
            d_obs = (redshift*u.Mpc).to(u.meter).value  #1909*Mpc
            strain = (4/d_obs)*rg_chirp*(np.pi*nu_gw[j]*rg_chirp/const.c.value)**(2/3)
            # But power builds up in band over multiple cycles! 
            # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
            strain_factor = 1

            if nu_gw[j] < 10**(-6):
            # char amplitude = strain_factor*h0
            #                = sqrt(N/8)*h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
                strain_factor = np.sqrt(nu_gw[j]*np.pi*(10**7)/8)

            if nu_gw[j] > 10**(-6):
            #For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt).
            # char amplitude = strain_factor*h0
            #                = sqrt(freq^2/(dfreq/dt)/8)*h0

                dnu = np.abs(old_bbh_freq[j]-nu_gw[j])
                dnu_dt = dnu/timestep_secs
                nusq = nu_gw[j]*nu_gw[j]
                strain_factor = np.sqrt((nusq/dnu_dt)/8)
            char_strain[j] = strain_factor*strain        

    return (char_strain, nu_gw)


def bbh_gw_params(blackholes_binary, bh_binary_id_num_gw, smbh_mass, timestep_duration_yr, old_bbh_freq, agn_redshift):

    redshift_d_obs_dict = {0.1: 421*u.Mpc,
                           0.5: 1909*u.Mpc}

    timestep_units = (timestep_duration_yr*u.year).to(u.second)

    num_tracked = bh_binary_id_num_gw.size

    old_bbh_freq = old_bbh_freq*u.Hz

    while (num_tracked > len(old_bbh_freq)):
        old_bbh_freq = np.append(old_bbh_freq, (9.e-7)*u.Hz)

    while (num_tracked < len(old_bbh_freq)):
        old_bbh_freq = np.delete(old_bbh_freq, 0)

    #1rg =1AU=1.5e11m for 1e8Msun
    rg = 1.5e11*(smbh_mass/1.e8)*u.meter
    mass_1 = (blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_1") * cds.Msun).to(u.kg)
    mass_2 = (blackholes_binary.at_id_num(bh_binary_id_num_gw, "mass_2") * cds.Msun).to(u.kg)
    mass_total = mass_1 + mass_2
    bin_sep = blackholes_binary.at_id_num(bh_binary_id_num_gw, "bin_sep") * rg

    mass_chirp = np.power(mass_1 * mass_2, 3./5.) / np.power(mass_total, 1./5.)
    rg_chirp = ((const.G * mass_chirp) / np.power(const.c, 2)).to(u.meter)

    # If separation is less than rg_chirp then cap separation at rg_chirp.
    bin_sep[bin_sep < rg_chirp] = rg_chirp[bin_sep < rg_chirp]

    nu_gw = (1.0/np.pi)*np.sqrt(
                    mass_total *
                    const.G /
                    (bin_sep**(3.0)))
    nu_gw = nu_gw.to(u.Hz)

    # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc 
    d_obs = redshift_d_obs_dict[agn_redshift].to(u.meter)
    #d_obs = (1909*u.Mpc).to(u.meter)
    # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
    # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
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
    delta_nu = np.abs(old_bbh_freq - nu_gw)
    delta_nu_delta_timestep = delta_nu/timestep_units
    nu_squared = (nu_gw*nu_gw)
    strain_factor[nu_gw > (1e-6)*u.Hz] = np.sqrt((nu_squared[nu_gw > (1e-6)*u.Hz] / delta_nu_delta_timestep[nu_gw > (1e-6)*u.Hz])/8.)
    char_strain = strain_factor*strain

    return (char_strain.value, nu_gw.value)



def evolve_emri_gw(inner_disk_locations,inner_disk_masses, smbh_mass,timestep_duration_yr,old_gw_freq):
    """This function evaluates the EMRI gravitational wave frequency and strain at the end of each timestep_duration_yr

    Set up binary GW frequency nu_gw = 1/pi *sqrt(GM_bin/a_bin^3). 
    Set up binary strain of h0 = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
    where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
    Assume binary is located at z=0.1=422Mpc for now.
    For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc and 1Mpc = 3.1e22m.
    From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
    (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
    But power builds up in band over multiple cycles!
    So characteristic strain amplitude measured by e.g. LISA is given by
            h_char^2 = N/8*h_0^2
    where N is number of cycles per year & divide by 8 to average over viewing angles.
    So, h_char is given by
    char amplitude = strain_factor*h0
                   = sqrt(N/8)*h_0
    and for a source that is approximately constant frequency over a year, N=freq*1yr.
    For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt). In that case:
    char amplitude = strain_factor*h0
                   = sqrt(freq^2/(dfreq/dt)/8)*h_0

    Parameters
    ----------
    disk_bin_bhbh_pro_array : float array
        Full binary array.
    bbh_gw_indices : int
        indices of bhbh pro binaries in array that have bin separations < min_bbh_gw_sep (=2.0r_g,SMBH by default).
        These BHBH are at small enough separations that they are in the LISA GW band.
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    timestep_duration_yr : float
        size of timestep_duration_yr in years
    old_gw_freq : float
        gw freq of this EMRI pro on previous timestep, so characteristic strain can be calculated (function of rate of change over timestep)

    Returns
    -------
    disk_bin_bhbh_pro_array : float array
        Returns modified disk_bin_bhbh_pro_array with updated GW properties (strain,freq) bhbh.

    """
    # Set up binary GW frequency
    # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
    num_emris = np.size(inner_disk_locations)

    char_strain = np.zeros(num_emris)
    nu_gw = np.zeros(num_emris)

    m1 = smbh_mass

    #If number of EMRIs has grown since last timestep_duration_yr, add a new component to old_gw_freq to carry out dnu/dt calculation
    #if num_emris > len(old_gw_freq):
    #    old_gw_freq = np.append(old_gw_freq,9.e-7)

    while (num_emris < len(old_gw_freq)):
        old_gw_freq = np.delete(old_gw_freq, 0)
    while num_emris > len(old_gw_freq):
        old_gw_freq = np.append(old_gw_freq, 9.e-7)

    for i in range(0,num_emris):
        m2 = inner_disk_masses[i]
        temp_bin_mass = m1 + m2
        temp_bin_separation = inner_disk_locations[i]
        # Catch issues when sep is already 0.0--pretend it's just at the event horizon
        if temp_bin_separation < 1.0: temp_bin_separation = 1.0
        #1rg =1AU=1.5e11m for 1e8Msun
        rg = 1.5e11*(smbh_mass/1.e8)
        m_sun = 2.0e30
        temp_mass_1_kg = m_sun*m1
        temp_mass_2_kg = m_sun*m2
        temp_bin_mass_kg = m_sun*temp_bin_mass
        temp_bin_separation_meters = temp_bin_separation*rg
        #Year in seconds. Multiply to get timestep_duration_yr in seconds
        year = 3.15e7
        timestep_secs = year*timestep_duration_yr

        # Set up binary strain of GW
        # h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
        # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)

        m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3/5))/(temp_bin_mass_kg**(1/5))
        rg_chirp = (scipy.constants.G * m_chirp)/(scipy.constants.c**(2.0))
        # If separation is less than rg_chirp then cap separation at rg_chirp.
        if temp_bin_separation_meters < rg_chirp:
            temp_bin_separation_meters = rg_chirp

        nu_gw[i] = (1.0/scipy.constants.pi)*np.sqrt(temp_bin_mass_kg*scipy.constants.G/(temp_bin_separation_meters**(3.0)))

        # For local distances, approx d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
        # 1Mpc = 3.1e22m.
        Mpc = 3.1e22
        # From Ned Wright's calculator https://www.astro.ucla.edu/~wright/CosmoCalc.html
        # (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
        d_obs = 1909*Mpc
        strain = (4/d_obs)*rg_chirp*(np.pi*nu_gw[i]*rg_chirp/scipy.constants.c)**(2/3)
        # But power builds up in band over multiple cycles!
        # So characteristic strain amplitude measured by e.g. LISA is given by h_char^2 = N/8*h_0^2 where N is number of cycles per year & divide by 8 to average over viewing angles
        strain_factor = 1

        if nu_gw[i] < 10**(-6):
            # char amplitude = strain_factor*h0
            #                = sqrt(N/8)*h_0 and N=freq*1yr for approx const. freq. sources over ~~yr.
            strain_factor = np.sqrt(nu_gw[i]*np.pi*(10**7)/8)

        if nu_gw[i] > 10**(-6):
            #For a source changing rapidly over 1 yr, N~freq^2/ (dfreq/dt).
            # char amplitude = strain_factor*h0
            #                = sqrt(freq^2/(dfreq/dt)/8)
            dnu = np.abs(old_gw_freq[i]-nu_gw[i])
            dnu_dt = dnu/timestep_secs
            nusq = nu_gw[i]*nu_gw[i]
            strain_factor = np.sqrt((nusq/dnu_dt)/8)


        char_strain[i] = strain_factor*strain

    return char_strain,nu_gw

def ionization_check(disk_bin_bhbh_pro_array, bin_index, smbh_mass):
    """This function tests whether a binary has been softened beyond some limit.
        Returns index of binary to be ionized. Otherwise returns -1.
        The limit is set to some fraction of the binary Hill sphere, frac_R_hill

        Default is frac_R_hill =1.0 (ie binary is ionized at the Hill sphere).
        Change frac_R_hill if you're testing binary formation at >R_hill.

        R_hill = a_com*(M_bin/3M_smbh)^1/3

        where a_com is the radial disk location of the binary center of mass (given by disk_bin_bhbh_pro_array[9,*]),
        M_bin = M_1 + M_2 = disk_bin_bhbh_pro_array[2,*]+disk_bin_bhbh_pro_array[3,*] is the binary mass
        M_smbh is the SMBH mass (given by smbh_mass)

        and binary separation is in disk_bin_bhbh_pro_array[8,*].
        Condition is
        if bin_separation > frac_R_hill*R_hill:
            Ionize binary. Return flag valued at index of binary in disk_bin_bhbh_pro_array.
            Then in test1.py remove binary from disk_bin_bhbh_pro_array! decrease bin_index by 1.
            Add two new singletons to the singleton arrays.
        Parameters
        ----------
        disk_bin_bhbh_pro_array : float array
            Full binary array.
        bin_index : int
            number of binaries in array
        smbh_mass : float
            mass of supermassive black hole in units of solar masses

        Returns
        -------
        ionization_flag : int 
            Returns index of binary to be ionized. Otherwise returns -1.

    """
    #Define Ionization threshold as fraction of Hill sphere radius
    #Default is 1.0. Change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    #Default return for the function is -1
    ionization_flag = -1.0
    for j in range(0,bin_index):
        # Read in binary masses (units of M_sun)
        temp_mass_1 = disk_bin_bhbh_pro_array[2,j]
        temp_mass_2 = disk_bin_bhbh_pro_array[3,j]
        # Set up binary mass (units of M_sun)
        temp_bin_mass = temp_mass_1 + temp_mass_2
        # Mass ratio of binary to SMBH (unitless)
        temp_mass_ratio = temp_bin_mass/smbh_mass
        # Read in binary separation (units of r_g of the SMBH =GM_smbh/c^2)
        temp_bin_separation = disk_bin_bhbh_pro_array[8,j]
        # Read in binary com disk location ( units of r_g of the SMBH = GM_smbh/c^2)
        temp_bin_com_radius = disk_bin_bhbh_pro_array[9,j]
        #Define binary Hill sphere (units of r_g of SMBH where 1r_g = GM_smbh/c^2 = 1AU for 10**8Msun SMBH
        temp_hill_sphere = temp_bin_com_radius*((temp_mass_ratio/3)**(1/3))

        if temp_bin_separation > frac_rhill*temp_hill_sphere:
            #Commented out for now
            # print("Ionize binary!", temp_bin_separation, frac_rhill*temp_hill_sphere)
            #Ionize binary!!!
            # print("disk_bin_bhbh_pro_array index",j)
            ionization_flag = j

    return ionization_flag


def ionization_check_obj(blackholes_binary, smbh_mass):
    # Remove returning -1 if that's not how it's supposed to work
    # Define ionization threshold as a fraction of Hill sphere radius
    # Default is 1.0, change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    mass_ratio = blackholes_binary.mass_total/smbh_mass
    hill_sphere = blackholes_binary.bin_orb_a * np.power(mass_ratio / 3, 1. / 3.)

    bh_id_nums = blackholes_binary.id_num[np.where(blackholes_binary.bin_sep > (frac_rhill*hill_sphere))[0]]

    return (bh_id_nums)


def contact_check(disk_bin_bhbh_pro_array, bin_index, smbh_mass):
    """ This function tests to see if the binary separation has shrunk so that the binary is touching!
        Touching condition is where binary separation is <= R_g(M_chirp).
        Since binary separation is in units of r_g (GM_smbh/c^2) then condition is simply:
            binary_separation < M_chirp/M_smbh
        Parameters
        ----------
        disk_bin_bhbh_pro_array : float array
            Full binary array.
        bin_index : int
            number of binaries in array
        smbh_mass : float
            mass of supermassive black hole in units of solar masses

        Returns
        -------
        disk_bin_bhbh_pro_array : float array
            Returns modified disk_bin_bhbh_pro_array with updated GW properties (strain,freq) bhbh.
        """
    for j in range(0,bin_index):
        #Read in mass 1, mass 2 (units of M_sun)
        temp_mass_1 = disk_bin_bhbh_pro_array[2,j]
        temp_mass_2 = disk_bin_bhbh_pro_array[3,j]
        #Total binary mass
        temp_bin_mass = temp_mass_1 + temp_mass_2
        #Binary separation in units of r_g=GM_smbh/c^2
        temp_bin_separation = disk_bin_bhbh_pro_array[8,j]

        #Condition is if binary separation < R_g(M_chirp). 
        # Binary separation is in units of r_g(M_smbh) so 
        # condition is separation < R_g(M_chirp)/R_g(M_smbh) =M_chirp/M_smbh
        # where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
        # M1,M2, M_smbh are all in units of M_sun
        m_chirp = ((temp_mass_1*temp_mass_2)**(3/5))/(temp_bin_mass**(1/5))
        condition = m_chirp/smbh_mass
        # If binary separation < merge condition, set binary separation to merge condition
        if temp_bin_separation < condition:
            disk_bin_bhbh_pro_array[8,j] = condition
            disk_bin_bhbh_pro_array[11,j] = -2
    return disk_bin_bhbh_pro_array


def contact_check_obj(blackholes_binary, smbh_mass):

    disk_bin_bhbh_pro_array = obj_to_binary_bh_array(blackholes_binary)

    bin_index = blackholes_binary.num

    disk_bin_bhbh_pro_array = contact_check(disk_bin_bhbh_pro_array, bin_index, smbh_mass)

    blackholes_binary.bin_sep = disk_bin_bhbh_pro_array[8, :]
    blackholes_binary.flag_merging = disk_bin_bhbh_pro_array[11, :]

    return(blackholes_binary)


def ionization_check(disk_bin_bhbh_pro_array, bin_index, smbh_mass):
    """This function tests whether a binary has been softened beyond some limit.
        Returns index of binary to be ionized. Otherwise returns -1.
        The limit is set to some fraction of the binary Hill sphere, frac_R_hill

        Default is frac_R_hill =1.0 (ie binary is ionized at the Hill sphere).
        Change frac_R_hill if you're testing binary formation at >R_hill.

        R_hill = a_com*(M_bin/3M_smbh)^1/3

        where a_com is the radial disk location of the binary center of mass (given by disk_bin_bhbh_pro_array[9,*]),
        M_bin = M_1 + M_2 = disk_bin_bhbh_pro_array[2,*]+disk_bin_bhbh_pro_array[3,*] is the binary mass
        M_smbh is the SMBH mass (given by smbh_mass)

        and binary separation is in disk_bin_bhbh_pro_array[8,*].
        Condition is
        if bin_separation > frac_R_hill*R_hill:
            Ionize binary. Return flag valued at index of binary in disk_bin_bhbh_pro_array.
            Then in test1.py remove binary from disk_bin_bhbh_pro_array! decrease bin_index by 1.
            Add two new singletons to the singleton arrays.
        Parameters
        ----------
        disk_bin_bhbh_pro_array : float array
            Full binary array.
        bin_index : int
            number of binaries in array
        smbh_mass : float
            mass of supermassive black hole in units of solar masses

        Returns
        -------
        disk_bin_bhbh_pro_array : float array
            Returns modified disk_bin_bhbh_pro_array with updated GW properties (strain,freq) bhbh.

    """
    #Define Ionization threshold as fraction of Hill sphere radius
    #Default is 1.0. Change only if condition for binary formation is set for separation > R_hill
    frac_rhill = 1.0

    #Default return for the function is -1
    ionization_flag = -1.0
    for j in range(0,bin_index):
        # Read in binary masses (units of M_sun)
        temp_mass_1 = disk_bin_bhbh_pro_array[2,j]
        temp_mass_2 = disk_bin_bhbh_pro_array[3,j]
        # Set up binary mass (units of M_sun)
        temp_bin_mass = temp_mass_1 + temp_mass_2
        # Mass ratio of binary to SMBH (unitless)
        temp_mass_ratio = temp_bin_mass/smbh_mass
        # Read in binary separation (units of r_g of the SMBH =GM_smbh/c^2)
        temp_bin_separation = disk_bin_bhbh_pro_array[8,j]
        # Read in binary com disk location ( units of r_g of the SMBH = GM_smbh/c^2)
        temp_bin_com_radius = disk_bin_bhbh_pro_array[9,j]
        #Define binary Hill sphere (units of r_g of SMBH where 1r_g = GM_smbh/c^2 = 1AU for 10**8Msun SMBH
        temp_hill_sphere = temp_bin_com_radius*((temp_mass_ratio/3)**(1/3))

        if temp_bin_separation > frac_rhill*temp_hill_sphere:
            #Comment out for now
            # print("Ionize binary!", temp_bin_separation, frac_rhill*temp_hill_sphere)
            #Ionize binary!!!
            # print("disk_bin_bhbh_pro_array index",j)
            ionization_flag = j

    return ionization_flag

def reality_check_old(disk_bin_bhbh_pro_array, bin_index, nbin_properties):
    """ This function tests to see if the binary is real. If location = 0 or mass = 0 *and* any other element is NON-ZERO then discard this binary element.
        Returns flag, negative for default, if positive it is the index of the binary column to be deleted.

        Parameters
        ----------
        disk_bin_bhbh_pro_array : float array 
            Full binary array.
        bin_index : int
            number of binaries in array
        nbin_properties : int
            number of binary properties

        Returns
        -------
        reality_flag : int
            -2 if binaries are real, greater than 0 if binaries are not real.
        """
    reality_flag = -2

    for j in range(0,bin_index):

        #Check other elements in disk_bin_bhbh_pro_array are NON-ZERO
        for i in range(0,nbin_properties):
            #Read in mass 1, mass 2 (units of M_sun)
            temp_mass_1 = disk_bin_bhbh_pro_array[2,j]
            temp_mass_2 = disk_bin_bhbh_pro_array[3,j]
            #Read in location 1, location 2 (units of R_g (M_smbh))
            temp_location_1 = disk_bin_bhbh_pro_array[0,j]
            temp_location_2 = disk_bin_bhbh_pro_array[1,j]
            #Read in bin coms (units of R_g(M_smbh))
            temp_bin_com = disk_bin_bhbh_pro_array[9,j]
            #If any element in binary other than the location or mass is non-zero
            if disk_bin_bhbh_pro_array[i,j] > 0:
                #Check if any of locations or masses is zero
                if temp_mass_1 == 0 or temp_mass_2 == 0 or temp_location_1 == 0 or temp_location_2 == 0 or np.isnan(temp_bin_com):
                    #Flag this binary
                    reality_flag = j

    return reality_flag


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
