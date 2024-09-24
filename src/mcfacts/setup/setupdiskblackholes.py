import numpy as np
from mcfacts.mcfacts_random_state import rng



def setup_disk_blackholes_location(disk_bh_num, disk_outer_radius,disk_inner_stable_circ_orb):
    """Returns an array of initial single BH semi-major axes (disk_bh_orb_a) distributed randomly uniformly through disk of radial size disk_outer_radius

    Parameters
    ----------
    disk_bh_num : int
        Integer number of BH initially embedded in disk
    disk_outer_radius : float
        Outer radius of disk in units of r_g,SMBH
    disk_inner_stable_circ_orb : float
        Inner radius of disk in units of r_g,SMBH    
    Returns
    -------
    bh_initial_locations: float array
        Array of initial BH locations in disk (disk_bh_orb_a)
    """
    # ISCO defined here. Need to put this in the .ini file.
    #disk_inner_stable_circ_orbit = 6.0
    integer_nbh = int(disk_bh_num)
    bh_initial_locations = disk_outer_radius*rng.random(integer_nbh)
    sma_too_small = np.where(bh_initial_locations < disk_inner_stable_circ_orb)
    bh_initial_locations[sma_too_small] = disk_inner_stable_circ_orb
    return bh_initial_locations

def setup_prior_blackholes_indices(prograde_n_bh, prior_bh_locations):
    #Return an array of indices which allow us to read prior BH properties & replace prograde BH with these.
    integer_nbh = int(prograde_n_bh)
    len_prior_locations = (prior_bh_locations.size)-1
    bh_indices = np.rint(len_prior_locations*rng.random(integer_nbh))
    return bh_indices

def setup_disk_blackholes_masses(disk_bh_num,nsc_bh_imf_mode,nsc_bh_imf_max_mass,nsc_bh_imf_powerlaw_index,mass_pile_up):
    """Return an array of disk BH initial masses of size disk_bh_num for user defined inputs.

    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
        nsc_bh_imf_mode : float
            Nuclear Star Cluster BH initial mass function , mode. Units of M_sun. User set (default = 10)
        nsc_bh_inf_max_mass : float
            Nuclear Star Cluster BH IMF, maximum mass. Units of M_sun. User set (default = 40)
        nsc_bh_imf_powerlaw_index : float
            Nuclear Star Cluster BH IMF (e.g. M^-2). Powerlaw index. Unitless. User set (default = 2)
        mass_pile_up : float
            Mass pile up term <nsc_bh_inf_max_mass. Units of M_sun. User set (default =35.) 
            Used to make a uniform pile up in mass between [mass_pile_up,nsc_bh_inf_max_mass] for masses selected
            from nsc_bh_imf_powerlaw_index beyond nsc_bh_inf_max_mass. E.g default [35,40] pile up of masses.    

    Returns:
        disk_bh_initial_masses: float array
            Array of disk BH initial masses
    """
    
    integer_nbh = int(disk_bh_num)
    disk_bh_initial_masses = (rng.pareto(nsc_bh_imf_powerlaw_index,integer_nbh)+1)*nsc_bh_imf_mode
    #impose mass pile up condition (should be set in .ini). Default is 35Msun (for max of 40Msun).
    #critical_bh_mass = 35.0
    mass_diff = nsc_bh_imf_max_mass - mass_pile_up
    disk_bh_initial_masses[disk_bh_initial_masses > nsc_bh_imf_max_mass] = mass_pile_up + np.rint(mass_diff*rng.random())
    return disk_bh_initial_masses


def setup_disk_blackholes_spins(disk_bh_num, nsc_bh_spin_dist_mu, nsc_bh_spin_dist_sigma):
    """Return an array of disk BH initial spins (unitless spin parameter) of size disk_bh_num for user defined 
    Gaussian spin distribution centered on mu (default = 0) and variance sigma(default = 0.2).

    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
        nsc_bh_spin_dist_mu : float
            Mu of BH spin distribution (centroid of Gaussian)
        nsc_bh_spin_dist_sigma : float
            Sigma of BH spin distribution (variance of Gaussian)

    Returns
    -------
        disk_bh_initial_spins : float array
            Array of initial BH spins of size disk_bh_num
    """
    
    integer_nbh = int(disk_bh_num)
    disk_bh_initial_spins = rng.normal(nsc_bh_spin_dist_mu, nsc_bh_spin_dist_sigma, integer_nbh)
    return disk_bh_initial_spins


def setup_disk_blackholes_spin_angles(disk_bh_num, disk_bh_initial_spins):
    """Return an array of disk BH initial spin_angles (units of radians) of size disk_bh_num drawn from random uniform distribution.
    Positive (negative) spin magnitudes have spin angles [0,1.57]([1.5701,3.14])rads
    All BH spin angles are initially drawn from a uniform distribution of [0,1.57]rads. For BH with negative spins, we add +1.57rads.
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
        disk_bh_initial_spins : float array
            Array of initial BH spins 

    Returns
    -------
        disk_bh_initial_spins : float array
            Array of initial BH spins of size disk_bh_num
    """
    
    integer_nbh = int(disk_bh_num)
    bh_initial_spin_indices = np.array(disk_bh_initial_spins)
    negative_spin_indices = np.where(bh_initial_spin_indices < 0.)
    disk_bh_initial_spin_angles = rng.uniform(0.,1.57,integer_nbh)
    disk_bh_initial_spin_angles[negative_spin_indices] = disk_bh_initial_spin_angles[negative_spin_indices] + 1.57
    return disk_bh_initial_spin_angles


def setup_disk_blackholes_orb_ang_mom(disk_bh_num):
    """Return an array of disk BH initial orbital angular momenta. 
    Assume either initially fully prograde (+1) or retrograde (-1)
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk

    Returns
    -------
        disk_bh_initial_orb_ang_mom : float array
            Array of initial BH orb ang mom of size disk_bh_num
    """
    
    integer_nbh = int(disk_bh_num)
    random_uniform_number = rng.random((integer_nbh,))
    disk_bh_initial_orb_ang_mom = (2.0*np.around(random_uniform_number)) - 1.0
    return disk_bh_initial_orb_ang_mom

def setup_disk_blackholes_eccentricity_thermal(disk_bh_num):
    """Return an array of disk BH initial orbital eccentricities of size disk_bh_num. 
    Assumes a thermal distribution (uniform in e^2, i.e. e^2=[0,1] so median(e^2)=0.5 and so median(e)~0.7. 
    This might be appropriate for e.g. a galactic nucleus that is very relaxed 
    and has not had any nuclear activity for a long time.
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk

    Returns
    -------
        disk_bh_initial_orb_ecc : float array
            Array of initial BH orb eccentricity of size disk_bh_num
    """
    
    integer_nbh = int(disk_bh_num)
    random_uniform_number = rng.random((integer_nbh,))
    disk_bh_initial_orb_ecc = np.sqrt(random_uniform_number)
    return disk_bh_initial_orb_ecc

def setup_disk_blackholes_eccentricity_uniform(disk_bh_num, disk_bh_orb_ecc_max_init):
    """Return an array of disk BH initial orbital eccentricities of size disk_bh_num. 
    Assumes a uniform distribution in orb_ecc, up to disk_bh_orb_ecc_max_init 
    i.e. e=[0,disk_bh_orb_ecc_max_init] so median(e)=disk_bh_orb_ecc_max_init/2. 
    This might be appropriate for e.g. a galactic nucleus that is recently post-AGN 
    so not had much time to relax. Most real clusters/binaries lie between thermal & uniform 
    (e.g. Geller et al. 2019, ApJ, 872, 165)
    Cap of max_initial_eccentricity allows for previous recent episode of AGN 
    where the population is relaxating from previously circularized.
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
        disk_bh_orb_ecc_max_init : float
            Maximum initial orb ecc assumed for embedded BH population in disk. 
    Returns
    -------
        disk_bh_initial_orb_ecc : float array
            Array of initial BH orb eccentricity of size disk_bh_num
    """
    
    integer_nbh = int(disk_bh_num)
    random_uniform_number = rng.random((integer_nbh,))
    bh_initial_orb_ecc = random_uniform_number*disk_bh_orb_ecc_max_init
    return bh_initial_orb_ecc

def setup_disk_blackholes_inclination(disk_bh_num):
    """Return an array of disk BH initial orbital inclinations of size disk_bh_num. 
    Right now returns an initial distribution of inclination angles that are 0.0
    To do: initialize inclinations so random draw with i <h (so will need to input bh_locations and disk_aspect_ratio)
    and then damp inclination.
    To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    Then damp (i,e) as appropriate. Return an initial distribution of inclination angles that are 0 deg.
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
    Returns
    -------
        disk_bh_initial_orb_inc : float array
            Array of initial BH orb eccentricity of size disk_bh_num
    """
    
    integer_nbh = int(disk_bh_num)
    # For now, inclinations are zeros
    disk_bh_orb_inc_init = np.zeros((integer_nbh,),dtype = float)
    return disk_bh_orb_inc

def setup_disk_blackholes_incl(disk_bh_num, disk_bh_locations, disk_bh_orb_ang_mom, disk_aspect_ratio):
    """Return an array of disk BH initial orbital inclinations of size disk_bh_num. 
    Right now returns an initial distribution of inclination angles that are 0.0
    To do: initialize inclinations so random draw with i <h (so will need to input bh_locations and disk_aspect_ratio)
    and then damp inclination.
    To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    Then damp (i,e) as appropriate. Return an initial distribution of inclination angles that are 0 deg.
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
        disk_bh_locations : float array
            Array of initial BH semi-major axes in disk
        disk_bh_orb_ang_mom : float array
            Array of initial BH orb ang mom in the disk
        disk_aspect_ratio : float array
            Array of disk height as a function of disk radius in units of r_g(SMBH)       
    Returns
    -------
        disk_bh_orb_inc_init : float array
            Array of initial BH orb eccentricity of size disk_bh_num
    """
    # Return an array of BH orbital inclinations
    # initial distribution is not 0.0
    integer_nbh = int(disk_bh_num)
    # what is the max height at the orbiter location that keeps it in the disk?
    max_height = disk_bh_locations * disk_aspect_ratio(disk_bh_locations)
    # reflect that height to get the min
    min_height = -max_height
    random_uniform_number = rng.random((integer_nbh,))
    # pick the actual height between the min and max, then reset zero point
    height_range = max_height - min_height
    actual_height_range = height_range * random_uniform_number
    actual_height = actual_height_range + min_height
    # inclination is arctan of height over radius, modulo pro or retrograde
    disk_bh_orb_inc_init = np.arctan(actual_height/disk_bh_locations)
    # for retrogrades, add 180 degrees
    disk_bh_orb_inc_init[disk_bh_orb_ang_mom < 0.0] = disk_bh_orb_inc_init[disk_bh_orb_ang_mom < 0.0] + np.pi

    return disk_bh_orb_inc_init

def setup_disk_blackholes_circularized(disk_bh_num,disk_bh_pro_orb_ecc_crit):
    """Return an array of disk BH initial orbital eccentricities of size disk_bh_num. 
    Assumes a circularized distribution in orb_ecc. Right now set to orb_ecc=0.0
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
        disk_bh_pro_orb_ecc_crit : float
            Disk BH orb ecc critical value below which orbits are assumed circularized. 
    Returns
    -------
        disk_bh_orb_ecc_init : float array
            Array of initial BH orb eccentricity of size disk_bh_num. Assumed circularized.
    """
    
    integer_nbh = int(disk_bh_num)
    
    disk_bh_orb_ecc_init = disk_bh_pro_orb_ecc_crit*np.zeros((integer_nbh,),dtype = float)
    return disk_bh_orb_ecc_init

def setup_disk_blackholes_arg_periapse(disk_bh_num):
    """Return an array of disk BH initial orb arg periapse of size disk_bh_num. 
    Assumes a orb arg. periapse either 0 or pi/2 radians. TO DO: 
    Set between [0,2pi] uniformly. 
    But issue with calculating retrograde capture when uniform to be fixed.
    
    Parameters
    ----------
        disk_bh_num : int
            Integer number of BH initially embedded in disk
        
    Returns
    -------
        disk_bh_orb_ecc_init : float array
            Array of initial BH orb eccentricity of size disk_bh_num. Assumed circularized.
    """
    
    integer_nbh = int(disk_bh_num)
    random_uniform_number = rng.random((integer_nbh,))
    
    bh_initial_orb_arg_periapse = 0.5 * np.pi * np.around(random_uniform_number)

    return bh_initial_orb_arg_periapse

def setup_disk_nbh(nsc_mass,nsc_ratio_bh_num_star_num,nsc_ratio_mbh_mass_star_mass,nsc_radius_outer,nsc_density_index_outer,smbh_mass,disk_radius_outer,disk_aspect_ratio_avg,nsc_radius_crit,nsc_density_index_inner):
    """Return the integer number of BH in the AGN disk as calculated from User inputs for NSC and disk. 
    
    Parameters
    ----------
        nsc_mass : float
            Mass of Nuclear Star Cluster (in units of M_sun). Set by User. Default is mass of Milky Way NSC = 3e7M_sun.
        nsc_ratio_bh_num_star_num : float 
            Ratio of number of BH in NSC to number of stars (unitless). Set by User. Default is 1.e-3.
        nsc_ratio_mbh_mass_star_mass : float
            Ratio of mass of typical BH in NSC to typical star in NSC (unitless). Set by User. Default is 10 (BH=10M_sun,star=1M_sun)
        nsc_radius_outer : float
            Outer radius of NSC (in units of parsec). Set by User. Default is 5pc.
        nsc_density_index_outer : float
            NSC density powerlaw index in outer regions. Set by User. 
            NSC density n(r) is assumed to consist of a broken powerlaw distribution, 
            with one powerlaw in inner regions (Bahcall-Wolf, r^-7/4 usually) and one in the outer regions. 
            This is the outer region NSC powerlaw density index. Default is n(r) \propto r^-5/2
        smbh_mass : float
            Mass of the SMBH in units of M_sun. Set by User. Default is 1.e8M_sun.
        disk_radius_outer : float
            Outer radius of disk in units of r_g (SMBH). Set by User. Default is 5.e4r_g (or 0.25pc around a 10^8M_sun)
        disk_aspect_ratio_avg : float
            Average disk aspect ratio (unitless). Set by User. Default is h=0.03.
        nsc_radius_crit : float
            NSC critical radius (units of pc). Set by User. 
            Radius at which NSC density changes from inner powerlaw index to outer powerlaw index.                             
        nsc_density_index_inner : float
            NSC density powerlaw index in inner regions (Unitless). Set by User. 
            Default is n(r) \propto r^-7/4 (Bahcall-Wolf)
        
    Returns
    -------
        disk_bh_orb_ecc_init : float array
            Array of initial BH orb eccentricity of size disk_bh_num. Assumed circularized.
    """
    
    # Convert outer disk radius in r_g to units of pc. 
    # 1r_g =1AU (M_smbh/10^8Msun) and 
    # 1pc =2e5AU =2e5 r_g(M/10^8Msun)^-1
    convert_1pc_to_rg_SMBH = 2.e5*((smbh_mass/1.e8)**(-1.0))
    #Convert user defined outer disk radius to pc.
    disk_radius_outer_pc = disk_radius_outer/convert_1pc_to_rg_SMBH
    # Total mass of BH in NSC
    total_mass_bh_in_nsc = nsc_mass * nsc_ratio_bh_num_star_num * nsc_ratio_mbh_mass_star_mass
    # Total average number of BH in NSC
    nsc_bh_num = total_mass_bh_in_nsc / nsc_ratio_mbh_mass_star_mass
    
    #Relative volumes:
    #   of central 1 pc^3 to size of NSC
    relative_volumes_at1pc = (1.0/nsc_radius_outer)**(3.0)
    #   of r_nsc_crit^3 to size of NSC
    relative_volumes_at_nsc_radius_crit = (nsc_radius_crit/nsc_radius_outer)**(3.0)
    
    #Total number of BH 
    #   at R<1pc (should be ~10^4 for Milky Way parameters; 3x10^7Msun, 5pc, r^-5/2 in outskirts)
    nsc_bh_num_inside_pc = nsc_bh_num * relative_volumes_at1pc * (1.0/nsc_radius_outer)**(-nsc_density_index_outer)
    #   at nsc_radius_crit
    nsc_bh_num_inside_radius_crit = nsc_bh_num_inside_pc * relative_volumes_at_nsc_radius_crit * (nsc_radius_crit/nsc_radius_outer)**(-nsc_density_index_outer)
    
    #Calculate Total number of BH in volume R < disk_outer_radius, assuming disk_outer_radius<=1pc.
    if disk_radius_outer_pc >= nsc_radius_crit:
        relative_volumes_at_disk_outer_radius = (disk_radius_outer_pc/1.0)**(3.0)
        nsc_bh_vol_disk_radius_outer = nsc_bh_num_inside_pc * relative_volumes_at_disk_outer_radius * ((disk_radius_outer_pc/1.0)**(-nsc_density_index_outer))          
    else:
        relative_volumes_at_disk_outer_radius = (disk_radius_outer_pc/nsc_radius_crit)**(3.0)
        nsc_bh_vol_disk_radius_outer = nsc_bh_num_inside_radius_crit * relative_volumes_at_disk_outer_radius * ((disk_radius_outer_pc/nsc_radius_crit)**(-nsc_density_index_inner))
     
    # Total number of BH in disk
    disk_bh_num = np.rint(nsc_bh_vol_disk_radius_outer * disk_aspect_ratio_avg)
     
    return np.int64(disk_bh_num)

