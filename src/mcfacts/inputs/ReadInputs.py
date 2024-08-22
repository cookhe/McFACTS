import numpy as np

import configparser as ConfigParser
from io import StringIO

# Grab those txt files
from importlib import resources as impresources
from mcfacts.inputs import data

# Dictionary of types
INPUT_TYPES = {
    'disk_model_name' : str,
    'disk_model_use_pagn': bool,
    'mass_smbh' : float,
    'trap_radius' : float,
    'disk_outer_radius' : float,
    'max_disk_radius_pc' : float,
    'alpha' : float,
    'n_iterations' : int,
    'mode_mbh_init' : float,
    'max_initial_bh_mass' : float,
    'mbh_powerlaw_index' : float,
    'mu_spin_distribution' : float,
    'sigma_spin_distribution' : float,
    'spin_torque_condition' : float,
    'frac_Eddington_ratio' : float,
    'max_initial_eccentricity' : float,
    'timestep' : float,
    'number_of_timesteps' : int,
    'retro' : float,
    'feedback' : int,
    'capture_time' : float,
    'outer_capture_radius' : float,
    'crit_ecc' : float,
    'r_nsc_out' : float,
    'M_nsc' : float,
    'r_nsc_crit' : float,
    'nbh_nstar_ratio' : float,
    'mbh_mstar_ratio' : float,
    'nsc_index_inner' : float,
    'nsc_index_outer' : float,
    'h_disk_average' : float,
    'dynamic_enc' : int,
    'de' : float,
    'orb_ecc_damping' : int,
}

def ReadInputs_ini(fname='inputs/model_choice.txt', verbose=False):
    """This function reads your input choices from a file user specifies or
    default (inputs/model_choice.txt), and returns the chosen variables for 
    manipulation by main.    

    Required input formats and units are given in IOdocumentation.txt file.

    See below for full output list, including units & formats

    Example
    -------
    To run, ensure a model_choice.txt is in the same directory and type:

        $ python ReadInputs_ini.py

    Notes
    -----
    Function will tell you what it is doing via print statements along the way.

    Attributes
    ----------
    Output variables:
    mass_smbh : float
        Mass of the supermassive black hole (M_sun)
    trap_radius : float
        Radius of migration trap in gravitational radii (r_g = G*mass_smbh/c^2)
        Should be set to zero if disk model has no trap
    n_iterations : int
        Number of iterations of code run (e.g. 1 for testing, 30 for a quick run)
    mode_mbh_init : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--mode of initial mass dist (M_sun)
    max_initial_bh_mass : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--mass of cutoff (M_sun)
    mbh_powerlaw_index : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--powerlaw index for Pareto dist
    min_initial_star_mass : float
        Initial mass distribution for stars is assumed Salpeter
    max_initial_star_mass : float
        Initial mass distribution for stars is assumed Salpeter
    star_mass_powerlaw_index : float
        Initial mass distribution for stars is assumed Salpeter, alpha = 2.35
    mu_spin_distribution : float
        Initial spin distribution for stellar bh is assumed to be Gaussian
        --mean of spin dist
    sigma_spin_distribution : float
        Initial spin distribution for stellar bh is assumed to be Gaussian
        --standard deviation of spin dist
    spin_torque_condition : float
        fraction of initial mass required to be accreted before BH spin is torqued 
        fully into alignment with the AGN disk. We don't know for sure but 
        Bogdanovic et al. says between 0.01=1% and 0.1=10% is what is required.
    frac_Eddington_ratio : float
        assumed accretion rate onto stellar bh from disk gas, in units of Eddington
        accretion rate
    max_initial_eccentricity : float
        assuming initially flat eccentricity distribution among single orbiters around SMBH
        out to max_initial_eccentricity. Eventually this will become smarter.
    mu_star_spin_distribution : float
        Initial spin distribution for stars is assumed to be Gaussian
    sigma_star_spin_distribution : float
        Initial spin distribution for stars is assumed to be Gaussian
        --standard deviation of spin dist
    spin_star_torque_condition : float
        fraction of initial mass required to be accreted before star spin is torqued 
        fully into alignment with the AGN disk. We don't know for sure but 
        Bogdanovic et al. says between 0.01=1% and 0.1=10% is what is required.
    frac_star_Eddington_ratio : float
        assumed accretion rate onto stars from disk gas, in units of Eddington
        accretion rate
    max_initial_star_eccentricity : float
        assuming initially flat eccentricity distribution among single orbiters around SMBH
        out to max_initial_eccentricity. Eventually this will become smarter.
    stars_initial_X  : float
        Stellar initial hydrogen mass fraction
    stars_initial_Y : float
        Stellar initial helium mass fraction
    stars_initial_Z : float
        Stellar initial metallicity mass fraction
    timestep : float
        How long is your timestep in years?
    number_of_timesteps : int
        How many timesteps are you taking (timestep*number_of_timesteps = disk_lifetime)
    disk_model_radius_array : float array
        The radii along which your disk model is defined in units of r_g (=G*mass_smbh/c^2)
        drawn from modelname_surface_density.txt
    disk_inner_radius : float
        0th element of disk_model_radius_array (units of r_g)
    disk_outer_radius : float
        final element of disk_model_radius_array (units of r_g)
    max_disk_radius_pc: float
        Maximum disk size in parsecs (0. for off)
    surface_density_array : float array
        Surface density corresponding to radii in disk_model_radius_array (units of kg/m^2)
        Yes, it's in SI not cgs. Get over it. Kisses.
        drawn from modelname_surface_density.txt
    aspect_ratio_array : float array
        Aspect ratio corresponding to radii in disk_model_radius_array
        drawn from modelname_aspect_ratio.txt
    retro : float
        Fraction of BBH that form retrograde to test (q,X_eff) relation. Default retro=0.1 
    feedback : int
        Switch (1) turns feedback from embedded BH on.
    orb_ecc_damping : int
        Switch (1) turns orb. ecc damping on. If switch = 0, assumes all bh are circularized (at e=e_crit)
    r_nsc_out : float
        Radius of NSC (units of pc)
    M_nsc : float
        Mass of NSC (units of M_sun)
    r_nsc_crit : float
        Radius where NSC density profile flattens (transition to Bahcall-Wolf) (units of pc)
    nbh_nstar_ratio : float
        Ratio of number of BH to stars in NSC (typically spans 3x10^-4 to 10^-2 in Generozov+18)
    mbh_mstar_ratio : float
        Ratio of mass of typical BH to typical star in NSC (typically 10:1 in Generozov+18)
    nsc_index_inner : float
        Index of radial density profile of NSC inside r_nsc_crit (usually Bahcall-Wolf, 1.75)
    nsc_index_outer : float
        Index of radial density profile of NSC outside r_nsc_crit (e.g. 2.5 in Generozov+18 or 2.25 if Peebles)
    h_disk_average : float
        Average disk scale height (e.g. about 3% in Sirko & Goodman 2003 out to ~0.3pc)
    dynamic_enc : int
        Switch (1) turns dynamical encounters between embedded BH on.
    de : float
        Average energy change per strong interaction. de can be 20% in cluster interactions. May be 10% on average (with gas)                
    prior_agn : int
        Switch (1) uses BH from a prior AGN episode (in file /recipes/postagn_bh_pop1.dat)
    """

    config = ConfigParser.ConfigParser()
    config.optionxform=str # force preserve case! Important for --choose-data-LI-seglen

    # Default format has no section headings ...
    config.read(fname)
    #with open(fname) as stream:
    #    stream = StringIO("[top]\n" + stream.read())
    #    config.read_file(stream)

    # convert to dict
    input_variables = dict(config.items('top'))


    # try to pretty-convert these to quantites
    for name in input_variables:
        if name in INPUT_TYPES:
            if INPUT_TYPES[name] == bool:
                input_variables[name] = bool(int(input_variables[name]))
            else:
                input_variables[name] = INPUT_TYPES[name](input_variables[name])
        elif '.' in input_variables[name]:
            input_variables[name]=float(input_variables[name])
        elif input_variables[name].isdigit():
            input_variables[name] =int(input_variables[name])
        else:
            input_variables[name] = str(input_variables[name])
    # Clean up strings
    for name in input_variables:
        if isinstance(input_variables[name], str):
            input_variables[name] = input_variables[name].strip("'")

    # Set default : not use pagn.  this allows us not to provide it
    if not('disk_model_use_pagn' in input_variables):
        input_variables['disk_model_use_pagn'] = False

    # Make sure you got all of the ones you were expecting
    for name in INPUT_TYPES:
        print(name)
        assert name in input_variables
        assert type(input_variables[name]) == INPUT_TYPES[name]
        
    if verbose:
        print("input_variables:")
        for key in input_variables:
            print(key, input_variables[key], type(input_variables[key]))
        print("I put your variables where they belong")

    # Return the arguments
    return input_variables

def construct_disk_interp(
    mass_smbh,
    disk_outer_radius,
    disk_model_name,
    alpha,
    frac_Eddington_ratio,
    max_disk_radius_pc=0.,
    disk_model_use_pagn=False,
    verbose=False,
    ):
    '''Construct the disk array interpolators

    Parameters
    ----------
        mass_smbh : float
            Mass of the supermassive black hole (M_sun)
        disk_outer_radius : float
            final element of disk_model_radius_array (units of r_g)
        alpha : ???
            ??? #TODO
        frac_Eddington_ratio : float
            assumed accretion rate onto stellar bh from disk gas, in units of Eddington
            accretion rate
        max_disk_radius_pc : float
            Maximum disk size in parsecs (0. for off)
        disk_model_use_pagn : bool
            use pAGN?
        verbose : bool
            Print extra stuff?

    Returns
    ------
        surf_dens_func : scipy.interpolate.UnivariateSpline.UnivariateSpline object
            Surface density interpolator
        aspect_ratio_func : scipy.interpolate.UnivariateSpline.UnivariateSpline object
            Aspect ratio interpolator
    '''
    ## Check inputs ##
    # Check mass_smbh
    assert type(mass_smbh) == float, "mass_smbh expected float, got %s"%(type(mass_smbh))
        
    ## Check outer disk radius in parsecs
    # Scale factor for parsec distance in r_g
    pc_dist = 2.e5*((mass_smbh/1.e8)**(-1.0))
    # Calculate outer disk radius in pc
    disk_outer_radius_pc = disk_outer_radius/pc_dist
    # Check max_disk_radius_pc argument
    if max_disk_radius_pc == 0.:
        # Case 1: max_disk_radius_pc is disabled
        pass
    elif max_disk_radius_pc < 0.:
        # Case 2: max_disk_radius_pc is negative
        # Always assign disk_outer_radius to given distance in parsecs
        disk_outer_radius = -1. * max_disk_radius_pc * pc_dist
    else:
        # Case 3: max_disk_radius_pc is positive
        # Cap disk_outer_radius at given value
        if disk_outer_radius_pc > max_disk_radius_pc:
            # calculate scale factor
            disk_radius_scale = max_disk_radius_pc / disk_outer_radius_pc
            # Adjust disk_outer_radius as needed
            disk_outer_radius = disk_outer_radius * disk_radius_scale
        
    # open the disk model surface density file and read it in
    # Note format is assumed to be comments with #
    #   density in SI in first column
    #   radius in r_g in second column
    #   infile = model_surface_density.txt, where model is user choice
    if not(disk_model_use_pagn):
        infile_suffix = '_surface_density.txt'
        infile = disk_model_name+infile_suffix
        infile = impresources.files(data) / infile
        dat = np.loadtxt(infile)
        disk_model_radius_array = dat[:,1]
        surface_density_array = dat[:,0]
        #truncate disk at outer radius
        truncated_disk = np.extract(
            np.where(disk_model_radius_array < disk_outer_radius),
            disk_model_radius_array
        )
        #print('truncated disk', truncated_disk)
        truncated_surface_density_array = surface_density_array[0:len(truncated_disk)]

        # open the disk model aspect ratio file and read it in
        # Note format is assumed to be comments with #
        #   aspect ratio in first column
        #   radius in r_g in second column must be identical to surface density file
        #       (radius is actually ignored in this file!)
        #   filename = model_aspect_ratio.txt, where model is user choice
        infile_suffix = '_aspect_ratio.txt'
        infile = disk_model_name+infile_suffix
        infile = impresources.files(data) / infile
        dat = np.loadtxt(infile)
        aspect_ratio_array = dat[:,0]
        truncated_aspect_ratio_array=aspect_ratio_array[0:len(truncated_disk)]


        # Now redefine arrays used to generate interpolating functions in terms of truncated arrays
        disk_model_radius_array = truncated_disk
        surface_density_array = truncated_surface_density_array
        aspect_ratio_array = truncated_aspect_ratio_array

        # Now geenerate interpolating functions
        import scipy.interpolate
        # create surface density & aspect ratio functions from input arrays
        surf_dens_func_log = scipy.interpolate.CubicSpline(
            np.log(disk_model_radius_array), np.log(surface_density_array))
        surf_dens_func = lambda x, f=surf_dens_func_log: np.exp(f(np.log(x)))

        aspect_ratio_func_log = scipy.interpolate.CubicSpline(
                np.log(disk_model_radius_array), np.log(aspect_ratio_array))
        aspect_ratio_func = lambda x, f=aspect_ratio_func_log: np.exp(f(np.log(x)))

    else:
        # instead, populate with pagn
        import mcfacts.external.DiskModelsPAGN as dm_pagn
        import pagn.constants as ct
        pagn_name = "Sirko"
        base_args = { 'Mbh': mass_smbh*ct.MSun,\
                      'alpha':alpha, \
                      'le':frac_Eddington_ratio}
        if 'thompson' in disk_model_name:
            pagn_name = 'Thompson'
            base_args['Rout'] = disk_outer_radius;
        # note Rin default is 3 Rs
        
        pagn_model =dm_pagn.AGNGasDiskModel(disk_type=pagn_name,**base_args)
        
        surf_dens_func, aspect_ratio_func, Ragn  = pagn_model.return_disk_surf_model()
        
    
    #Truncate disk models at outer disk radius
    if verbose:
        print("I read and digested your disk model")
        print("Sending variables back")

    return surf_dens_func, aspect_ratio_func

def ReadInputs_prior_mergers(fname='recipes/sg1Myrx2_survivors.dat', verbose=False):
    """This function reads your prior mergers from a file user specifies or
    default (recipies/prior_mergers_population.dat), and returns the chosen variables for 
    manipulation by main.    

    Required input formats and units are given in IOdocumentation.txt file.

    See below for full output list, including units & formats

    Example
    -------
    To run, ensure a prior_mergers_population.dat is in the same directory and type:

        $ python ReadInputs_prior_mergers.py

    Notes
    -----
    Function will tell you what it is doing via print statements along the way.

    Attributes
    ----------
    Output variables:
    radius_bh : float
        Location of BH in disk
    mass_bh : float
        Mass of BH (M_sun)
    spin_bh : float    
        Magnitude of BH spin (dimensionless)
    spin_angle_bh : float
        Angle of BH spin wrt L_disk (radians). 0(pi) radians = aligned (anti-aligned) with L_disk
    gen_bh: float
        Generation of BH (integer). 1.0 =1st gen (wasn't involved in merger in previous episode; but accretion=mass/spin changed)        
    )                
    """

    #with open('../recipes/prior_mergers_x2_population.dat') as filedata:
    #    prior_mergers_file = np.genfromtxt('../recipes/prior_mergers_x2_population.dat', unpack = True)
    
    with open('../recipes/sg1Myrx2_survivors.dat') as filedata:
        prior_mergers_file = np.genfromtxt('../recipes/sg1Myrx2_survivors.dat', unpack = True)
    
    
    #Clean the file of iteration lines (of form 3.0 3.0 3.0 3.0 3.0 etc for it=3.0, same value across each column)
    cleaned_prior_mergers_file = prior_mergers_file
    
    radius_list = []
    masses_list = []
    spins_list = []
    spin_angles_list = []
    gens_list = []
    len_columns = prior_mergers_file.shape[1]
    rows_to_be_removed = []

    for i in range(0,len_columns):
        # If 1st and 2nd entries in row i are same, it's an iteration marker, delete row.
        if prior_mergers_file[0,i] == prior_mergers_file[1,i]:
            rows_to_be_removed = np.append(rows_to_be_removed,int(i))
            
    rows_to_be_removed=rows_to_be_removed.astype('int32')
    cleaned_prior_mergers_file = np.delete(cleaned_prior_mergers_file,rows_to_be_removed,axis=1)
    
    radius_list = cleaned_prior_mergers_file[0,:]
    masses_list = cleaned_prior_mergers_file[1,:]
    spins_list = cleaned_prior_mergers_file[2,:]
    spin_angles_list = cleaned_prior_mergers_file[3,:]
    gens_list = cleaned_prior_mergers_file[4,:]

    return radius_list,masses_list,spins_list,spin_angles_list,gens_list
