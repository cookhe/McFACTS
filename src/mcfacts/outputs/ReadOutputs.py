"""Define output handling functions for McFACTS

Logfile
-------
    "bin_num_max"                   : int
        Maximum allowable number of binaries at a given time. Sets array size.
    "fname_ini"                     : str
        Name of the ini file used for this run.
    "fname_output_mergers"          : str
        Name of the output file for mergers.
    "fname_output"                  : str
        Name of the output file for the run.
    "fname_snapshots_bh"            : str
        Name of the output file recording BH info at each snapshot.
    "saves_snapshots"               : bool
        True: save snapshots, False: don't save snapshots
    "verbose"                       : bool
        True: print extra info, False: don't print extra info
    "work_directory"                : str
        The working directory used for the run.
    "seed"                          : int
        Random seed used for the run.
    "fname_log"                     : str
        Name of the log file storing details of the run.
    "runtime_directory"             : str
        The runtime directory used for the run.
    "disk_model_name"               : str
        'sirko_goodman' or 'thompson_etal'
    "flag_use_pagn"                 : bool
        Use pAGN to generate disk model?
    "flag_add_stars"                : bool
        Add stars to the disk
    "flag_initial_stars_BH_immortal": float
        If stars over disk_star_initial_mass_cutoff turn into BH (0) or hold at cutoff (1, immortal)
    "smbh_mass"                     : float
        Mass of the supermassive black hole (solMass)
    "disk_radius_trap"              : float
        Radius of migration trap in gravitational radii (r_g = G*`smbh_mass`/c^2)
        Should be set to zero if disk model has no trap
    "disk_radius_outer"             : float
        final element of disk_model_radius_array (units of r_g)
    "disk_radius_max_pc"            : float
        Maximum disk size in parsecs (0. for off)
    "disk_alpha_viscosity"          : float
        disk viscosity 'alpha'
    "nsc_radius_outer"              : float
        Radius of NSC (units of pc)
    "nsc_mass"                      : float
        Mass of NSC (units of M_sun)
    "nsc_radius_crit"               : float
        Radius where NSC density profile flattens (transition to Bahcall-Wolf) (units of pc)
    "nsc_ratio_bh_num_star_num"     : float
        Ratio of number of BH to stars in NSC (typically spans 3x10^-4 to 10^-2 in Generozov+18)
    "nsc_ratio_bh_mass_star_mass"   : float
        Ratio of mass of typical BH to typical star in NSC (typically 10:1 in Generozov+18)
    "nsc_density_index_inner"       : float
        Index of radial density profile of NSC inside r_nsc_crit (usually Bahcall-Wolf, 1.75)
    "nsc_density_index_outer"       : float
        Index of radial density profile of NSC outside r_nsc_crit
        (e.g. 2.5 in Generozov+18 or 2.25 if Peebles)
    "disk_aspect_ratio_avg"    : float
        Average disk scale height (e.g. about 3% in Sirko & Goodman 2003 out to ~0.3pc)
    "nsc_spheroid_normalization"    : float
        Spheroid normalization
    "nsc_imf_bh_mode"               : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--mode of initial mass dist (M_sun)
    "nsc_imf_bh_powerlaw_index"     : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--powerlaw index for Pareto dist
    "nsc_imf_bh_mass_max"           : float
        Initial mass distribution for stellar bh is assumed to be Pareto
        with high mass cutoff--mass of cutoff (M_sun)
    "nsc_bh_spin_dist_mu"           : float
        Initial spin distribution for stellar bh is assumed to be Gaussian
        --mean of spin dist
    "nsc_bh_spin_dist_sigma"        : float
        Initial spin distribution for stellar bh is assumed to be Gaussian
        --standard deviation of spin dist
    "disk_bh_torque_condition"      : float
        fraction of initial mass required to be accreted before BH spin is torqued
        fully into alignment with the AGN disk. We don't know for sure but
        Bogdanovic et al. says between 0.01=1% and 0.1=10% is what is required.
    "disk_bh_eddington_ratio"       : float
        Eddington ratio for disk bh
    "disk_bh_orb_ecc_max_init"      : float
        assumed accretion rate onto stellar bh from disk gas, in units of Eddington
        accretion rate
    "disk_star_mass_max_init"       : float
        Initial mass distribution for stars is assumed Salpeter
    "disk_star_mass_min_init"       : float
        Initial mass distribution for stars is assumed Salpeter
    "nsc_imf_star_powerlaw_index"   : float
        Initial mass distribution for stars is assumed Salpeter, disk_alpha_viscosity = 2.35
    "disk_star_scale_factor"        : float
        Scale factor to go from number of BH to number of stars.
    "disk_star_initial_mass_cutoff" : float
        Cutoff for initial star behavior
    "nsc_imf_star_mass_mode"       : float
        Mass mode for star IMF
    "disk_star_torque_condition"    : float
        fraction of initial mass required to be accreted before star spin is torqued
        fully into alignment with the AGN disk. We don't know for sure but
        Bogdanovic et al. says between 0.01=1% and 0.1=10% is what is required.
    "disk_star_eddington_ratio"     : float
        assumed accretion rate onto stars from disk gas, in units of Eddington
        accretion rate
    "disk_star_orb_ecc_max_init"    : float
        assuming initially flat eccentricity distribution among single orbiters around SMBH
        out to max_initial_eccentricity. Eventually this will become smarter.
    "nsc_star_metallicity_x_init"   : float
        Stellar initial hydrogen mass fraction
    "nsc_star_metallicity_y_init"   : float
        Stellar initial helium mass fraction
    "nsc_star_metallicity_z_init"   : float
        Stellar initial metallicity mass fraction
    "timestep_duration_yr"          : float
        How long is your timestep in years?
    "timestep_num"                  : int
        How many timesteps are you taking (timestep*number_of_timesteps = disk_lifetime)
    "galaxy_num"                    : int
        Number of galaxies of code run (e.g. 1 for testing, 30 for a quick run)
    "fraction_bin_retro"            : float
        Fraction of BBH that form retrograde to test (q,X_eff) relation. Default retro=0.1
    "flag_thermal_feedback"         : int
        Switch (1) turns feedback from embedded BH on.
    "flag_orb_ecc_damping"          : int
        Switch (1) turns orb. ecc damping on.
        If switch = 0, assumes all bh are circularized (at e=e_crit)
    "capture_time_yr"              : float
        Capture time in years Secunda et al. (2021) assume capture rate 1/0.1 Myr
    "disk_radius_capture_outer"     : float
        Disk capture outer radius (units of r_g)
        Secunda et al. (2001) assume <2000r_g from Fabj et al. (2020)
    "disk_bh_pro_orb_ecc_crit"      : float
        Critical eccentricity (limiting eccentricity, below which assumed circular orbit)
    "flag_dynamic_enc"              : int
        Switch (1) turns dynamical encounters between embedded BH on.
    "delta_energy_strong"           : float
        Average energy change per strong interaction.
        de can be 20% in cluster interactions. May be 10% on average (with gas)
    "inner_disk_outer_radius"       : float
        Outer radius of the inner disk (Rg)
    "disk_inner_stable_circ_orb"    : float
        Innermost Stable Circular Orbit around SMBH
    "mass_pile_up"                  : float
        Pile-up of masses caused by cutoff (M_sun)

"""

# Things everyone needs
import configparser as ConfigParser
from io import StringIO
from importlib import resources as impresources
# Third party
import numpy as np
import scipy.interpolate
# pAGN imports 
import pagn.constants as pagn_ct
# Local imports 
import mcfacts.external.DiskModelsPAGN as dm_pagn
from mcfacts.inputs import data as mcfacts_input_data
from astropy import constants as ct

# Dictionary of types
OUTPUT_TYPES = {
    "bin_num_max"                   : int,
    "fname_ini"                     : str,
    "fname_output_mergers"          : str,
    "fname_output"                  : str,
    "fname_snapshots_bh"            : str,
    "saves_snapshots"               : bool,
    "verbose"                       : bool,
    "work_directory"                : str,
    "seed"                          : int,
    "fname_log"                     : str,
    "runtime_directory"             : str,
    "disk_model_name"               : str,
    "flag_use_pagn"                 : bool,
    "flag_add_stars"                : bool,
    "flag_initial_stars_BH_immortal": bool,
    "smbh_mass"                     : float,
    "disk_radius_trap"              : float,
    "disk_radius_outer"             : float,
    "disk_radius_max_pc"            : float,
    "disk_alpha_viscosity"          : float,
    "nsc_radius_outer"              : float,
    "nsc_mass"                      : float,
    "nsc_radius_crit"               : float,
    "nsc_ratio_bh_num_star_num"     : float,
    "nsc_ratio_bh_mass_star_mass"   : float,
    "nsc_density_index_inner"       : float,
    "nsc_density_index_outer"       : float,
    "disk_aspect_ratio_avg"         : float,
    "nsc_spheroid_normalization"    : float,
    "nsc_imf_bh_mode"               : float,
    "nsc_imf_bh_powerlaw_index"     : float,
    "nsc_imf_bh_mass_max"           : float,
    "nsc_bh_spin_dist_mu"           : float,
    "nsc_bh_spin_dist_sigma"        : float,
    "disk_bh_torque_condition"      : float,
    "disk_bh_eddington_ratio"       : float,
    "disk_bh_orb_ecc_max_init"      : float,
    "disk_star_mass_max_init"       : float,
    "disk_star_mass_min_init"       : float,
    "nsc_imf_star_powerlaw_index"   : float,
    "disk_star_scale_factor"        : float,
    "disk_star_initial_mass_cutoff" : float,
    "nsc_imf_star_mass_mode"        : float,
    "disk_star_torque_condition"    : float,
    "disk_star_eddington_ratio"     : float,
    "disk_star_orb_ecc_max_init"    : float,
    "nsc_star_metallicity_x_init"   : float,
    "nsc_star_metallicity_y_init"   : float,
    "nsc_star_metallicity_z_init"   : float,
    "timestep_duration_yr"          : float,
    "timestep_num"                  : int,
    "galaxy_num"                    : int,
    "fraction_bin_retro"            : float,
    "flag_thermal_feedback"         : int,
    "flag_orb_ecc_damping"          : int,
    "capture_time_yr"               : float,
    "disk_radius_capture_outer"     : float,
    "disk_bh_pro_orb_ecc_crit"      : float,
    "flag_dynamic_enc"              : int,
    "delta_energy_strong"           : float,
    "inner_disk_outer_radius"       : float,
    "disk_inner_stable_circ_orb"    : float,
    "mass_pile_up"                  : float,
    "save_snapshots"                : bool,
    "mean_harden_energy_delta"      : float,
    "var_harden_energy_delta"       : float
}

def ReadLog(fname_log, verbose=False):
    """Output log parser

    Parameters
    ----------
    fname_log : str
        Path and name to the log file from a McFACTS run
    verbose : bool, optional
        Print extra info, by default False
    """

    # Read in the file
    with open(fname_log, 'r') as f:
        lines = f.readlines()

    # Initialize the dictionary
    log_dict = {}
    extra_values = {}

    # Loop through the lines
    for line in lines:
        # Split the line
        key, value = line.split('=')
        # Strip the values of any whitespace
        key = key.strip()
        value = value.strip()
        # Convert the values
        if key in OUTPUT_TYPES:
            log_dict[key] = OUTPUT_TYPES[key](value)
        else:
            log_dict[key] = value
            extra_values[key] = value
        
    # Print the dictionary
    if verbose:
        for key, value in log_dict.items():
            print(f"{key}: {value}")

    if len(extra_values) > 0:
        print(f"~~~~~~~~~~~~~~~~~~~~~~\n",
               "[ReadLog] Warning!: The log file you're using contains additional\n",
               "entries not found in OUTPUT_TYPES. They have been added to the log\n",
               "dictionary as a STRING type. Please verify their types before you\n",
               "use them in your analysis or add them to the OUTPUT_TYPES dictionary.")
        for key, value in extra_values.items():
            print(f"  {key}: {value}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~")
        

    return log_dict
    