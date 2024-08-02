import numpy as np


def setup_disk_stars_location(rng, n_stars, disk_outer_radius):
    #Return an array of BH locations distributed randomly uniformly in disk
    integer_nstars = int(n_stars)
    stars_initial_locations = disk_outer_radius*rng.random(integer_nstars)
    return stars_initial_locations


def setup_disk_stars_masses(rng,n_star,min_initial_star_mass,max_initial_star_mass,mstar_powerlaw_index):
    #Return an array of star initial masses for a given alpha and min/max mass with the Salpeter IMF.
    #Taken from here: https://python4mpia.github.io/fitting_data/MC-sampling-from-Salpeter.html

    #Convert limits from M to logM
    log_M_min = np.log(min_initial_star_mass)
    log_M_max = np.log(max_initial_star_mass)

    #Max likelihood is at M_min
    maxlik = np.power(min_initial_star_mass, 1.0 - mstar_powerlaw_index)

    #Make array for output
    masses = []
    while(len(masses) < n_star):
        #draw candidate from logM interval
        logM = rng.uniform(low=log_M_min, high=log_M_max)
        M = np.exp(logM)
        #Compute likelihood of candidate from Salpeter SMF
        likelihood = np.power(M,1.0-mstar_powerlaw_index)
        #Accept randomly
        u = rng.uniform(low=0.0, high=maxlik)
        if(u < likelihood):
            masses.append(M)
    masses = np.array(masses)
    return(masses)

def setup_disk_stars_radii(masses):
    #For now the radius is just set to the ZAMS radius via the mass-radius relation and does not change
    radii = np.power(masses,0.8)
    return(radii)

def setup_disk_stars_comp(n_star,star_ZAMS_metallicity, star_ZAMS_hydrogen, star_ZAMS_helium):
    #For now the numbers are set in the input file. Maybe we want to just set 2 of them (X and Y?) and calculate the final so it adds to 1?
    star_Z = np.full(n_star,star_ZAMS_metallicity)
    star_X = np.full(n_star,star_ZAMS_hydrogen)
    star_Y = np.full(n_star,star_ZAMS_helium)
    return(star_X, star_Y, star_Z)


def setup_disk_stars_spins(rng, n_stars, mu_star_spin_distribution, sigma_star_spin_distribution):
    #Return an array of BH initial spin magnitudes for a given mode and sigma of a distribution
    integer_nstars = int(n_stars)
    stars_initial_spins = rng.normal(mu_star_spin_distribution, sigma_star_spin_distribution, integer_nstars)
    return stars_initial_spins


def setup_disk_stars_spin_angles(rng, n_stars, stars_initial_spins):
    #Return an array of BH initial spin angles (in radians).
    #Positive (negative) spin magnitudes have spin angles [0,1.57]([1.5701,3.14])rads
    #All BH spin angles drawn from [0,1.57]rads and +1.57rads to negative spin indices
    integer_nstars = int(n_stars)
    stars_initial_spin_indices = np.array(stars_initial_spins)
    negative_spin_indices = np.where(stars_initial_spin_indices < 0.)
    stars_initial_spin_angles = rng.uniform(0.,1.57,integer_nstars)
    stars_initial_spin_angles[negative_spin_indices] = stars_initial_spin_angles[negative_spin_indices] + 1.57
    return stars_initial_spin_angles


def setup_disk_stars_orb_ang_mom(rng, n_stars):
    #Return an array of BH initial orbital angular momentum.
    #Assume either fully prograde (+1) or retrograde (-1)
    integer_nstars = int(n_stars)
    random_uniform_number = rng.random((integer_nstars,))
    stars_initial_orb_ang_mom = (2.0*np.around(random_uniform_number)) - 1.0
    return stars_initial_orb_ang_mom

def setup_disk_stars_eccentricity_thermal(rng, n_stars):
    # Return an array of BH orbital eccentricities
    # For a thermal initial distribution of eccentricities, select from a uniform distribution in e^2.
    # Thus (e=0.7)^2 is 0.49 (half the eccentricities are <0.7). 
    # And (e=0.9)^2=0.81 (about 1/5th eccentricities are >0.9)
    # So rnd= draw from a uniform [0,1] distribution, allows ecc=sqrt(rnd) for thermal distribution.
    # Thermal distribution in limit of equipartition of energy after multiple dynamical encounters
    integer_nstars = int(n_stars)
    random_uniform_number = rng.random((integer_nstars,))
    stars_initial_orb_ecc = np.sqrt(random_uniform_number)
    return stars_initial_orb_ecc

def setup_disk_stars_eccentricity_uniform(rng, n_stars):
    # Return an array of BH orbital eccentricities
    # For a uniform initial distribution of eccentricities, select from a uniform distribution in e.
    # Thus half the eccentricities are <0.5
    # And about 1/10th eccentricities are >0.9
    # So rnd = draw from a uniform [0,1] distribution, allows ecc = rnd for uniform distribution
    # Most real clusters/binaries lie between thermal & uniform (e.g. Geller et al. 2019, ApJ, 872, 165)
    integer_nstars = int(n_stars)
    random_uniform_number = rng.random((integer_nstars,))
    stars_initial_orb_ecc = random_uniform_number
    return stars_initial_orb_ecc

def setup_disk_stars_inclination(rng, n_stars):
    # Return an array of BH orbital inclinations
    # Return an initial distribution of inclination angles that are 0.0
    #
    # To do: initialize inclinations so random draw with i <h (so will need to input bh_locations and disk_aspect_ratio)
    # and then damp inclination.
    # To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    # Then damp (i,e) as appropriate
    integer_nstars = int(n_stars)
    # For now, inclinations are zeros
    stars_initial_orb_incl = np.zeros((integer_nstars,),dtype = float)
    return stars_initial_orb_incl

def setup_disk_stars_circularized(rng, n_stars,crit_ecc):
    # Return an array of BH orbital inclinations
    # Return an initial distribution of inclination angles that are 0.0
    #
    # To do: initialize inclinations so random draw with i <h (so will need to input bh_locations and disk_aspect_ratio)
    # and then damp inclination.
    # To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    # Then damp (i,e) as appropriate
    integer_nstars = int(n_stars)
    # For now, inclinations are zeros
    #bh_initial_orb_ecc = crit_ecc*np.ones((integer_nbh,),dtype = float)
    #Try zero eccentricities
    stars_initial_orb_ecc = crit_ecc*np.zeros((integer_nstars,),dtype = float)
    return stars_initial_orb_ecc

def setup_disk_nstars(M_nsc,nbh_nstar_ratio,mbh_mstar_ratio,r_nsc_out,nsc_index_outer,mass_smbh,disk_outer_radius,h_disk_average,r_nsc_crit,nsc_index_inner):
    # Return the integer number of stars in the AGN disk as calculated from NSC inputs assuming isotropic distribution of NSC orbits
    # To do: Calculate when R_disk_outer is not equal to the r_nsc_crit
    # To do: Calculate when disky NSC population of BH in plane/out of plane.
    # Housekeeping:
    # Convert outer disk radius in r_g to units of pc. 1r_g =1AU (M_smbh/10^8Msun) and 1pc =2e5AU =2e5 r_g(M/10^8Msun)^-1
    pc_dist = 2.e5*((mass_smbh/1.e8)**(-1.0))
    critical_disk_radius_pc = disk_outer_radius/pc_dist
    #Total average mass of stars in NSC
    M_stars_nsc = M_nsc * (1./nbh_nstar_ratio) * (1./mbh_mstar_ratio)
    #print("M_bh_nsc",M_bh_nsc)
    #Total number of stars in NSC
    N_stars_nsc = M_stars_nsc * mbh_mstar_ratio
    #print("N_bh_nsc",N_bh_nsc)
    #Relative volumes:
    #   of central 1 pc^3 to size of NSC
    relative_volumes_at1pc = (1.0/r_nsc_out)**(3.0)
    #   of r_nsc_crit^3 to size of NSC
    relative_volumes_at_r_nsc_crit = (r_nsc_crit/r_nsc_out)**(3.0)
    #print(relative_volumes_at1pc)
    #Total number of stars 
    #   at R<1pc (should be about 10^4 for Milky Way parameters; 3x10^7Msun, 5pc, r^-5/2 in outskirts)
    N_stars_nsc_pc = N_stars_nsc * relative_volumes_at1pc * (1.0/r_nsc_out)**(-nsc_index_outer)
    #   at r_nsc_crit
    N_stars_nsc_crit = N_stars_nsc * relative_volumes_at_r_nsc_crit * (r_nsc_crit/r_nsc_out)**(-nsc_index_outer)
    #print("Normalized N_bh at 1pc",N_bh_nsc_pc)
    
    #Calculate Total number of stars in volume R < disk_outer_radius, assuming disk_outer_radius<=1pc.
    
    if critical_disk_radius_pc >= r_nsc_crit:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/1.0)**(3.0)
        Nstars_disk_volume = N_stars_nsc_pc * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/1.0)**(-nsc_index_outer))          
    else:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/r_nsc_crit)**(3.0)
        Nstars_disk_volume = N_stars_nsc_crit * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/r_nsc_crit)**(-nsc_index_inner))
     
    # Total number of BH in disk
    Nstars_disk_total = np.rint(Nstars_disk_volume * h_disk_average)
    #print("Nbh_disk_total",Nbh_disk_total)  
    return np.int64(Nstars_disk_total)

