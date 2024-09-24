import numpy as np

def binary_check(
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        smbh_mass,
        disk_bh_pro_orbs_ecc,
        disk_bh_pro_orb_ecc_crit
        ):
    """Which prograde BH will form binaries in this timestep.
    
    Takes as inputs the singleton BH locations,masses & orbital eccentricities,
    and takes the candidate binary population from BH with orbital eccentricities
    damped to < orb_ecc_crit. Among this damped population, checks if their 
    separations are less than the mutual Hill sphere of any 2 adjacent BH. If this 
    is the case, determine the smallest separation pairs (in units of their mutual 
    Hill sphere) to form a set of actual binaries (this module does handle cases where 
    3 or more bodies *might* form some set of binaries which would be mutually exclusive; 
    however it does not handle or even flag the implied triple system dynamics). 
    Returns a 2xN array of the relevant binary indices, for further handling to form actual 
    binaries & assign additional parameters (e.g. angular momentum of the binary).

    Parameters
    ----------
    disk_bh_pro_orbs_a : float array
        locations of prograde singleton BH at start of timestep in units of 
        gravitational radii (r_g=GM_SMBH/c^2)
    disk_bh_pro_masses : float array
        initial masses of bh in prograde orbits around SMBH in units of solar 
        masses
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    disk_bh_pro_orbs_ecc : float array
        Orbital ecc of singleton BH after damping during timestep
    disk_bh_pro_orb_ecc_crit : float
        Critical eccentricity allowing bin formation and migration
      
    Returns
    -------
    disk_bin_bhbh_pro_indices : [2,N] int array
        array of indices corresponding to locations in disk_bh_pro_orbs_a, 
        disk_bh_pro_masses, etc. which corresponds to binaries that form in 
        this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.

    Notes
    -----
    Internal variable names not standardized. Fix later.
    """

    # First check for BH with sufficiently damped orbital eccentricity
    # (orb_ecc<=orb_ecc_crit (usually 0.01)).
    # This population is the sub-set of prograde BH from which we CAN form
    # binaries.
    
    # Singleton BH with orb ecc < e_crit (candidates for binary formation)
    prograde_bh_can_form_bins = np.ma.masked_where(disk_bh_pro_orbs_ecc > disk_bh_pro_orb_ecc_crit, disk_bh_pro_orbs_ecc)    
    indices_bh_can_form_bins = np.ma.nonzero(prograde_bh_can_form_bins)
    # Indices of those candidates for binary formation
    allowed_to_form_bins = np.array(indices_bh_can_form_bins[0])
    # Sort the location of the candidates
    sorted_bh_locations = np.sort(disk_bh_pro_orbs_a[allowed_to_form_bins])
    # Sort the indices of all singleton BH (the superset)
    sorted_bh_location_indices_superset = np.argsort(disk_bh_pro_orbs_a)
    # Set the condition for membership in candidate array to be searched/tested
    condition = np.isin(sorted_bh_location_indices_superset, allowed_to_form_bins)
    # Here is the subset of indices that can be tested for binarity
    subset = np.extract(condition, sorted_bh_location_indices_superset)
    
    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    # Note length of separations is 1 less than disk_bh_pro_orbs_a
    # This is the set of separations between the sorted candidate BH
    separations = np.diff(sorted_bh_locations)
 
    # Now compute mutual hill spheres of all possible candidate binaries if can test    
    if len(separations) > 0:
        R_Hill_possible_binaries = (sorted_bh_locations[:-1] + separations/2.0) * \
            pow(((disk_bh_pro_masses[subset[:-1]] + \
                  disk_bh_pro_masses[subset[1:]]) / \
                    (smbh_mass * 3.0)), (1.0/3.0))
        # compare separations to mutual Hill spheres - negative values mean possible binary formation
        minimum_formation_criteria = separations - R_Hill_possible_binaries
        
        # collect indices of possible real binaries (where separation is less than mutual Hill sphere)
        index_formation_criteria = np.where(minimum_formation_criteria < 0)
        
        #Here's the index of the array of candidates
        test_idx = index_formation_criteria[0]
        
        #If we actually have any candidates this time step
        if np.size(test_idx) > 0:
            # Start with real index (from full singleton array) of 1st candidate binary component (implicit + 1 partner since separations are ordered )
            bin_indices = np.array([subset[test_idx[0]],subset[test_idx[0]+1]])
            # If only 1 binary this timestep, return this binary!
            disk_bin_bhbh_pro_indices = np.array([subset[test_idx],subset[test_idx+1]])
            
            for i in range(len(test_idx)):
                #If more than 1 binary
                if i >0:
                    # append nth binary indices formed this timestep
                    bin_indices = np.append(bin_indices,[subset[test_idx[i]],subset[test_idx[i]+1]])
                    
                    #Check to see if repeat binaries among the set of binaries formed (e.g. (1,2)(2,3) )
                    #If repeats, only form a binary from the pair with smallest fractional Hill sphere separation

                    # Compute separation/R_Hill for all
                    sequences_to_test = (separations[test_idx])/(R_Hill_possible_binaries[test_idx])
                    # sort sep/R_Hill for all 'binaries' that need checking & store indices
                    sorted_sequences = np.sort(sequences_to_test)
                    #Sort the indices for the test
                    sorted_sequences_indices = np.argsort(sequences_to_test)

                    # Assume the smallest sep/R_Hill should form a binary, so
                    if len(sorted_sequences) > 0:
                        #Index of smallest sorted fractional Hill radius binary so far
                        checked_binary_index = np.array([test_idx[sorted_sequences_indices[0]]])
                    else:
                        checked_binary_index = []    
                    for j in range(len(sorted_sequences)): 
                        # if we haven't already counted it
                        if (test_idx[sorted_sequences_indices[j]] not in checked_binary_index):
                            # and it isn't the implicit partner of something we've already counted
                            if (test_idx[sorted_sequences_indices[j]] not in checked_binary_index+1):
                                # and the implicit partner of this thing isn't already counted
                                if (test_idx[sorted_sequences_indices[j]]+1 not in checked_binary_index):
                                    # and the implicit partner of this thing isn't already an implicit partner we've counted
                                    if (test_idx[sorted_sequences_indices[j]]+1 not in checked_binary_index+1):
                                        # then you can count it as a real binary
                                        checked_binary_index = np.append(checked_binary_index, test_idx[sorted_sequences_indices[j]])
                    disk_bin_bhbh_pro_indices = np.array([subset[checked_binary_index],subset[checked_binary_index+1]])

        else:
            # No binaries from candidates this time step
            disk_bin_bhbh_pro_indices = []
        
    else:
        # No candidate for binarity testing yet
        disk_bin_bhbh_pro_indices = []
    
    return disk_bin_bhbh_pro_indices
