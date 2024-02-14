import sys

import numpy as np
import astropy.constants as const

G = const.G.cgs.value # gravitational constant


def calculate_hill_sphere(prograde_bh_locations, prograde_bh_masses, mass_smbh):
    #Return the Hill sphere radius (R_Hill) for an array of prograde BH where
    # R_Hill=a(q/3)^1/3 where a=semi-major axis, q=m_bh/M_SMBH
    
    bh_smbh_mass_ratio = prograde_bh_masses/(3.0*mass_smbh)
    mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
    bh_hill_sphere = prograde_bh_locations*mass_ratio_factor
    #Return the BH Hill sphere radii for all orbiters. Prograde should have much larger Hill sphere
    return bh_hill_sphere

def encounter_test(prograde_bh_locations, bh_hill_sphere):
    #Using Hill sphere size and BH locations see if there are encounters within the Hill sphere
    # return indices of BH involved.

    # First sort the prograde bh locations in order from inner disk to outer disk
    sorted_bh_locations = np.sort(prograde_bh_locations)
    #Returns the indices of the original array in order, to get the sorted array
    sorted_bh_location_indices = np.argsort(prograde_bh_locations)
   
    #Find the appropriate (sorted) Hill sphere radii
    sorted_hill_spheres = bh_hill_sphere[sorted_bh_location_indices]

    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    separations = np.diff(sorted_bh_locations)

    # Note that separations are -1 of length of bh_locations
    # Take 1st location off locations array
    sorted_bh_locations_minus_first_element = sorted_bh_locations[1:len(sorted_bh_locations)]
    #Take last location off locations array
    sorted_bh_locations_minus_last_element = sorted_bh_locations[0:len(sorted_bh_locations)-1]

    # Separations are -1 of length of Hill_sphere array
    # Take 1st Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_first = sorted_hill_spheres[1:len(sorted_hill_spheres)]
    # Take last Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_last = sorted_hill_spheres[0:len(sorted_hill_spheres)-1]

    # Compare the Hill sphere distance for each BH with separation to neighbor BH
    #so we compare e.g. r2-r1 vs R_H1, r3-r2 vs R_H2
    comparison_distance_inwards = separations-sorted_hill_spheres_minus_last
    # and e.g. compare r2-r1 vs R_H2, r3-r2 vs R_H3
    comparison_distance_outwards = separations-sorted_hill_spheres_minus_first

    index_in = np.where(comparison_distance_inwards < 0)
    if isinstance(index_in,tuple):
        index_in = index_in[0]
    # E.g say r3-r2 <R_H2 then we'll want the info for the BH at r2 & r3. (i,i+1)
    index_out = np.where(comparison_distance_outwards < 0)
    if isinstance(index_out,tuple):
        index_out = index_out[0]
    #E.g. say r3-r2 <R_H3 then we'll want the info for the BH at r3 and r2 (i,i-1)
    length_index_in = len(index_in)
    length_index_out = len(index_out)

    new_indx_in = list(range(2*len(index_in)))
    new_indx_out = list(range(2*len(index_out)))
    #new_indx_in_bin_array = np.array[2*len(index_in),2]
    #temp_new_bin_array = np.ndarray[2*length_index_in,2]

    for ind in range(length_index_in):
        temp_index = index_in[ind]
        new_indx_in[2*ind] = temp_index
        new_indx_in[(2*ind)+1] = temp_index+1
    #    temp_new_bin_array = 
    #print("new_indx_in",new_indx_in)
    temp_new_bin_in_array=np.reshape(new_indx_in,(len(index_in),2))
    #print("ordered as bins",temp_new_bin_in_array)
    
    test_remove_bin = new_indx_in
    
    # Dynamics Here! Potential Double-binary or triple interaction!
    # For now! 
    # 0. Construct array of pairs of indices.
    # 1. Select binaries based on distance
    #     For binaries [i-1,i], [i,i+1] compare distance r(i)-r(i-1) to r(i+1)-r(i) and select smallest.
    #     Remove larger distane pair. E.g. if [i+1,i] is smaller, remove [i-1,i]
    #    But want to calculate 
    # 2. Fractional R_Hill for [i-1,i] vs [i,i+1]. Smaller fractional Hill radius wins
    # 3. TO DO Write a module for DiLaurentii+22 or Rowan+22 or LANL+22 phase space encounter and apply to all encounters 
    #          over timestep (10kyrs; assume random phase & number of encounters during timestep; pick randomly 
    #          from phase plots.) Also look at LANL group papers on binding energy of encounter.
    # 4. Ideally, consider the triple-dynamics encounter 
    
    # Search for repeat indices in binary array 
    unique_element,unique_index,unique_ct = np.unique(temp_new_bin_in_array,return_inverse = True,return_counts = True)
    repeats = unique_ct > 1
    
    repeat_indices = unique_element[repeats]
    
    duplicate_indices,dup_index,dup_ct = np.unique(unique_index, return_inverse = True, return_counts = True)
    rep_idx = dup_ct >1
    dupl_indices = duplicate_indices[rep_idx]
    #print("indices of repeats",dupl_indices)
    

    #return all those elements that only occur once
    new_array = unique_element[unique_ct == 1]
    #print("new array",new_array)
    #return the index where the element occurs first e.g. if element 59 occurs at index 9,10 in the array and
    #element 60 occurs at index 11,12, then index returned is =[...8,9,11,13,...]
    unique_values,indices_list = np.unique(temp_new_bin_in_array, return_index=True)
    #unique_values,repeat_indices_list = np.unique(temp_new_bin_in_array, )
    #print("indices list",indices_list)   

    
    #print("repeat_indices",repeat_indices)
    #test_remove_bin=np.delete(test_remove_bin,repeat_indices)
    #print("test remove bin",test_remove_bin)

    
    #Outer bin (i,i+1) here
    temp_dist_outer_bin_in = comparison_distance_inwards[repeat_indices]
    temp_dist_outer_bin_out = comparison_distance_outwards[repeat_indices]
    smallest_sep_outer_bin = np.fmin(temp_dist_outer_bin_in,temp_dist_outer_bin_out)
    #print("outer bin",smallest_sep_outer_bin)
    #Inner bin (i-1,i) here
    temp_dist_in_minus1 = comparison_distance_inwards[repeat_indices-1]
    temp_dist_out_minus1 = comparison_distance_outwards[repeat_indices-1]
    temp_dist_inner_bin = np.fmin(temp_dist_in_minus1,temp_dist_out_minus1)
    #print("inner bin",temp_dist_inner_bin)
    #print("repeat_indices",repeat_indices)
    #These are the indices of binaries to be deleted. 
    #element_to_be_kept = np.where(smallest_sep_outer_bin < temp_dist_inner_bin, repeat_indices, repeat_indices + 1)
    #element_to_be_deleted = np.where(smallest_sep_outer_bin > temp_dist_inner_bin,repeat_indices,repeat_indices + 1)

    new_test_remove_bin = test_remove_bin
    #print("test_remove_bin",new_test_remove_bin)
    #set up array of indices to be removed
    idx_array=np.zeros(len(2*dupl_indices))

    for ind in range(len(dupl_indices)):
        idx = dupl_indices[ind] - (ind)
        value=2*ind
        # If outer bin to be kept    
        if smallest_sep_outer_bin[ind] < temp_dist_inner_bin[ind]:
            #delete inner bin 
            remove_indxs=[idx - 1,idx]
            
            #idx_array[value]=idx - 1
            #idx_array[value+1]=idx
            new_test_remove_bin = np.delete(new_test_remove_bin,remove_indxs)
            #print("test remove",new_test_remove_bin)
        else:
            #delete outer bin
            remove_indxs=[idx+1,idx+2]

            #idx_array[value]=idx+1
            #idx_array[value]=idx+2

            new_test_remove_bin = np.delete(new_test_remove_bin,remove_indxs)
            #print("test remove",new_test_remove_bin)
    
    #print("idx_array",idx_array)

    # Say [10,11,15] Not 9-10;so 10-11 then 13:14.  
    # New=[0:9,11:14,15:end]
    #while ind in range(2*dupl_indices):
    #    new_array=test_remove_bin[1:ind(i),ind(i+2):ind(i+3)-1,ind(i)]
    
    #CHANGE THIS going forward: For removing repeats. Compute mutual Hill sphere (R_H=((M1+M_2)/M_SMBH)^{1/3}) and
    # see how small a fraction of the mutual Hill sphere, the binary separation *would* be.
    #E.g. (r2-R1)=0.5R_H(m1,m2) vs (r3-r2)=0.2R_H(m3,m2) form the tighter binary (r3,r2)

    #Compare nearest neighbour separations for BH in repeat_indices
    # E.g. separations =[r2-r1,r3-r2,r3-4]. If BH at r2 repeats in binary array 
    # e.g. bin_array=[[r1,r2] [r2,r3]..] then compare separations[r2-r1] to separations[r3-r2]
    # If r2-r1 < r3-r2 then make [r1,r2] the binary and remove [r2,r3]
    
    #if separations[repeat_indices] < separations[repeat_indices + 1]:
    #    temp_new_bin_in_array = np.delete(temp_new_bin_in_array,repeat_indices + 1)
    #    print("temp_bin_array",temp_new_bin_in_array)

#    for j in range(2*len(index_in)):    
#        temp_bin_in = new_indx_in_bin_array[temp_index[]]
   

#    for j in range(2*len(index_in)):    
#        temp_bin_in = new_indx_in_bin_array[temp_index[]]
   
    for ind in range(length_index_out):
        temp_index = index_out[ind]
        new_indx_out[2*ind] = temp_index
        new_indx_out[(2*ind)+1] = temp_index+1

    #print("new_indx_out",new_indx_out)    
    temp_new_bin_out_array=np.reshape(new_indx_out,(len(index_out),2))
    #print("ordered as bins",temp_new_bin_out_array)

    new_indxs = new_indx_in+new_indx_out
    #rindx = np.sort(new_indxs)
    result = np.asarray(new_test_remove_bin)
    #result = np.asarray(new_indx_in)
    sorted_in_result = np.sort(result)

    new_result = np.asarray(new_indx_out)
    sorted_out_result = np.sort(new_result)
    #print("sorted in result",sorted_in_result)
    #print("sorted out result",sorted_out_result)
    # Concatenate the two lists, and remove duplicates
    final_bin_indices = np.array(list(set(list(sorted_in_result) + list(sorted_out_result))))
    sorted_final_bin_indices = np.sort(final_bin_indices)
    #print("total final bin indices",sorted_final_bin_indices)
    #print("check if sorted_in & sorted_out arrays are the same")
    check = np.array_equiv(sorted_in_result, sorted_out_result)
    #print(check)
    # Return the indices of those elements in separation array <0
    # (ie BH are closer than 1 R_Hill)
    # In inwards case, r_i+1 -r_i <R_H_i, so relevant BH indices are i,i+1
   
    # In outwards case, r_i - r_i-1 <R_H_i so relevant BH indices are i,i-1
    final_1d_indx_array = sorted_in_result.flatten()
    sorted_final_1d_indx_array = np.sort(final_1d_indx_array)

    #if len(sorted_final_1d_indx_array) > 0:
    #     print("Binary", sorted_final_1d_indx_array)

    return sorted_final_1d_indx_array

def binary_check(prograde_bh_locations, prograde_bh_masses, mass_smbh, prograde_bh_orb_ecc, e_crit):
    """Determines which prograde BH will form binaries in this timestep. Takes as inputs
    the singleton BH locations & masses, and checks if their separations are less than
    the mutual Hill sphere of any 2 adjacent BH. If this is the case, determine the
    smallest separation pairs (in units of their mutual Hill sphere) to form a set of
    actual binaries (this module does handle cases where 3 or more bodies *might* form
    some set of binaries which would be mutually exclusive; however it does not handle
    or even flag the implied triple system dynamics). Returns a 2xN array of the relevant
    binary indices, for further handling to form actual binaries & assign additional
    parameters (e.g. angular momentum of the binary).

    Parameters
    ----------
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of 
        gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        initial masses of bh in prograde orbits around SMBH in units of solar masses
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_orb_ecc : float array
        Orbital ecc of singleton BH after damping during timestep
    e_crit : float
        Critical eccentricity allowing bin formation and migration    
    Returns
    -------
    all_binary_indices : [2,N] int array
        array of indices corresponding to locations in prograde_bh_locations, prograde_bh_masses,
        prograde_bh_spins, prograde_bh_spin_angles, and prograde_bh_generations which corresponds
        to binaries that form in this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.
    """

    #print('bh locations',prograde_bh_locations)
    # First sort the prograde bh locations in order from inner disk to outer disk
    sorted_bh_locations = np.sort(prograde_bh_locations)
    #print('sorted bh locations',sorted_bh_locations)
    # Returns the indices of the original array in order, to get the sorted array
    sorted_bh_location_indices = np.argsort(prograde_bh_locations)
    # Returns the indices of the orb ecc array, to get sorted array of orb ecc
    #print('sorted bh location indices',sorted_bh_location_indices)
    #print('bh orb ecc',prograde_bh_orb_ecc)
    sorted_bh_ecc_array = np.empty_like(prograde_bh_orb_ecc)
    sorted_bh_ecc_array = prograde_bh_orb_ecc[np.argsort(prograde_bh_locations)]
    #print('orb ecc of sorted bh', sorted_bh_ecc_array)
    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    # Note length of separations is 1 less than prograde_bh_locations
    separations = np.diff(sorted_bh_locations)
    # Now compute mutual hill spheres of all possible binaries
    # same length as separations
    R_Hill_possible_binaries = (sorted_bh_locations[:-1] + separations/2.0) * \
        pow(((prograde_bh_masses[sorted_bh_location_indices[:-1]] + \
              prograde_bh_masses[sorted_bh_location_indices[1:]]) / \
                (mass_smbh * 3.0)), (1.0/3.0))
    # compare separations to mutual Hill spheres - negative values mean possible binary formation
    minimum_formation_criteria = separations - R_Hill_possible_binaries
    # collect indices of possible real binaries (where separation is less than mutual Hill sphere)
    index_formation_criteria = np.where(minimum_formation_criteria < 0)
    # Now deal with sequences: compute separation/R_Hill for all
    sequences_to_test = (separations[index_formation_criteria])/(R_Hill_possible_binaries[index_formation_criteria])
    #print(sequences_to_test)
    # sort sep/R_Hill for all 'binaries' that need checking & store indices
    sorted_sequences = np.sort(sequences_to_test)
    #print(sorted_sequences)
    sorted_sequences_indices = np.argsort(sequences_to_test)
    #print(sorted_sequences_indices)
    # the smallest sep/R_Hill should always form a binary, so
    checked_binary_index = np.array([sorted_sequences_indices[0]])
    #print(checked_binary_index)
    for i in range(len(sorted_sequences)): 
        # if we haven't already counted it
        if (sorted_sequences_indices[i] not in checked_binary_index):
            # and it isn't the implicit partner of something we've already counted
            if (sorted_sequences_indices[i] not in checked_binary_index+1):
                # and the implicit partner of this thing isn't already counted
                if (sorted_sequences_indices[i]+1 not in checked_binary_index):
                    # and the implicit partner of this thing isn't already an implicit partner we've counted
                    if (sorted_sequences_indices[i]+1 not in checked_binary_index+1):
                        # then you can count it as a real binary
                        checked_binary_index = np.append(checked_binary_index, sorted_sequences_indices[i])

    #print("THIS IS SAAVIK'S OUTPUT!!")
    # create array of all real binaries
    # BUT what are we returning? need indices of original arrays
    # that go to singleton vs binary assignments--actual binary formation *should* happen elsewhere!
    # I have the indices of the sorted_bh_locations array that correspond to actual binaries
    # these are checked_binary_index, checked_binary_index+1
    all_binary_indices = np.array([sorted_bh_location_indices[checked_binary_index], sorted_bh_location_indices[checked_binary_index+1]])
    #print(np.shape(all_binary_indices))
    #print(np.shape(all_binary_indices)[1])
    
    #HERE is where we check that both BH have damped orbital eccentricity (e<=0.01). Otherwise do not form binary
    # TO DO: Make new bin formation condition at modest ecc & encounters. 
    # E.g. Remember in a timestep of 10^4yrs, there are 10^4 orbits at R=10^3r_g
    #Singleton BH with orb ecc > e_crit
    prograde_bh_not_form_bins = np.ma.masked_where(prograde_bh_orb_ecc <= e_crit, prograde_bh_orb_ecc)
    #Singleton BH with orb ecc < e_crit
    prograde_bh_can_form_bins = np.ma.masked_where(prograde_bh_orb_ecc >e_crit, prograde_bh_orb_ecc)
    #Indices of singleton BH with orb ecc > e_crit
    indices_bh_not_form_bins = np.ma.nonzero(prograde_bh_not_form_bins) 
    #Indices of singleton BH with orb ecc < e_crit
    indices_bh_can_form_bins = np.ma.nonzero(prograde_bh_can_form_bins)
    

    #print('ORIGINAL INDICES BH ALLOWED FORM BINS',np.array(indices_bh_can_form_bins[0]))
    allowed_to_form_bins = np.array(indices_bh_can_form_bins[0])
    #print('allowed bh locs',prograde_bh_locations[allowed_to_form_bins])
    sorted_allowed_bh_loc = np.sort(prograde_bh_locations[allowed_to_form_bins])
    #print('sorted allowed bh locs',sorted_allowed_bh_loc)
    #print('allowed bh eccs',prograde_bh_orb_ecc[allowed_to_form_bins])
    #print('allowed bh masses',prograde_bh_masses[allowed_to_form_bins])
    allowed_separations = np.diff(sorted_allowed_bh_loc)
    # Now compute mutual hill spheres of all possible binaries
    # same length as separations
    R_Hill_allowed_bin_test = (sorted_allowed_bh_loc[:-1] + allowed_separations/2.0) * \
        pow(((prograde_bh_masses[allowed_to_form_bins[:-1]] + \
              prograde_bh_masses[allowed_to_form_bins[1:]]) / \
                (mass_smbh * 3.0)), (1.0/3.0))
    # compare separations to mutual Hill spheres - negative values mean possible binary formation
    allowed_min_form_criteria = allowed_separations - R_Hill_allowed_bin_test
    print('criteria',allowed_min_form_criteria)
    # collect indices of possible real binaries (where separation is less than mutual Hill sphere)
    allowed_indx_form_criteria = np.where(allowed_min_form_criteria < 0)
    print(allowed_indx_form_criteria)
    allowed_idx_crit = allowed_indx_form_criteria[0]
    print(allowed_idx_crit)

    if np.size(allowed_indx_form_criteria) >0: 
        #If multiple negative results in criteria
        item1 = np.empty(len(allowed_idx_crit))
        item2 = np.empty(len(allowed_idx_crit))
        item1_idx = np.empty(len(allowed_idx_crit))
        item2_idx = np.empty(len(allowed_idx_crit))
        idx1 = np.empty(len(allowed_idx_crit))
        idx2 = np.empty(len(allowed_idx_crit))
        for i in range(len(allowed_idx_crit)):
            item1[i] = sorted_allowed_bh_loc[allowed_idx_crit[i]]
            item2[i] = sorted_allowed_bh_loc[allowed_idx_crit[i]+1]         
            print(item1[0],item2[0])
            item1_idx = np.where(prograde_bh_locations == item1[i])
            item2_idx = np.where(prograde_bh_locations == item2[i])
            print(item1_idx[0],item2_idx[0])
            idx1 = item1_idx[0] 
            idx2 = item2_idx[0]
            print(idx1,idx2)

        #for j in range(len(idx1)):
            #print(idx1[j],idx2[j]) 
            #print(prograde_bh_locations[idx1[j]],prograde_bh_locations[idx2[j]],prograde_bh_orb_ecc[idx1[j]],prograde_bh_orb_ecc[idx2[j]])
    
    # Now deal with sequences: compute separation/R_Hill for all
    allowed_sequences_to_test = (allowed_separations[allowed_indx_form_criteria])/(R_Hill_allowed_bin_test[allowed_indx_form_criteria])
    #print('allowed seqs to test', allowed_sequences_to_test)
    # sort sep/R_Hill for all 'binaries' that need checking & store indices
    sorted_allowed_sequences = np.sort(allowed_sequences_to_test)
    #print(sorted_allowed_sequences)
    sorted_allowed_sequences_indices = np.argsort(allowed_sequences_to_test)
    #print(sorted_allowed_sequences_indices)
    # the smallest sep/R_Hill should always form a binary, so
    if np.count_nonzero(sorted_allowed_sequences_indices) > 0:
        allowed_checked_binary_index = np.array([sorted_allowed_sequences_indices[0]],dtype = int)
        #print(allowed_checked_binary_index)
    
    for i in range(len(sorted_allowed_sequences)): 
        allowed_checked_binary_index = np.array([sorted_allowed_sequences_indices[0]],dtype = int)
        # if we haven't already counted it
        if (sorted_allowed_sequences_indices[i] not in allowed_checked_binary_index):
            # and it isn't the implicit partner of something we've already counted
            if (sorted_allowed_sequences_indices[i] not in allowed_checked_binary_index+1):
                # and the implicit partner of this thing isn't already counted
                if (sorted_allowed_sequences_indices[i]+1 not in allowed_checked_binary_index):
                    # and the implicit partner of this thing isn't already an implicit partner we've counted
                    if (sorted_allowed_sequences_indices[i]+1 not in allowed_checked_binary_index+1):
                        # then you can count it as a real binary
                        allowed_checked_binary_index = np.append(allowed_checked_binary_index, sorted_allowed_sequences_indices[i])

    
    #Check if any of the checked binary index array are in the array of indices that can form bins
    #check_overlap = np.isin(checked_binary_index,allowed_to_form_bins)
    #final_overlap = np.array(check_overlap.nonzero()[0])
    #print('final overlap',final_overlap)
    #print('sorted bh locs',sorted_bh_locations[final_overlap])
    #print('orb eccs',sorted_bh_ecc_array[final_overlap])
    #Check for adjacent integers in final overlap
    #Take the difference of integers in final overlap
    #diffs_final_overlap = np.diff(final_overlap)
    #print(diffs_final_overlap)
    #possible_bins = np.where(diffs_final_overlap == 1)
    #print(possible_bins)
    #print('final checked binary index ',checked_binary_index)
    
    # create array of all real binaries
    # BUT what are we returning? need indices of original arrays
    # that go to singleton vs binary assignments--actual binary formation *should* happen elsewhere!
    # I have the indices of the sorted_bh_locations array that correspond to actual binaries
    # these are checked_binary_index, checked_binary_index+1
    if np.count_nonzero(sorted_allowed_sequences) > 0:
        final_binary_indices = np.array([idx1,idx2])
#        final_binary_indices = np.array([sorted_bh_location_indices[allowed_checked_binary_index], sorted_bh_location_indices[allowed_checked_binary_index+1]])
        print(np.shape(final_binary_indices))
        print(np.shape(final_binary_indices)[1])
        #print(prograde_bh_locations[allowed_checked_binary_index],prograde_bh_locations[allowed_checked_binary_index+1])
    else: 
        final_binary_indices=np.empty_like(allowed_sequences_to_test)
    #return all_binary_indices
    return final_binary_indices

def eccentricity_check(prograde_bh_locations, prograde_bh_orb_ecc, e_crit):
    """If singleton orbital eccentricities are below the critical value (e <= e_crit (usually 0.01)), 
    they are eligible for binary formation.

    Parameters
    ----------
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of 
        gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_orb_ecc : float array
        Orbital ecc of singleton BH after damping during timestep
    e_crit : float
        Critical eccentricity allowing bin formation and migration

    Returns
    -------
    subset : [2,N] int array
        Array of indices corresponding to locations in prograde_bh_locations, prograde_bh_masses,
        prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_orb_ecc and prograde_bh_generations which corresponds
        to binaries that form in this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.
    sorted_bh_locations : 
    """
    
    #Singleton BH with orb ecc < e_crit (candidates for binary formation)
    print(prograde_bh_orb_ecc, e_crit, prograde_bh_orb_ecc)
    prograde_bh_can_form_bins = np.ma.masked_where(prograde_bh_orb_ecc >e_crit, prograde_bh_orb_ecc)    
    indices_bh_can_form_bins = np.ma.nonzero(prograde_bh_can_form_bins)
    # Indices of those candidates for binary formation
    allowed_to_form_bins = np.array(indices_bh_can_form_bins[0])
    #Sort the location of the candidates
    sorted_bh_locations = np.sort(prograde_bh_locations[allowed_to_form_bins])
    #Sort the indices of all singleton BH (the superset)
    sorted_bh_location_indices_superset = np.argsort(prograde_bh_locations)
    #Set the condition for membership in candidate array to be searched/tested
    condition = np.isin(sorted_bh_location_indices_superset, allowed_to_form_bins)
    #Here is the subset of indices that can be tested for binarity
    subset = np.extract(condition,sorted_bh_location_indices_superset)

    return subset, sorted_bh_locations


def binary_check2(prograde_bh_locations, prograde_bh_masses, mass_smbh, prograde_bh_orb_ecc, e_crit):
    """Determines which prograde BH will form binaries in this timestep. Takes as inputs
    the singleton BH locations,masses & orbital eccentricities, and takes the candidate binary population from 
    BH with orbital eccentricities damped to <e_crit.
    Among this damped population, checks if their separations are less than
    the mutual Hill sphere of any 2 adjacent BH. If this is the case, determine the
    smallest separation pairs (in units of their mutual Hill sphere) to form a set of
    actual binaries (this module does handle cases where 3 or more bodies *might* form
    some set of binaries which would be mutually exclusive; however it does not handle
    or even flag the implied triple system dynamics). Returns a 2xN array of the relevant
    binary indices, for further handling to form actual binaries & assign additional
    parameters (e.g. angular momentum of the binary).

    Parameters
    ----------
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of 
        gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        initial masses of bh in prograde orbits around SMBH in units of solar masses
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_orb_ecc : float array
        Orbital ecc of singleton BH after damping during timestep
    e_crit : float
        Critical eccentricity allowing bin formation and migration

    Returns
    -------
    all_binary_indices : [2,N] int array
        array of indices corresponding to locations in prograde_bh_locations, prograde_bh_masses,
        prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_orb_ecc and prograde_bh_generations which corresponds
        to binaries that form in this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.
    """

    # First check for BH with sufficiently damped orbital eccentricity (e<=e_crit (usually 0.01)). 
    # This population is the sub-set of prograde BH from which we CAN form binaries.
    
    subset, sorted_bh_locations = eccentricity_check(prograde_bh_locations, prograde_bh_orb_ecc, e_crit)
    
    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    # Note length of separations is 1 less than prograde_bh_locations
    
    #This is the set of separations between the sorted candidate BH
    separations = np.diff(sorted_bh_locations)
 
    # Now compute mutual hill spheres of all possible candidate binaries if can test    
    if len(separations) > 0:
        R_Hill_possible_binaries = (sorted_bh_locations[:-1] + separations/2.0) * \
            pow(((prograde_bh_masses[subset[:-1]] + \
                  prograde_bh_masses[subset[1:]]) / \
                    (mass_smbh * 3.0)), (1.0/3.0))
        # compare separations to mutual Hill spheres - negative values mean possible binary formation
        minimum_formation_criteria = separations - R_Hill_possible_binaries
        #print("sep -R_hill", minimum_formation_criteria)
        # collect indices of possible real binaries (where separation is less than mutual Hill sphere)
        index_formation_criteria = np.where(minimum_formation_criteria < 0)
        
        #Here's the index of the array of candidates
        test_idx = index_formation_criteria[0]
        #print("test_idx", test_idx)
        
        #If we actually have any candidates this time step
        if np.size(test_idx) >0:
            #print("subset(test_idx)",subset[test_idx][0],subset[test_idx+1][0])
            #Start with real index (from full singleton array) of 1st candidate binary component (implicit + 1 partner since separations are ordered )
            bin_indices = np.array([subset[test_idx[0]],subset[test_idx[0]+1]])
            #If only 1 binary this timestep, return this binary!
            all_binary_indices = np.array([subset[test_idx],subset[test_idx+1]])
            
            for i in range(len(test_idx)):
                #If more than 1 binary
                if i >0:
                    # append nth binary indices formed this timestep
                    bin_indices = np.append(bin_indices,[subset[test_idx[i]],subset[test_idx[i]+1]])
                    #print("2XBin", bin_indices)
                    
                    #Check to see if repeat binaries among the set of binaries formed (e.g. (1,2)(2,3) )
                    #If repeats, only form a binary from the pair with smallest fractional Hill sphere separation

                    # Compute separation/R_Hill for all
                    sequences_to_test = (separations[test_idx])/(R_Hill_possible_binaries[test_idx])
                    #print(sequences_to_test)
                    # sort sep/R_Hill for all 'binaries' that need checking & store indices
                    sorted_sequences = np.sort(sequences_to_test)
                    #print(sorted_sequences)
                    #Sort the indices for the test
                    sorted_sequences_indices = np.argsort(sequences_to_test)
                    #print(sorted_sequences_indices)

                    # Assume the smallest sep/R_Hill should form a binary, so
                    if len(sorted_sequences) > 0:
                        #Index of smallest sorted fractional Hill radius binary so far
                        checked_binary_index = np.array([test_idx[sorted_sequences_indices[0]]])
                    else:
                        checked_binary_index = []    
                    #print(checked_binary_index)
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
                                        #print("checked_binary_index",checked_binary_index)
                    all_binary_indices = np.array([subset[checked_binary_index],subset[checked_binary_index+1]])

            
            

        else:
            #No binaries from candidates this time step
            all_binary_indices = []
        
    else:
        #No candidate for binarity testing yet
        all_binary_indices = []
    
    return all_binary_indices
        
def qian24_test(rng, prograde_bh_locations, prograde_bh_masses, mass_smbh, prograde_bh_orb_ecc, e_crit):
    """Form binaries depending on results from scenario 2 outlined in Qian+24 (https://arxiv.org/pdf/2310.12208.pdf).
    According to figure 10, about 1/3 of encounters that reach separations between 1.4 and 2.3 times their mutual hill
    sphere with gas dynamical friction timescales: 0.05 < 1/tau < 0.4.

    Parameters
    ----------
    rng : _type_
        Random number generator that is set for this instantiation of McFACTS.
    prograde_bh_locations : float array
        Locations of prograde singleton BH at start of timestep in units of 
        gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        Initial masses of bh in prograde orbits around SMBH in units of solar masses
    mass_smbh : float
        Mass of supermassive black hole in units of solar masses
    prograde_bh_orb_ecc : float array
        Orbital ecc of singleton BH after damping during timestep
    e_crit : float
        Critical eccentricity allowing bin formation and migration    
    Returns
    -------
    all_binary_indices : [2,N] int array
        array of indices corresponding to locations in prograde_bh_locations, prograde_bh_masses,
        prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_orb_ecc and prograde_bh_generations which corresponds
        to binaries that form in this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.
    """

    # First check for BH with sufficiently damped orbital eccentricity (e<=e_crit (usually 0.01)). 
    # This population is the sub-set of prograde BH from which we CAN form binaries.
    
    subset, sorted_bh_locations = eccentricity_check(prograde_bh_locations, prograde_bh_orb_ecc, e_crit)

    # This is the set of separations between the sorted candidate BHs
    separations = np.diff(sorted_bh_locations)
 
    # Now compute the selection critera based on QianLiLai24 figure 11
    if len(separations) > 0:

        # Separation criteria: 1.4 < K < 2.2 , where K = a1-a2/R_Hill  (factors of Hill radius R_Hill)

        # Mutual Hill sphere of objects
        R_Hill_possible_binaries = (sorted_bh_locations[:-1] + separations/2.0) * \
            pow(((prograde_bh_masses[subset[:-1]] + \
                  prograde_bh_masses[subset[1:]]) / \
                    (mass_smbh * 3.0)), (1./3))

        num_R_Hill = separations / R_Hill_possible_binaries
        pass_R_Hill_indices = np.where((num_R_Hill > 1.4) & (num_R_Hill < 2.3))[0]

        if len(pass_R_Hill_indices) == 0:
            # no circularized objects pass the Hill radius test this timestep
            all_binary_indices = []

        else: # apply the friction timescale test
            # Get indices of each pair of objects that pass the Hill radius test
            pass_R_Hill_index_pairs = np.zeros((len(pass_R_Hill_indices), 2), dtype=int)
            for i, ind in enumerate(pass_R_Hill_indices):
                # Indices of the two objects with the relevant separation
                pass_R_Hill_index_pairs[i] = subset[ind], subset[ind+1]

            # Gas drag friction timescales (tau) criteria: 0.05 < 1/tau < 0.40
            # Chose values from SG03 model at ~700 Rg around 1e8 Msun SMBH
            sound_speed = 1e7
            gas_density = 1e-10

            pass_friction_pairs_indices = []
            for j, ind_pairs in enumerate(pass_R_Hill_index_pairs):
                larger_bh_index = prograde_bh_masses[ind_pairs].argmax()
                larger_bh_mass = prograde_bh_masses[ind_pairs][larger_bh_index]
                larger_bh_location = prograde_bh_locations[ind_pairs[larger_bh_index]]

                # Use Eq. (7) from Model 2 to calculate the constant timescale used in Model 1 for the larger mass
                friction_timescale = 1 / (4 * np.pi * G**2) * sound_speed**3 / (gas_density * larger_bh_mass)
                keplerian_velocity = pow(G * mass_smbh, 1./2) * pow(larger_bh_location, -3./2)

                friction_time = friction_timescale * keplerian_velocity

                # Apply friction timescale criteria
                if (0.05 < 1/friction_time) & (1/friction_time < 0.40):
                    pass_friction_pairs_indices.append(ind_pairs)
            
            all_binary_indices = np.asarray(pass_friction_pairs_indices, dtype=int)
            if len(all_binary_indices) > 0:
                import ipdb; ipdb.set_trace()

            # Need to check whether a given black hole is present in two or more new binaries...
    else:
        all_binary_indices = []


    return all_binary_indices