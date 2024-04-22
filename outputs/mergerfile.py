import numpy as np

names_rec = ['CM', 'M', 'chi_eff', 'a_tot', 'spin_angle', 'm1', 'm2', 'a1', 'a2', 'theta1', 'theta2', 'gen1', 'gen2', 't_merge', 'chi_p']
dtype_rec = np.dtype( [(x, float) for x in names_rec])


def extend_rec_merged_bh(merged_rec_array, n_mergers_so_far, merger_indices,chi,mass,spin,nprops_mergers,num_mergers,chi_p):
    number_of_mergers = int(num_mergers)

    # Reallocate record array in larger size.  Note
    my_type = merged_rec_array.dtype
    my_len = len(merged_rec_array)
    if not(my_len < number_of_mergers + n_mergers_so_far):
        # Define new size
        my_new_len = int(my_len*1.5)  # fudge factor
        # Allocate
        new_recarray = np.empty(my_new_len,dtype=my_type)
        # Copy
        for name in merged_rec_array:
            new_recarray[name][:my_len] = merged_rec_array[name]

    # Now add new elements to end.  Works with items and sequences
    indx_start = n_mergers_so_far
    indx_end = n_mergers_so_far+num_mergers
    merged_rec_array['CM'][indx_start:indx_end] = bin_array[9,merger_indices]
    merged_rec_array['M'][indx_start:indx_end] = mass
    merged_rec_array['chi_eff'][indx_start:indx_end] = chi
    merged_rec_array['a_tot'][indx_start:indx_end] = spin
    merged_rec_array['spin_angle'][indx_start:indx_end] = 0.0
    merged_rec_array['m1'][indx_start:indx_end] = bin_array[2,merger_indices]
    merged_rec_array['m2'][indx_start:indx_end] = bin_array[3,merger_indices]
    merged_rec_array['a1'][indx_start:indx_end] = bin_array[4,merger_indices]
    merged_rec_array['a2'][indx_start:indx_end] = bin_array[5,merger_indices]
    merged_rec_array['theta1'][indx_start:indx_end] = bin_array[6,merger_indices]
    merged_rec_array['theta2'][indx_start:indx_end] = bin_array[7,merger_indices]
    merged_rec_array['gen1'][indx_start:indx_end] = bin_array[14,merger_indices]
    merged_rec_array['gen2'][indx_start:indx_end] = bin_array[15,merger_indices]
    merged_rec_array['t_merge'][indx_start:indx_end] = bin_array[12,merger_indices]
    merged_rec_array['chi_p'][indx_start:indx_end] = chi_p
    return merged_rec_array


def merged_bh(merged_array,bin_array,merger_indices,i,chi_here,mass_here,spin_here,nprops_mergers,n_mergers_so_far,chi_p):
    """
    Recording merger event.  Designed only to handle one event at a time, with hardcoded labels
    """
    #Comment here
    #Return an array with properties of merger
    #Center of mass, M_total, Chi_eff, a_tot, spin_angle, m1,m2,a1,a2,theta1,theta2,gen1,gen2,time of merger
    
    #merger_indices = np.array(merger_indices_here)
    #print(merger_indices)
    #print("length merger indices",len(merger_indices))
    #n_mergers_so_far=int(num_mergers)
    #n_mergers_this_timestep = len(merger_indices)

    #if len(merger_indices)>1:   # deal with the fact that we have multiple mergers
    #    print(" Multiple mergers - we are about to lose events ")
    #    print(mass_here, chi_here, spin_here)
    #    merger_indices = merger_indices[0]
    #    if isinstance(mass_here, np.ndarray):
    #        mass =  mass_here[0]
    #    else:
    #        mass = mass_here
    #    if isinstance(spin_here, np.ndarray):
    #        spin = spin_here[0]
    #        chi = chi_here[0]
    #    else:
    #        spin = spin_here
    #        chi = chi_here
    #else:
    mass= mass_here
    spin = spin_here
    chi = chi_here
    
    #num_mergers_this_timestep = len(merger_indices)
                
    #print("num mergers this timestep",n_mergers_this_timestep)
    #print("n_mergers_so_far + i",n_mergers_so_far + i)    
    #for i in range (0,n_mergers_this_timestep):
        #            merged_bh_com = merged_bh_array[0,n_mergers_so_far + i]
    merged_array[0,n_mergers_so_far + i] = bin_array[9,merger_indices[i]]
    merged_array[1,n_mergers_so_far + i] = mass
    merged_array[2,n_mergers_so_far + i] = chi
    merged_array[3,n_mergers_so_far + i] = spin
    merged_array[4,n_mergers_so_far + i] = 0.0
    merged_array[5,n_mergers_so_far + i] = bin_array[2,merger_indices[i]]
    merged_array[6,n_mergers_so_far + i] = bin_array[3,merger_indices[i]]
    merged_array[7,n_mergers_so_far + i] = bin_array[4,merger_indices[i]]
    merged_array[8,n_mergers_so_far + i] = bin_array[5,merger_indices[i]]
    merged_array[9,n_mergers_so_far + i] = bin_array[6,merger_indices[i]]
    merged_array[10,n_mergers_so_far + i] = bin_array[7,merger_indices[i]]
    merged_array[11,n_mergers_so_far + i] = bin_array[14,merger_indices[i]]
    merged_array[12,n_mergers_so_far + i] = bin_array[15,merger_indices[i]]
    merged_array[13,n_mergers_so_far + i] = bin_array[12,merger_indices[i]]
    merged_array[14,n_mergers_so_far + i] = chi_p
    return merged_array[:,n_mergers_so_far + i]
