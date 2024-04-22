import os
import pathlib

from cgi import print_arguments
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import scipy.interpolate

import sys
import argparse

from inputs import ReadInputs

from setup import setupdiskblackholes
from physics.migration.type1 import type1
from physics.accretion.eddington import changebhmass
from physics.accretion.torque import changebh
from physics.feedback.hankla21 import feedback_hankla21
from physics.dynamics import dynamics
from physics.eccentricity import orbital_ecc
from physics.binary.formation import hillsphere
from physics.binary.formation import add_new_binary
#from physics.binary.formation import secunda20
from physics.binary.evolve import evolve
from physics.binary.harden import baruteau11
from physics.binary.merge import tichy08
from physics.binary.merge import chieff
from physics.binary.merge import tgw
#from tests import tests
from outputs import mergerfile

verbose=False
#Maximum number of binaries allowed in a run
n_bins_max = 1000
n_bins_max_out = 100

binary_field_names="R1 R2 M1 M2 a1 a2 theta1 theta2 sep com t_gw merger_flag t_mgr  gen_1 gen_2  bin_ang_mom bin_ecc bin_incl"
merger_field_names=' '.join(mergerfile.names_rec)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--use-ini",help="Filename of configuration file", default=None)
parser.add_argument("--fname-output-mergers",default="output_mergers.dat",help="output merger file (if any)")
parser.add_argument("--fname-snapshots-bh",default="output_bh_[single|binary]_$(index).dat",help="output of BH index file ")
parser.add_argument("--no-snapshots", action='store_true')
parser.add_argument("--verbose",action='store_true')
parser.add_argument("-w", "--work-directory", default=pathlib.Path().parent.resolve(), help="Set the working directory for saving output. Default: current working directory", type=str)
parser.add_argument("--seed", type=int, default=None, help="Set the random seed. Randomly sets one if not passed. Default: None")
opts = parser.parse_args()
verbose = opts.verbose

# Get the parent path to this file and cd to that location for runtime
runtime_directory = pathlib.Path(__file__).parent.resolve()
os.chdir(runtime_directory)

# Get the user-defined or default working directory / output location
work_directory = pathlib.Path(opts.work_directory).resolve()
try: # check if working directory for output exists
    os.stat(work_directory)
except FileNotFoundError as e:
    raise e
print(f"Output will be saved to {work_directory}")

# set the seed for random number generation and reproducibility if not user-defined
if opts.seed == None:
    opts.seed = np.random.randint(low=0, high=int(1e18))
    print(f'Random number generator seed set to: {opts.seed}')


def main():
    """
    """

    # Setting up automated input parameters
    # see IOdocumentation.txt for documentation of variable names/types/etc.

    fname = "inputs/model_choice.txt"
    if opts.use_ini:
        fname = opts.use_ini
    mass_smbh, trap_radius, disk_outer_radius, alpha, n_iterations, mode_mbh_init, max_initial_bh_mass, \
         mbh_powerlaw_index, mu_spin_distribution, sigma_spin_distribution, \
             spin_torque_condition, frac_Eddington_ratio, max_initial_eccentricity, orb_ecc_damping, \
                 timestep, number_of_timesteps, disk_model_radius_array, disk_inner_radius,\
                     disk_outer_radius, surface_density_array, aspect_ratio_array, retro, feedback, capture_time, outer_capture_radius, crit_ecc, \
                        r_nsc_out, M_nsc, r_nsc_crit, nbh_nstar_ratio, mbh_mstar_ratio, nsc_index_inner, nsc_index_outer, h_disk_average, dynamic_enc, de\
                        = ReadInputs.ReadInputs_ini(fname)

    # create surface density & aspect ratio functions from input arrays
    surf_dens_func_log = scipy.interpolate.UnivariateSpline(disk_model_radius_array, np.log(surface_density_array))
    surf_dens_func = lambda x, f=surf_dens_func_log: np.exp(f(x))

    aspect_ratio_func_log = scipy.interpolate.UnivariateSpline(disk_model_radius_array, np.log(aspect_ratio_array))
    aspect_ratio_func = lambda x, f=aspect_ratio_func_log: np.exp(f(x))
    
    merged_bh_array_pop = []
    
    for iteration in range(n_iterations):
        print("Iteration", iteration)
        # Set random number generator for this run with incremented seed
        rng = np.random.default_rng(opts.seed + iteration)

        # Make subdirectories for each iteration
        # Fills run number with leading zeros to stay sequential
        iteration_zfilled_str = f"{iteration:>0{int(np.log10(n_iterations))+1}}"
        try: # Make subdir, exit if it exists to avoid clobbering.
            os.makedirs(os.path.join(work_directory, f"run{iteration_zfilled_str}"), exist_ok=False)
        except FileExistsError:
            raise FileExistsError(f"Directory \'run{iteration_zfilled_str}\' exists. Exiting so I don't delete your data.")

        # can index other parameter lists here if needed.
        # galaxy_type = galaxy_models[iteration] # e.g. star forming/spiral vs. elliptical
        # NSC mass
        # SMBH mass


        #Set up number of BH in disk
        n_bh = setupdiskblackholes.setup_disk_nbh(M_nsc,nbh_nstar_ratio,mbh_mstar_ratio,r_nsc_out,nsc_index_outer,mass_smbh,disk_outer_radius,h_disk_average,r_nsc_crit,nsc_index_inner)

        # generate initial BH parameter arrays
        print("Generate initial BH parameter arrays")
        bh_initial_locations = setupdiskblackholes.setup_disk_blackholes_location(rng, n_bh, disk_outer_radius)
        bh_initial_masses = setupdiskblackholes.setup_disk_blackholes_masses(rng, n_bh, mode_mbh_init, max_initial_bh_mass, mbh_powerlaw_index)
        bh_initial_spins = setupdiskblackholes.setup_disk_blackholes_spins(rng, n_bh, mu_spin_distribution, sigma_spin_distribution)
        bh_initial_spin_angles = setupdiskblackholes.setup_disk_blackholes_spin_angles(rng, n_bh, bh_initial_spins)
        bh_initial_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(rng, n_bh)
        if orb_ecc_damping == 1:
            bh_initial_orb_ecc = setupdiskblackholes.setup_disk_blackholes_eccentricity_uniform(rng, n_bh)
        else:
            bh_initial_orb_ecc = setupdiskblackholes.setup_disk_blackholes_circularized(rng,n_bh,crit_ecc)

        bh_initial_orb_incl = setupdiskblackholes.setup_disk_blackholes_inclination(rng, n_bh)
        #print("orb ecc",bh_initial_orb_ecc)
        #bh_initial_generations = np.ones((integer_nbh,),dtype=int)  

        bh_initial_generations = np.ones((n_bh,),dtype=int)

        # assign functions to variable names (continuity issue)
        # Disk surface density (in kg/m^2) is a function of radius, where radius is in r_g
        disk_surface_density = surf_dens_func
        # and disk aspect ratio is also a function of radius, where radius is in r_g
        disk_aspect_ratio = aspect_ratio_func
        # Housekeeping: Set up time
        initial_time = 0.0
        final_time = timestep*number_of_timesteps

        # Find prograde BH orbiters. Identify BH with orb. ang mom =+1
        bh_orb_ang_mom_indices = np.array(bh_initial_orb_ang_mom)
        prograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == 1)
        #retrograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == -1)
        prograde_bh_locations = bh_initial_locations[prograde_orb_ang_mom_indices]
        sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
        print("Sorted prograde BH locations:", len(sorted_prograde_bh_locations), len(prograde_bh_locations))
        print(sorted_prograde_bh_locations)
        print(prograde_bh_locations)
        #print("Aspect ratio",aspect_ratio_func(prograde_bh_locations))
        #Use masses of prograde BH only
        prograde_bh_masses = bh_initial_masses[prograde_orb_ang_mom_indices]
        print("Prograde BH initial masses", len(prograde_bh_masses))


        # Orbital eccentricities
        prograde_bh_orb_ecc = bh_initial_orb_ecc[prograde_orb_ang_mom_indices]
        print("Prograde orbital eccentricities",prograde_bh_orb_ecc)
        # Find which orbital eccentricities are <=h the disk aspect ratio and set up a mask
        #prograde_bh_crit_ecc = np.ma.masked_where(prograde_bh_orb_ecc >= aspect_ratio_func(prograde_bh_locations),prograde_bh_orb_ecc)
        # Orb eccentricities <2h (simple exponential damping): mask entries > 2*aspect_ratio
        #prograde_bh_modest_ecc = np.ma.masked_where(prograde_bh_orb_ecc > 2.0*aspect_ratio_func(prograde_bh_locations),prograde_bh_orb_ecc)
        #Orb eccentricities >2h (modified exponential damping): mask entries < 2*aspect_ratio
        #prograde_bh_large_ecc = np.ma.masked_where(prograde_bh_orb_ecc < 2.0*aspect_ratio_func(prograde_bh_locations),prograde_bh_orb_ecc)
        # Apply ecc damping to this masked array (where true)
        #prograde_bh_orb_ecc_damp = orbital_ecc.orbital_ecc_damping(mass_smbh, prograde_bh_locations, prograde_bh_masses, surf_dens_func, aspect_ratio_func, prograde_bh_orb_ecc, timestep, crit_ecc)

        #print('modest ecc ',prograde_bh_modest_ecc)
        #print('damped ecc',prograde_bh_orb_ecc_damp) 
         
        # Test dynamics
        #post_dynamics_orb_ecc = dynamics.circular_singles_encounters_prograde(rng,mass_smbh, prograde_bh_locations, prograde_bh_masses, surf_dens_func, aspect_ratio_func, prograde_bh_orb_ecc, timestep, crit_ecc, de)
    
        # Migrate
        # First if feedback present, find ratio of feedback heating torque to migration torque
        #if feedback > 0:
        #        ratio_heat_mig_torques = feedback_hankla21.feedback_hankla(prograde_bh_locations, surf_dens_func, frac_Eddington_ratio, alpha)
        #else:
        #        ratio_heat_mig_torques = np.ones(len(prograde_bh_locations))   
        # then migrate as usual
        #prograde_bh_locations_new = type1.type1_migration(mass_smbh , prograde_bh_locations, prograde_bh_masses, disk_surface_density, disk_aspect_ratio, timestep, ratio_heat_mig_torques, trap_radius, prograde_bh_orb_ecc,crit_ecc)
        

        #Orbital inclinations
        prograde_bh_orb_incl = bh_initial_orb_incl[prograde_orb_ang_mom_indices]
        print("Prograde orbital inclinations")

        # Housekeeping: Fractional rate of mass growth per year at 
        # the Eddington rate(2.3e-8/yr)
        mass_growth_Edd_rate = 2.3e-8
    
        # Housekeeping: minimum spin angle resolution 
        # (ie less than this value gets fixed to zero) 
        # e.g 0.02 rad=1deg
        spin_minimum_resolution = 0.02
        #Torque prograde orbiting BH only
        prograde_bh_spins = bh_initial_spins[prograde_orb_ang_mom_indices]
        prograde_bh_spin_angles = bh_initial_spin_angles[prograde_orb_ang_mom_indices]
        prograde_bh_generations = bh_initial_generations[prograde_orb_ang_mom_indices]

        # Housekeeping:
        # Number of binary properties that we want to record (e.g. R1,R2,M1,M2,a1,a2,theta1,theta2,sep,com,t_gw,merger_flag,time of merger, gen_1,gen_2, bin_ang_mom, bin_ecc, bin_incl)o
        number_of_bin_properties = len(binary_field_names.split())+1
        integer_nbinprop = int(number_of_bin_properties)
        bin_index = 0
        test_bin_number = n_bins_max
        integer_test_bin_number = int(test_bin_number)
        number_of_mergers = 0

        # Set up empty initial Binary array
        # Initially all zeros, then add binaries plus details as appropriate
        binary_bh_array = np.zeros((integer_nbinprop,integer_test_bin_number))
        # Set up normalization for t_gw (SF: I do not like this way of handling, flag for update)
        norm_t_gw = tgw.normalize_tgw(mass_smbh)
        print("Scale of t_gw (yrs)=", norm_t_gw)
    
        # Set up merger array (identical to binary array)
        merger_array = np.zeros((integer_nbinprop,integer_test_bin_number))
    
        # Set up output array (mergerfile)
        nprop_mergers=len(mergerfile.names_rec)
        integer_nprop_merge=int(nprop_mergers)
        merged_bh_array = np.zeros((integer_nprop_merge,integer_test_bin_number))

        # Start Loop of Timesteps
        print("Start Loop!")
        time_passed = initial_time
        print("Initial Time(yrs) = ",time_passed)

        n_mergers_so_far = 0
        n_timestep_index = 0
        n_merger_limit =1e4

        while time_passed < final_time:
            # Record 
            if not(opts.no_snapshots):
                n_bh_out_size = len(prograde_bh_locations)

                #svals = list(map( lambda x: x.shape,[prograde_bh_locations, prograde_bh_masses, prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_orb_ecc, prograde_bh_generations[:n_bh_out_size]]))
                # Single output:  does work
                np.savetxt(os.path.join(work_directory, f"run{iteration_zfilled_str}/output_bh_single_{n_timestep_index}.dat"), np.c_[prograde_bh_locations.T, prograde_bh_masses.T, prograde_bh_spins.T, prograde_bh_spin_angles.T, prograde_bh_orb_ecc.T, prograde_bh_generations[:n_bh_out_size].T], header="r_bh m a theta ecc gen")
                # np.savetxt(os.path.join(work_directory, "output_bh_single_{}.dat".format(n_timestep_index)), np.c_[prograde_bh_locations.T, prograde_bh_masses.T, prograde_bh_spins.T, prograde_bh_spin_angles.T, prograde_bh_orb_ecc.T, prograde_bh_generations[:n_bh_out_size].T], header="r_bh m a theta ecc gen")
                # Binary output: does not work
                np.savetxt(os.path.join(work_directory, f"run{iteration_zfilled_str}/output_bh_binary_{n_timestep_index}.dat"), binary_bh_array[:,:n_mergers_so_far+1].T, header=binary_field_names)
                # np.savetxt(os.path.join(work_directory, "output_bh_binary_{}.dat".format(n_timestep_index)), binary_bh_array[:,:n_mergers_so_far+1].T, header=binary_field_names)
                n_timestep_index +=1

            #Order of operations: 
            # No migration until orbital eccentricity damped to e_crit (To do: actually should be h)
            # 1. check orb. eccentricity to see if any prograde_bh_location BH have orb. ecc. <e_crit.
            #    Create array prograde_bh_location_ecrit for those (mask prograde_bh_locations?)
            #       If yes, migrate those BH.
            #       All other BH, damp ecc and spin *down* BH (retrograde accretion), accrete mass.
            # 2. Run close encounters only on those prograde_bh_location_ecrit members.
        
            # Migrate
            # First if feedback present, find ratio of feedback heating torque to migration torque
            if feedback > 0:
                ratio_heat_mig_torques = feedback_hankla21.feedback_hankla(prograde_bh_locations, surf_dens_func, frac_Eddington_ratio, alpha)
            else:
                ratio_heat_mig_torques = np.ones(len(prograde_bh_locations))   
            # then migrate as usual
            #print("TEST locations",prograde_bh_locations)
            prograde_bh_locations = type1.type1_migration(mass_smbh , prograde_bh_locations, prograde_bh_masses, disk_surface_density, disk_aspect_ratio, timestep, ratio_heat_mig_torques, trap_radius, prograde_bh_orb_ecc,crit_ecc)
            #print("NEW locations",prograde_bh_locations)
            # Accrete
            prograde_bh_masses = changebhmass.change_mass(prograde_bh_masses, frac_Eddington_ratio, mass_growth_Edd_rate, timestep)
            # Spin up
            prograde_bh_spins = changebh.change_spin_magnitudes(prograde_bh_spins, frac_Eddington_ratio, spin_torque_condition, timestep, prograde_bh_orb_ecc, crit_ecc)
            # Torque spin angle
            prograde_bh_spin_angles = changebh.change_spin_angles(prograde_bh_spin_angles, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep, prograde_bh_orb_ecc, crit_ecc)

            # Damp BH orbital eccentricity
            prograde_bh_orb_ecc = orbital_ecc.orbital_ecc_damping(mass_smbh, prograde_bh_locations, prograde_bh_masses, surf_dens_func, aspect_ratio_func, prograde_bh_orb_ecc, timestep, crit_ecc)
            # Perturb eccentricity via dynamical encounters
            if dynamic_enc > 0:
                prograde_bh_locn_orb_ecc = dynamics.circular_singles_encounters_prograde(rng,mass_smbh, prograde_bh_locations, prograde_bh_masses, surf_dens_func, aspect_ratio_func, prograde_bh_orb_ecc, timestep, crit_ecc, de)
                prograde_bh_locations = prograde_bh_locn_orb_ecc[0]
                prograde_bh_orb_ecc = prograde_bh_locn_orb_ecc[1]
                prograde_bh_locations = prograde_bh_locations[0]
                prograde_bh_orb_ecc = prograde_bh_orb_ecc[0]
            
            # Do things to the binaries--first check if there are any:
            if bin_index > 0:
                # If there are binaries, evolve them
                # Harden binaries via gas
                #Choose between Baruteau et al. 2011 gas hardening, or gas hardening from LANL simulations. To do: include dynamical hardening/softening from encounters
                binary_bh_array = baruteau11.bin_harden_baruteau(binary_bh_array,integer_nbinprop,mass_smbh,timestep,norm_t_gw,bin_index,time_passed)
                #print("Harden binary")
                #print("Time passed = ", time_passed)
                # Accrete gas onto binary components
                binary_bh_array = evolve.change_bin_mass(binary_bh_array, frac_Eddington_ratio, mass_growth_Edd_rate, timestep, integer_nbinprop, bin_index)
                # Spin up binary components
                binary_bh_array = evolve.change_bin_spin_magnitudes(binary_bh_array, frac_Eddington_ratio, spin_torque_condition, timestep, integer_nbinprop, bin_index)
                # Torque angle of binary spin components
                binary_bh_array = evolve.change_bin_spin_angles(binary_bh_array, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep, integer_nbinprop, bin_index)
 
                #Migrate binaries
                # First if feedback present, find ratio of feedback heating torque to migration torque
                #print("feedback",feedback)
                if feedback > 0:
                    ratio_heat_mig_torques_bin_com = evolve.com_feedback_hankla(binary_bh_array, surf_dens_func, frac_Eddington_ratio, alpha)
                else:
                    ratio_heat_mig_torques_bin_com = np.ones(len(binary_bh_array[9,:]))   

                # Migrate binaries center of mass
                binary_bh_array = evolve.bin_migration(mass_smbh, binary_bh_array, disk_surface_density, disk_aspect_ratio, timestep,ratio_heat_mig_torques_bin_com,trap_radius)
            
                #Check and see if merger flagged during hardening (row 11, if negative)
                merger_flags = binary_bh_array[11,:]
                any_merger = np.count_nonzero(merger_flags)
                 
                if verbose:
                    print(merger_flags)
                merger_indices = np.where(merger_flags < 0.0)
                if isinstance(merger_indices,tuple):
                    merger_indices = merger_indices[0]
                if verbose:
                    print(merger_indices)
                #print(binary_bh_array[:,merger_indices])
                if any_merger > 0:
                    for i in range(any_merger):
                    #print("Merger!")
                    # send properties of merging objects to static variable names
                        #mass_1[i] = binary_bh_array[2,merger_indices[i]]
                        #mass_2[i] = binary_bh_array[3,merger_indices[i]]
                        #spin_1[i] = binary_bh_array[4,merger_indices[i]]
                        #spin_2[i] = binary_bh_array[5,merger_indices[i]]
                        #angle_1[i] = binary_bh_array[6,merger_indices[i]]
                        #angle_2[i] = binary_bh_array[7,merger_indices[i]]
                        #bin_ang_mom[i] = binary_bh_array[16,merger_indices]

                    # calculate merger properties
                        merged_mass = tichy08.merged_mass(binary_bh_array[2,merger_indices[i]], binary_bh_array[3,merger_indices[i]], binary_bh_array[4,merger_indices[i]], binary_bh_array[5,merger_indices[i]])
                        merged_spin = tichy08.merged_spin(binary_bh_array[2,merger_indices[i]], binary_bh_array[3,merger_indices[i]], binary_bh_array[4,merger_indices[i]], binary_bh_array[5,merger_indices[i]], binary_bh_array[16,merger_indices[i]])
                        merged_chi_eff = chieff.chi_effective(binary_bh_array[2,merger_indices[i]], binary_bh_array[3,merger_indices[i]], binary_bh_array[4,merger_indices[i]], binary_bh_array[5,merger_indices[i]], binary_bh_array[6,merger_indices[i]], binary_bh_array[7,merger_indices[i]], binary_bh_array[16,merger_indices[i]])
                        merged_bh_array[:,n_mergers_so_far + i] = mergerfile.merged_bh(merged_bh_array,binary_bh_array,merger_indices,i,merged_chi_eff,merged_mass,merged_spin,nprop_mergers,n_mergers_so_far)
                    # do another thing
                    merger_array[:,merger_indices] = binary_bh_array[:,merger_indices]
                    #Reset merger marker to zero
                    #n_mergers_so_far=int(number_of_mergers)
                    #Remove merged binary from binary array. Delete column where merger_indices is the label.
                    #print("!Merger properties!",binary_bh_array[:,merger_indices],merger_array[:,merger_indices],merged_bh_array)
                    binary_bh_array=np.delete(binary_bh_array,merger_indices,1)
                
                    #Reduce number of binaries by number of mergers
                    bin_index = bin_index - len(merger_indices)
                    #print("bin index",bin_index)
                    #Find relevant properties of merged BH to add to single BH arrays
                    num_mergers_this_timestep = len(merger_indices)
                
                    #print("num mergers this timestep",num_mergers_this_timestep)
                    #print("n_mergers_so_far",n_mergers_so_far)    
                    for i in range (0, num_mergers_this_timestep):
                        merged_bh_com = merged_bh_array[0,n_mergers_so_far + i]
                        merged_mass = merged_bh_array[1,n_mergers_so_far + i]
                        merged_spin = merged_bh_array[3,n_mergers_so_far + i]
                        merged_spin_angle = merged_bh_array[4,n_mergers_so_far + i]
                    #New bh generation is max of generations involved in merger plus 1
                        merged_bh_gen = np.maximum(merged_bh_array[11,n_mergers_so_far + i],merged_bh_array[12,n_mergers_so_far + i]) + 1.0 
                    #print("Merger at=",merged_bh_com,merged_mass,merged_spin,merged_spin_angle,merged_bh_gen)
                    # Add to number of mergers
                    n_mergers_so_far += len(merger_indices)
                    number_of_mergers += len(merger_indices)

                    # Append new merged BH to arrays of single BH locations, masses, spins, spin angles & gens
                    # For now add 1 new orb ecc term of 0.01. TO DO: calculate v_kick and resulting perturbation to orb ecc.
                    prograde_bh_locations = np.append(prograde_bh_locations,merged_bh_com)
                    prograde_bh_masses = np.append(prograde_bh_masses,merged_mass)
                    prograde_bh_spins = np.append(prograde_bh_spins,merged_spin)
                    prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,merged_spin_angle)
                    prograde_bh_generations = np.append(prograde_bh_generations,merged_bh_gen)
                    prograde_bh_orb_ecc = np.append(prograde_bh_orb_ecc,0.01)
                    prograde_bh_orb_incl = np.append(prograde_bh_orb_incl,0.0)
                    sorted_prograde_bh_locations=np.sort(prograde_bh_locations)
                    if verbose:
                        print("New BH locations", sorted_prograde_bh_locations)
                    #print("Merger Flag!")
                    #print(number_of_mergers)
                    #print("Time ", time_passed)
                    if verbose:
                        print(merger_array)
                else:                
                    # No merger
                    # do nothing! hardening should happen FIRST (and now it does!)
                    if verbose:
                        if bin_index>0: # verbose:
                            #print(" BH binaries ", bin_index,  binary_bh_array[:,:int(bin_index)].shape)
                            print(binary_bh_array[:,:int(bin_index)].T)  # this makes printing work as expected
            else:            
                if verbose:
                    print("No binaries formed yet")
                # No Binaries present in bin_array. Nothing to do.

            #If a close encounter within mutual Hill sphere add a new Binary

            # check which binaries should get made
            close_encounters2 = hillsphere.binary_check2(prograde_bh_locations, prograde_bh_masses, mass_smbh, prograde_bh_orb_ecc, crit_ecc)
                    
            # apply qian24 test
            # close_encounters2 = hillsphere.qian24_test(prograde_bh_locations, prograde_bh_masses, mass_smbh, prograde_bh_orb_ecc, crit_ecc, surf_dens_func, aspect_ratio_func)
            
            #print("Output of close encounters", close_encounters2)
            # print(close_encounters)
            if np.size(close_encounters2) > 0:
                #print("Make binary at time ", time_passed)
                #print("shape1",np.shape(close_encounters2)[1])
                #print("shape0",np.shape(close_encounters2)[0])
                # number of new binaries is length of 2nd dimension of close_encounters2
                #number_of_new_bins = np.shape(close_encounters2)[1]
                number_of_new_bins = np.shape(close_encounters2)[1]
                #print("number of new bins", number_of_new_bins)
                # make new binaries
                binary_bh_array = add_new_binary.add_to_binary_array2(rng, binary_bh_array, prograde_bh_locations, prograde_bh_masses, prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_generations, close_encounters2, bin_index, retro)
                bin_index = bin_index + number_of_new_bins
                #print("Binary array",binary_bh_array[:,0])
                # delete corresponding entries for new binary members from singleton arrays
                prograde_bh_locations = np.delete(prograde_bh_locations, close_encounters2)
                prograde_bh_masses = np.delete(prograde_bh_masses, close_encounters2)
                prograde_bh_spins = np.delete(prograde_bh_spins, close_encounters2)
                prograde_bh_spin_angles = np.delete(prograde_bh_spin_angles, close_encounters2)
                prograde_bh_generations = np.delete(prograde_bh_generations, close_encounters2)
                prograde_bh_orb_ecc = np.delete(prograde_bh_orb_ecc, close_encounters2)
                prograde_bh_orb_incl = np.delete(prograde_bh_orb_incl, close_encounters2)
            
            #Empty close encounters
            empty = []
            close_encounters2 = np.array(empty)

            #After this time period, was there a disk capture via orbital grind-down?
            # To do: What eccentricity do we want the captured BH to have? Right now ecc=0.0? Should it be ecc<h at a?             
            # Assume 1st gen BH captured and orb ecc =0.0
            # To do: Bias disk capture to more massive BH!
            capture = time_passed % capture_time
            if capture == 0:
                bh_capture_location = setupdiskblackholes.setup_disk_blackholes_location(rng, 1, outer_capture_radius)
                bh_capture_mass = setupdiskblackholes.setup_disk_blackholes_masses(rng, 1, mode_mbh_init, max_initial_bh_mass, mbh_powerlaw_index)
                bh_capture_spin = setupdiskblackholes.setup_disk_blackholes_spins(rng, 1, mu_spin_distribution, sigma_spin_distribution)
                bh_capture_spin_angle = setupdiskblackholes.setup_disk_blackholes_spin_angles(rng, 1, bh_capture_spin)
                bh_capture_gen = 1
                bh_capture_orb_ecc = 0.0
                bh_capture_orb_incl = 0.0
                #print("CAPTURED BH",bh_capture_location,bh_capture_mass,bh_capture_spin,bh_capture_spin_angle)
                # Append captured BH to existing singleton arrays. Assume prograde and 1st gen BH.
                prograde_bh_locations = np.append(prograde_bh_locations,bh_capture_location) 
                prograde_bh_masses = np.append(prograde_bh_masses,bh_capture_mass)
                prograde_bh_spins = np.append(prograde_bh_spins,bh_capture_spin)
                prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,bh_capture_spin_angle) 
                prograde_bh_generations = np.append(prograde_bh_generations,bh_capture_gen)
                prograde_bh_orb_ecc = np.append(prograde_bh_orb_ecc,bh_capture_orb_ecc)
                prograde_bh_orb_incl = np.append(prograde_bh_orb_incl,bh_capture_orb_incl)
            #Iterate the time step
            time_passed = time_passed + timestep
            #Print time passed every 10 timesteps for now
            time_iteration_tracker = 10.0*timestep
            if time_passed % time_iteration_tracker == 0:
                print("Time passed=",time_passed)

        #End Loop of Timesteps at Final Time, end all changes & print out results
    
        print("End Loop!")
        print("Final Time (yrs) = ",time_passed)
        if verbose:
            print("BH locations at Final Time")
            print(prograde_bh_locations)
        print("Number of binaries = ",bin_index)
        print("Total number of mergers = ",number_of_mergers)
        print("Mergers", merged_bh_array.shape)
        print("Nbh_disk",n_bh)
    
        if True and number_of_mergers > 0: #verbose:
                print(merged_bh_array[:,:number_of_mergers].T)

        iteration_save_name = f"run{iteration_zfilled_str}/{opts.fname_output_mergers}"
        np.savetxt(os.path.join(work_directory, iteration_save_name), merged_bh_array[:,:number_of_mergers].T, header=merger_field_names)

        # Add mergers to population array including the iteration number
        iteration_row = np.repeat(iteration, number_of_mergers)
        merged_bh_array_pop.append(np.concatenate((iteration_row[np.newaxis], merged_bh_array[:,:number_of_mergers])).T)

     # save all mergers from Monte Carlo
    merger_pop_field_names = "iter " + merger_field_names # Add "Iter" to field names
    population_header = f"Initial seed: {opts.seed}\n{merger_pop_field_names}" # Include initial seed
    basename, extension = os.path.splitext(opts.fname_output_mergers)
    population_save_name = f"{basename}_population{extension}"
    np.savetxt(os.path.join(work_directory, population_save_name), np.vstack(merged_bh_array_pop), header=population_header)

if __name__ == "__main__":
    main()
