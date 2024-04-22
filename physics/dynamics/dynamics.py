import numpy as np

import scipy

def circular_singles_encounters_prograde(rng, mass_smbh, prograde_bh_locations, prograde_bh_masses, disk_surf_model, disk_aspect_ratio_model, bh_orb_ecc, timestep, crit_ecc, de):
    """"Return array of modified singleton BH orbital eccentricities perturbed by encounters within f*R_Hill, where f is some fraction/multiple of Hill sphere radius R_H
    
    Assume encounters between damped BH (e<e_crit) and undamped BH (e>e_crit) are the only important ones for now. 
    Since the e<e_crit population is the most likely BBH merger source.
    
    1, find those orbiters with e<e_crit and their
        associated semi-major axes a_circ =[a_circ1, a_circ2, ..] and masses m_circ =[m_circ1,m_circ2, ..].

    2, calculate orbital timescales for a_circ1 and a_i and N_orbits/timestep. 
        For example, since
        T_orb =2pi sqrt(a^3/GM_smbh)
        and a^3/GM_smbh = (10^3r_g)^3/GM_smbh = 10^9 (a/10^3r_g)^3 (GM_smbh/c^2)^3/GM_smbh 
                    = 10^9 (a/10^3r_g)^3 (G M_smbh/c^3)^2 
                    
        So, T_orb   = 2pi 10^4.5 (a/10^3r_g)^3/2 GM_smbh/c^3 
                    = 2pi 10^4.5 (a/10^3r_g)^3/2 (6.7e-11*2e38/(3e8)^3) 
                    = 2pi 10^4.5 (a/10^3r_g)^3/2 (13.6e27/27e24) 
                    = pi 10^7.5  (a/10^3r_g)^3/2
                    ~ 3yr (a/10^3r_g)^3/2 (M_smbh/10^8Msun)
        i.e. Orbit~3yr at 10^3r_g around a 10^8Msun SMBH. 
        Therefore in a timestep=1.e4yr, a BH at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.

    3, among population of orbiters with e>e_crit, 
        find those orbiters (a_i,e_i) where a_i*(1-e_i)< a_circ1,j <a_i*(1-e_i) for all members a_circ1,j of the circularized population 
        so we can test for possible interactions.
    
    4, calculate mutual Hill sphere R_H of candidate binary (a_circ1,j ,a_i).
    
    5, calculate ratio of 2R_H of binary to size of circular orbit, or (2R_H/2pi a_circ1,j)
        Hill sphere possible on both crossing inwards and outwards once per orbit, 
        so 2xHill sphere =4R_H worth of circular orbit will have possible encounter. 
        Thus, (4R_H/2pi a_circ1)= odds that a_circ1 is in the region of cross-over per orbit.
        For example, for BH at a_circ1 = 1e3r_g, 
            R_h = a_circ1*(m_circ1 + m_i/3M_smbh)^1/3
                = 0.004a_circ1 (m_circ1/10Msun)^1/3 (m_i/10Msun)^1/3 (M_SMBH/1e8Msun)^-1/3
        then
            ratio (4R_H/2pi a_circ1) = 0.008/pi ~ 0.0026 
            (ie around 1/400 odds that BH at a_circ1 is in either area of crossing)         
            
    6, calculate number of orbits of a_i in 1 timestep. 
        If e.g. N_orb(a_i)/timestep = 200 orbits per timestep of 10kyr, then 
        probability of encounter = (200orbits/timestep)*(4R_H/2pi a_circ1) ~ 0.5, 
                                or 50% odds of an encounter on this timestep between (a_circ1,j , a_i).
        If probability > 1, set probability = 1.
    7, draw a random number from the uniform [0,1] distribution and 
        if rng < probability of encounter, there is an encounter during the timestep
        if rng > probability of encounter, there is no encounter during the timestep

    8, if encounter:
        Take energy (de) from high ecc. a_i and give energy (de) to a_circ1,j
        de is average fractional energy change per encounter.
            So, a_circ1,j ->(1+de)a_circ1,j.    
                e_circ1,j ->(crit_ecc + de)
            and
                a_i       ->(1-de)a_i
                e_i       ->(1-de)e_i              
        Could be that average energy in gas-free cluster case is  
        assume average energy transfer = 20% perturbation (from Sigurdsson & Phinney 1993). 
        
        Further notes for self:
        sigma_ecc = sqrt(ecc^2 + incl^2)v_kep so if incl=0 deg (for now)
        En of ecc. interloper = 1/2 m_i sigma_ecc^2.
            Note: Can also use above logic for binary encounters except use binary binding energy instead.
        
        or later could try 
            Deflection angle defl = tan (defl) = dV_perp/V = 2GM/bV^2 kg^-1 m^3 s^-2 kg / m (m s^-1)^2
        so de/e =2GM/bV^2 = 2 G M_bin/0.5R_hill*sigma^2
        and R_hill = a_circ1*(M_bin/3M_smbh)^1/3 and sigma^2 =ecc^2*v_kep^2
        So de/e = 4GM_bin/a_circ1(M_bin/3M_smbh)^1/3 ecc^2 v_kep^2 
        and v_kep = sqrt(GM_smbh/a_i)
        So de/e = 4GM_bin^(2/3)M_smbh^1/3 a_i/a_circ1 ecc^2 GM_smbh = 4(M_bin/M_smbh)^(2/3) (a_i/a_circ1)(1/ecc^2)
        where V_rel = sigma say and b=R_H = a_circ1 (q/3)^1/3
        So defl = 2GM/ a_circ1(q/3)^2/3 ecc^2 10^14 (m/s)^2 (R/10^3r_g)^-1
                = 2 6.7e-11 2.e31/  
        !!Note: when doing this for binaries. 
            Calculate velocity of encounter compared to a_bin.
            If binary is hard ie GM_bin/a_bin > m3v_rel^2 then:
            harden binary 
                a_bin -> a_bin -da_bin and
            new binary eccentricity 
                e_bin -> e_bin + de  
            and give  da_bin worth of binding energy to extra eccentricity of m3.
            If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
            soften binary 
                a_bin -> a_bin + da_bin and
            new binary eccentricity 
                e_bin -> e_bin + de
            and remove da_bin worth of binary energy from eccentricity of m3. 


        Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    bh_orb_ecc : float array
        orbital eccentricity of singleton BH     
    timestep : float
        size of timestep in years
    de : float
        average energy change per strong encounter    
        
        Returns
    -------
    bh_new_loc_orb_ecc : float array
        updated bh locations and orbital eccentricities perturbed by dynamics
        """
    # get surface density function, or deal with it if only a float
    if isinstance(disk_surf_model, float):
        disk_surface_density = disk_surf_model
    else:
        disk_surface_density = disk_surf_model(prograde_bh_locations)
    # ditto for aspect ratio
    if isinstance(disk_aspect_ratio_model, float):
        disk_aspect_ratio = disk_aspect_ratio_model
    else:
        disk_aspect_ratio = disk_aspect_ratio_model(prograde_bh_locations)
    #Set up new_bh_orb_ecc
    new_bh_orb_ecc=np.empty_like(bh_orb_ecc)
    
    #Calculate & normalize all the parameters above in t_damp
    # E.g. normalize q=bh_mass/smbh_mass to 10^-7 
    mass_ratio = prograde_bh_masses/mass_smbh
    
    #normalized_mass_ratio = mass_ratio/10**(-7)
    #normalized_bh_locations = prograde_bh_locations/1.e4
    #normalized_disk_surf_model = disk_surface_density/1.e5
    #normalized_aspect_ratio = disk_aspect_ratio/0.03

    #Assume all incoming eccentricities are prograde (for now)
    prograde_bh_orb_ecc = bh_orb_ecc

    #Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population = np.ma.masked_where(prograde_bh_orb_ecc > crit_ecc, prograde_bh_orb_ecc)
    #Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population = np.ma.masked_where(prograde_bh_orb_ecc < crit_ecc, prograde_bh_orb_ecc)
    #print('Circ prograde',circ_prograde_population)
    #Find the indices of the e<crit_ecc population     
    circ_prograde_population_indices = np.ma.nonzero(circ_prograde_population)
    ecc_prograde_population_indices = np.ma.nonzero(ecc_prograde_population)
    #print('Circ indices',circ_prograde_population_indices)
    #Find their locations and masses
    circ_prograde_population_locations = prograde_bh_locations[circ_prograde_population_indices]
    circ_prograde_population_masses = prograde_bh_masses[circ_prograde_population_indices]
    #print('Circ locations',circ_prograde_population_locations)
    #print('Circ masses',circ_prograde_population_masses)
    #Ecc locs
    ecc_prograde_population_locations = prograde_bh_locations[ecc_prograde_population_indices]
    ecc_prograde_population_masses = prograde_bh_masses[ecc_prograde_population_indices]

    #T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2) 
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = scipy.constants.pi*((circ_prograde_population_locations)**(1.5))*(2.e30*mass_smbh*scipy.constants.G)/(scipy.constants.c**(3.0)*3.15e7) 
    N_circ_orbs_per_timestep = timestep/orbital_timescales_circ_pops
    #print('Orb. timescales (yr)' , orbital_timescales_circ_pops)
    #print('N_circ_orb/timestep' , N_circ_orbs_per_timestep)
    ecc_orb_min = prograde_bh_locations[ecc_prograde_population_indices]*(1.0-prograde_bh_orb_ecc[ecc_prograde_population_indices])
    ecc_orb_max = prograde_bh_locations[ecc_prograde_population_indices]*(1.0+prograde_bh_orb_ecc[ecc_prograde_population_indices])
    #print('min',ecc_orb_min)
    #print('max',ecc_orb_max)
    #print('len(circ_locns)',len(circ_prograde_population_locations))
    num_poss_ints = 0
    num_encounters = 0
    if len(circ_prograde_population_locations) > 0:
            for i in range(0,len(circ_prograde_population_locations)):    
                for j in range (0,len(ecc_prograde_population_locations)):
                    if circ_prograde_population_locations[i] < ecc_orb_max[j] and circ_prograde_population_locations[i] > ecc_orb_min[j]:
                        #print("Possible Interaction!")
                        #print(prograde_bh_locations[j],prograde_bh_orb_ecc[j],ecc_orb_min[j],circ_prograde_population_locations[i],ecc_orb_max[j])
                        #prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                        # r_h = a_circ1(temp_bin_mass/3mass_smbh)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                        temp_bin_mass = circ_prograde_population_masses[i] + ecc_prograde_population_masses[j]
                        bh_smbh_mass_ratio = temp_bin_mass/(3.0*mass_smbh)
                        mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)                        
                        prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                        prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                        if prob_enc_per_timestep > 1:
                            prob_enc_per_timestep = 1
                        random_uniform_number = np.random.uniform(0,1)
                        if random_uniform_number < prob_enc_per_timestep:
                            #print('Encounter!!',random_uniform_number,prob_enc_per_timestep, i, j)
                            #print(circ_prograde_population, prograde_bh_orb_ecc[j], circ_prograde_population_locations)
                            #print(circ_prograde_population_indices,i,circ_prograde_population_indices[0])
                            indx_array = circ_prograde_population_indices[0]
                            #print(prograde_bh_orb_ecc[indx_array[i]])
                            num_encounters = num_encounters + 1
                            # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                            # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                            # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                            if prograde_bh_orb_ecc[indx_array[i]] <= crit_ecc:
                                #print(prograde_bh_orb_ecc[indx_array[i]],prograde_bh_orb_ecc[j])
                                prograde_bh_orb_ecc[indx_array[i]] = de
                                prograde_bh_locations[indx_array[i]] = prograde_bh_locations[indx_array[i]]*(1.0 + de)
                                prograde_bh_orb_ecc[j] = prograde_bh_orb_ecc[j]*(1 - de)
                                prograde_bh_locations[j] = prograde_bh_locations[j]*(1 - de)
                                #print(prograde_bh_orb_ecc[indx_array[i]],prograde_bh_orb_ecc[j])
                        
                        num_poss_ints = num_poss_ints + 1
                #print("Num encounters",i,num_poss_ints,num_encounters)
            num_poss_ints = 0
            num_encounters = 0
    
    prograde_bh_locn_orb_ecc = [[prograde_bh_locations],[prograde_bh_orb_ecc]]    
    return prograde_bh_locn_orb_ecc

def circular_binaries_encounters_ecc_prograde(rng,mass_smbh, prograde_bh_locations, prograde_bh_masses, bh_orb_ecc, timestep, crit_ecc, de,bin_array,bindex):
    """"Return array of modified binary BH separations and eccentricities perturbed by encounters within f*R_Hill, for eccentric singleton population, where f is some fraction/multiple of Hill sphere radius R_H
    Right now assume f=1.
    Logic:  
            0.  Find number of binaries in this timestep given by bindex
            1.  Find the binary center of mass (c.o.m.) and corresponding orbital velocities & binary total masses.
                bin_array[9,:] = bin c.o.m. = [R_bin1_com,R_bin2_com,...]. These are the orbital radii of the bins.
                bin_array[8,;] = bin_separation =[a_bin1,a_bin2,...]
                bin_array[2,:]+bin_array[3,:] = mass of binaries
                bin_array[13,:] = ecc of binary around com
                bin_array[18,:] = orb. ecc of binary com around SMBH
                Keplerian orbital velocity of the bin c.o.m. around SMBH: v_bin,i= sqrt(GM_SMBH/R_bin,i_com)= c/sqrt(R_bin,i_com)
            2.  Calculate the binary orbital time and N_orbits/timestep
                For example, since
                T_orb =2pi sqrt(R_bin_com^3/GM_smbh)
                and R_bin_com^3/GM_smbh = (10^3r_g)^3/GM_smbh = 10^9 (R_bin_com/10^3r_g)^3 (GM_smbh/c^2)^3/GM_smbh 
                    = 10^9 (R_bin_com/10^3r_g)^3 (G M_smbh/c^3)^2 
                    
                So, T_orb   
                    = 2pi 10^4.5 (R_bin_com/10^3r_g)^3/2 GM_smbh/c^3 
                    = 2pi 10^4.5 (R_bin_com/10^3r_g)^3/2 (6.7e-11*2e38/(3e8)^3) 
                    = 2pi 10^4.5 (R_bin_com/10^3r_g)^3/2 (13.6e27/27e24) 
                    = pi 10^7.5  (R_bin_com/10^3r_g)^3/2
                    ~ 3.15 yr (R_bin_com/10^3r_g)^3/2 (M_smbh/10^8Msun)
                i.e. Orbit~3.15yr at 10^3r_g around a 10^8Msun SMBH. 
                Therefore in a timestep=1.e4yr, a binary at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.
            3.  Calculate binding energy of bins = [GM1M2/sep_bin1, GMiMi+1,sep_bin2, ....] where sep_bin1 is in meters and M1,M2 are binary mass components in kg.
            4.  Find those single BH with e>e_crit and their
                associated semi-major axes a_ecc =[a_ecc1, a_ecc2, ..] and masses m_ecc =[m_ecc1,m_ecc2, ..]
                and calculate their average velocities v_ecc = [GM_smbh/a_ecc1, GM_smbh/a_ecc2,...]
            5.  Where (1-ecc_i)*a_ecc_i < R_bin_j_com < (1+ecc_i)*a_ecc_i, interaction possible
            6.  Among candidate encounters, calculate relative velocity of encounter.
                        v_peri,i=sqrt(Gm_ecc,i/a_ecc,i[1+ecc,i/1-ecc,i])
                        v_apo,i =sqrt(Gm_ecc,i/a_ecc,i[1-ecc,i/1+ecc,i])
                        v_ecc,i =sqrt(GM/a_ecc_i)..average Keplerian vel.
                    
                    v_rel = abs(v_bin,i - vecc,i)
            7. Calculate relative K.E. of tertiary, (1/2)m_ecc_i*v_rel_^2     
            8. Compare binding en of binary to K.E. of tertiary.
                Critical velocity for ionization of binary is v_crit, given by:
                    v_crit = sqrt(GM_1M_2(M_1+M_2+M_3)/M_3(M_1+M_2)a_bin)
                If binary is hard ie GM_1M_2/a_bin > m3v_rel^2 then:
                    harden binary 
                        a_bin -> a_bin -da_bin and
                    new binary eccentricity 
                        e_bin -> e_bin + de  
                    and give  +da_bin worth of binding energy (GM_bin/(a_bin -da_bin) - GM_bin/a_bin) 
                    to extra eccentricity ecc_i and a_ecc,i of m_ecc,i.
                    Say average en of encounter is de=0.1 (10%) then binary a_bin shrinks by 10%, ecc_bin is pumped by 10%
                    And a_ecc_i shrinks by 10% and ecc_i also shrinks by 10%
                If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
                    if v_rel (effectively v_infty) > v_crit
                        ionize binary
                            update singleton array with 2 new BH with orbital eccentricity e_crit+de
                            remove binary from binary array
                    else if v_rel < v_crit
                        soften binary 
                            a_bin -> a_bin + da_bin and
                        new binary eccentricity 
                            e_bin -> e_bin + de
                        and remove -da_bin worth of binary energy from eccentricity of m3.
            Note1: Will need to test binary eccentricity each timestep. 
                If bin_ecc> some value (0.9), check for da_bin due to GW bremsstrahlung at pericenter.
            9. As 4, except now include interactions between binaries and circularized BH. This should give us primarily
                hardening encounters as in Leigh+2018, since the v_rel is likely to be small for more binaries.

    Given array of binaries at locations [a_bbh1,a_bbh2] with 
    binary semi-major axes [a_bin1,a_bin2,...] and binary eccentricities [e_bin1,e_bin2,...],
    find all the single BH at locations a_i that within timestep 
        either pass between a_i(1-e_i)< a_bbh1 <a_i(1+e_i)

    Calculate velocity of encounter compared to a_bin.
    If binary is hard ie GM1M2/a_bin > m3v_rel^2 then:
      harden binary to a_bin = a_bin -da_bin and
      new binary eccentricity e_bin = e_bin + de around com and
      new binary orb eccentricity e_orb_com = e_orb_com + de and 
      now give  da_bin worth of binding energy to extra eccentricity of m3.
    If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
      soften binary to a_bin = a_bin + da_bin and
      new binary eccentricity e_bin = e_bin + de
      and take da_bin worth of binary energy from eccentricity of m3. 
    If binary is unbound ie GM_bin/a_bin << m3v_rel^2 then:
      remove binary from binary array
      add binary components m1,m2 back to singleton arrays with new orbital eccentricities e_1,e_2 from energy of encounter.
      Equipartition energy so m1v1^2 =m2 v_2^2 and 
      generate new individual orbital eccentricities e1=v1/v_kep_circ and e_2=v_2/v_kep_circ
      Take energy put into destroying binary from orb. eccentricity of m3.  
        Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    bh_orb_ecc : float array
        orbital eccentricity of singleton BH     
    timestep : float
        size of timestep in years
        
        Returns
    -------
    bh_new_orb_ecc : float array
        updated orbital eccentricities damped by AGN gas
        """
    #Read in binary array properties.
    #Loop through number of binaries (given by bindex) and number of bin properties (given by integer_nbinprop)
    
    #Housekeeping
    #Mass of Sun in kg
    solar_mass = 2.e30

    number_of_binaries = bindex
    # set up 1-d arrays for bin com, masses, separations, velocities of com, orbit time (in yrs), orbits/timestep
    bin_coms = np.zeros(number_of_binaries)
    bin_masses = np.zeros(number_of_binaries)
    bin_separations = np.zeros(number_of_binaries)
    bin_eccentricities = np.zeros(number_of_binaries)
    bin_orbital_eccentricities = np.zeros(number_of_binaries)
    bin_velocities = np.zeros(number_of_binaries)
    bin_orbital_times = np.zeros(number_of_binaries)
    bin_orbits_per_timestep = np.zeros(number_of_binaries)
    bin_binding_energy = np.zeros(number_of_binaries)

    #Read in values of binaries at start of timestep
    for j in range(0, number_of_binaries-1):
        bin_coms[j] = bin_array[9,j]
        bin_masses[j] = bin_array[2,j] + bin_array[3,j]
        bin_separations[j] = bin_array[8,j]
        # Eccentricity of binary around its own center of mass
        bin_eccentricities[j] = bin_array[13,j]
        # Orbital Eccentricity of binary c.o.m. around SMBH
        bin_orbital_eccentricities[j] = bin_array[18,j] 
        # Keplerian binary velocity of c.o.m. around SMBH
        bin_velocities[j] = scipy.constants.c/np.sqrt(bin_coms[j])
        # binary orbital time around SMBH in yrs (3.15yrs at 1000r_g from 1e8 Msun SMBH)
        bin_orbital_times[j] = 3.15*(mass_smbh/1.e8)*(bin_coms[j]/1.e3)**(1.5)
        # number of orbits per timestep
        bin_orbits_per_timestep[j] = timestep/bin_orbital_times[j]
        # binary binding energy (GM1M2/Sep) in Joules where M1,M2 in Kg and Sep in meters
        rg_in_meters = scipy.constants.G*(solar_mass*mass_smbh)/(scipy.constants.c)**2.0
        bin_binding_energy[j] = scipy.constants.G*((solar_mass)**2)*bin_array[2,j]*bin_array[3,j]/(bin_separations[j]*rg_in_meters)


    #Assume all incoming eccentricities are prograde (for now)
    prograde_bh_orb_ecc = bh_orb_ecc
    #Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population = np.ma.masked_where(prograde_bh_orb_ecc < crit_ecc, prograde_bh_orb_ecc)
    #Find the indices of the e<crit_ecc population     
    ecc_prograde_population_indices = np.ma.nonzero(ecc_prograde_population)
    #Find their locations and masses
    ecc_prograde_population_locations = prograde_bh_locations[ecc_prograde_population_indices]
    ecc_prograde_population_masses = prograde_bh_masses[ecc_prograde_population_indices]
    ecc_prograde_population_eccentricities = prograde_bh_orb_ecc[ecc_prograde_population_indices]
    #Find min and max radii around SMBH for eccentric orbiters
    ecc_orb_min = prograde_bh_locations[ecc_prograde_population_indices]*(1.0-prograde_bh_orb_ecc[ecc_prograde_population_indices])
    ecc_orb_max = prograde_bh_locations[ecc_prograde_population_indices]*(1.0+prograde_bh_orb_ecc[ecc_prograde_population_indices])
    #print('min',ecc_orb_min)
    #print('max',ecc_orb_max)
    # Keplerian velocity of ecc prograde orbiter around SMBH (=c/sqrt(a/r_g))
    ecc_velocities = scipy.constants.c/np.sqrt(ecc_prograde_population_locations)

    #Set up number of interactions/encounters
    num_poss_ints = 0
    num_encounters = 0
    if number_of_binaries > 0:
            for i in range(0, number_of_binaries-1):    
                for j in range (0,len(ecc_prograde_population_locations)):
                    #If binary com orbit lies inside eccentric orbit [min,max] radius
                    # i.e. does R_m3_minimum lie inside R_bin_maximum and does R_m3_max lie outside R_bin_minimum 
                    if (1.0-bin_orbital_eccentricities[i])*bin_coms[i] < ecc_orb_max[j] and (1.0+bin_orbital_eccentricities[i])*bin_coms[i] > ecc_orb_min[j]:
                        # Make a temporary Hill sphere treating binary + ecc interloper as a 'binary' = M_1+M_2+M_3
                        # r_h = a_circ1(temp_bin_mass/3mass_smbh)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                        temp_bin_mass = bin_masses[i] + ecc_prograde_population_masses[j]
                        bh_smbh_mass_ratio = temp_bin_mass/(3.0*mass_smbh)
                        mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)                        
                        prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                        prob_enc_per_timestep = prob_orbit_overlap * bin_orbits_per_timestep[i]
                        # critical velocity for ionization given by
                        # v_crit =sqrt(GM_1M_2(M_1+M_2+M_3)/M_3(M_1+M_2)a_bin) =sqrt(GM_1M_2(M_bin+M_3)/M_3M_bin a_bin)
                        # & divide by 1.e3 to get v_crit in km/s
                        v_crit_kms = np.sqrt(scipy.constants.G*bin_array[2,i]*bin_array[3,i]*temp_bin_mass*solar_mass/(ecc_prograde_population_masses[j]*bin_masses[i]*bin_separations[i]*rg_in_meters))/1.e3 

                        if prob_enc_per_timestep > 1:
                            prob_enc_per_timestep = 1
                        random_uniform_number = np.random.uniform(0,1)
                        if random_uniform_number < prob_enc_per_timestep:
                            #Perturb *this* ith binary depending on how hard it already is.
                            num_encounters = num_encounters + 1                            
                            #Line below is key. Right now it say only dynamically perturb bin if its circularized. If not circularized, leave alone.
                            #If comment out this line then all binaries are perturbed.
                            #if bin_orbital_eccentricities[i] <= crit_ecc:
                                # Find relative velocity of interloper in km/s so divide by 1.e3
                            rel_vel_kms = abs(bin_velocities[i] - ecc_velocities[j])/1.e3
                            rel_vel_ms = abs(bin_velocities[i] - ecc_velocities[j])
                                # K.E. of interloper
                            ke_interloper = 0.5*ecc_prograde_population_masses[j]*solar_mass*(rel_vel_ms**2.0)
                            hard = bin_binding_energy[i] - ke_interloper                                
                            if hard > 0:
                                # Binary is hard w.r.t interloper
                                # Change binary parameters; decr separation, incr ecc around com and orb_ecc 
                                #print("Hardening Ke3,BEb, bin dr,dr(1-de),e_b,e_b(1+de),e_orb_bin,e_orb_bin(1+de)", ke_interloper, bin_binding_energy[i], bin_separations[i],bin_separations[i]*(1-de),bin_eccentricities[i],bin_eccentricities[i]*(1+de),bin_orbital_eccentricities[i],bin_orbital_eccentricities[i]*(1+de))
                                bin_separations[i] = bin_separations[i]*(1-de)
                                bin_eccentricities[i] = bin_eccentricities[i]*(1+de)
                                bin_orbital_eccentricities[i] = bin_orbital_eccentricities[i]*(1+de)
                                #Change interloper parameters; increase a_ecc, increase e_ecc
                                ecc_prograde_population_locations[j] = ecc_prograde_population_locations[j]*(1+de)
                                ecc_prograde_population_eccentricities[j] = ecc_prograde_population_eccentricities[j]*(1+de)

                            if hard < 0:
                                #Binary is soft w.r.t. interloper
                                #Check to see if binary is ionized
                                #if rel_vel_kms > v_crit_kms:
                                #    print("Ionize bin, rel_vel, v_crit", rel_vel_kms, v_crit_kms)
                                    
                                # Change binary parameters; incr bin separation, decr ecc around com, incr orb_ecc
                                #print("Softening bin Ke3,BEb, dr,dr(1-de),e_b,e_b(1+de),e_orb_bin,e_orb_bin(1+de),rel_vel,v_crit", ke_interloper, bin_binding_energy[i], bin_separations[i],bin_separations[i]*(1+de),bin_eccentricities[i],bin_eccentricities[i]*(1-de),bin_orbital_eccentricities[i],bin_orbital_eccentricities[i]*(1+de),rel_vel_kms, v_crit_kms)
                                bin_separations[i] = bin_separations[i]*(1+de)
                                bin_eccentricities[i] = bin_eccentricities[i]*(1-de)
                                bin_orbital_eccentricities[i] = bin_orbital_eccentricities[i]*(1+de)
                                #Change interloper parameters; decrease a_ecc, decrease e_ecc
                                ecc_prograde_population_locations[j] = ecc_prograde_population_locations[j]*(1-de)
                                ecc_prograde_population_eccentricities[j] = ecc_prograde_population_eccentricities[j]*(1-de)


                                
                        num_poss_ints = num_poss_ints + 1
                #print("Num encounters",i,num_poss_ints,num_encounters)
            num_poss_ints = 0
            num_encounters = 0
        

            
    #Write new binary parameters (new separations, new orb. ecc around SMBH, new ecc around c.o.m.)
    for j in range(0,number_of_binaries-1):
        bin_array[8,j] = bin_separations[j]
        bin_array[18,j] = bin_orbital_eccentricities[j]
        bin_array[13,j] = bin_eccentricities[j]

    # TO DO: ALSO return new array of singletons with changed params.

    return bin_array

def circular_binaries_encounters_circ_prograde(rng,mass_smbh, prograde_bh_locations, prograde_bh_masses, bh_orb_ecc, timestep, crit_ecc, de,bin_array,bindex):
    """"Return array of modified binary BH separations and eccentricities perturbed by encounters within f*R_Hill, for circularized singleton population, where f is some fraction/multiple of Hill sphere radius R_H
    Right now assume f=1.
    Logic:  
            0.  Find number of binaries in this timestep given by bindex
            1.  Find the binary center of mass (c.o.m.) and corresponding orbital velocities & binary total masses.
                bin_array[9,:] = bin c.o.m. = [R_bin1_com,R_bin2_com,...]. These are the orbital radii of the bins.
                bin_array[8,;] = bin_separation =[a_bin1,a_bin2,...]
                bin_array[2,:]+bin_array[3,:] = mass of binaries
                bin_array[13,:] = ecc of binary around com
                bin_array[18,:] = orb. ecc of binary com around SMBH
                Keplerian orbital velocity of the bin c.o.m. around SMBH: v_bin,i= sqrt(GM_SMBH/R_bin,i_com)= c/sqrt(R_bin,i_com)
            2.  Calculate the binary orbital time and N_orbits/timestep
                For example, since
                T_orb =2pi sqrt(R_bin_com^3/GM_smbh)
                and R_bin_com^3/GM_smbh = (10^3r_g)^3/GM_smbh = 10^9 (R_bin_com/10^3r_g)^3 (GM_smbh/c^2)^3/GM_smbh 
                    = 10^9 (R_bin_com/10^3r_g)^3 (G M_smbh/c^3)^2 
                    
                So, T_orb   
                    = 2pi 10^4.5 (R_bin_com/10^3r_g)^3/2 GM_smbh/c^3 
                    = 2pi 10^4.5 (R_bin_com/10^3r_g)^3/2 (6.7e-11*2e38/(3e8)^3) 
                    = 2pi 10^4.5 (R_bin_com/10^3r_g)^3/2 (13.6e27/27e24) 
                    = pi 10^7.5  (R_bin_com/10^3r_g)^3/2
                    ~ 3.15 yr (R_bin_com/10^3r_g)^3/2 (M_smbh/10^8Msun)
                i.e. Orbit~3.15yr at 10^3r_g around a 10^8Msun SMBH. 
                Therefore in a timestep=1.e4yr, a binary at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.
            3.  Calculate binding energy of bins = [GM1M2/sep_bin1, GMiMi+1,sep_bin2, ....] where sep_bin1 is in meters and M1,M2 are binary mass components in kg.
            4.  Find those single BH with e>e_crit and their
                associated semi-major axes a_ecc =[a_ecc1, a_ecc2, ..] and masses m_ecc =[m_ecc1,m_ecc2, ..]
                and calculate their average velocities v_ecc = [GM_smbh/a_ecc1, GM_smbh/a_ecc2,...]
            5.  Where (1-ecc_i)*a_ecc_i < R_bin_j_com < (1+ecc_i)*a_ecc_i, interaction possible
            6.  Among candidate encounters, calculate relative velocity of encounter.
                        v_peri,i=sqrt(Gm_ecc,i/a_ecc,i[1+ecc,i/1-ecc,i])
                        v_apo,i =sqrt(Gm_ecc,i/a_ecc,i[1-ecc,i/1+ecc,i])
                        v_ecc,i =sqrt(GM/a_ecc_i)..average Keplerian vel.
                    
                    v_rel = abs(v_bin,i - vecc,i)
            7. Calculate relative K.E. of tertiary, (1/2)m_ecc_i*v_rel_^2     
            8. Compare binding en of binary to K.E. of tertiary.
                Critical velocity for ionization of binary is v_crit, given by:
                    v_crit = sqrt(GM_1M_2(M_1+M_2+M_3)/M_3(M_1+M_2)a_bin)
                If binary is hard ie GM_1M_2/a_bin > m3v_rel^2 then:
                    harden binary 
                        a_bin -> a_bin -da_bin and
                    new binary eccentricity 
                        e_bin -> e_bin + de  
                    and give  +da_bin worth of binding energy (GM_bin/(a_bin -da_bin) - GM_bin/a_bin) 
                    to extra eccentricity ecc_i and a_ecc,i of m_ecc,i.
                    Say average en of encounter is de=0.1 (10%) then binary a_bin shrinks by 10%, ecc_bin is pumped by 10%
                    And a_ecc_i shrinks by 10% and ecc_i also shrinks by 10%
                If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
                    if v_rel (effectively v_infty) > v_crit
                        ionize binary
                            update singleton array with 2 new BH with orbital eccentricity e_crit+de
                            remove binary from binary array
                    else if v_rel < v_crit
                        soften binary 
                            a_bin -> a_bin + da_bin and
                        new binary eccentricity 
                            e_bin -> e_bin + de
                        and remove -da_bin worth of binary energy from eccentricity of m3.
            Note1: Will need to test binary eccentricity each timestep. 
                If bin_ecc> some value (0.9), check for da_bin due to GW bremsstrahlung at pericenter.
            9. As 4, except now include interactions between binaries and circularized BH. This should give us primarily
                hardening encounters as in Leigh+2018, since the v_rel is likely to be small for more binaries.

    Given array of binaries at locations [a_bbh1,a_bbh2] with 
    binary semi-major axes [a_bin1,a_bin2,...] and binary eccentricities [e_bin1,e_bin2,...],
    find all the single BH at locations a_i that within timestep 
        either pass between a_i(1-e_i)< a_bbh1 <a_i(1+e_i)

    Calculate velocity of encounter compared to a_bin.
    If binary is hard ie GM1M2/a_bin > m3v_rel^2 then:
      harden binary to a_bin = a_bin -da_bin and
      new binary eccentricity e_bin = e_bin + de around com and
      new binary orb eccentricity e_orb_com = e_orb_com + de and 
      now give  da_bin worth of binding energy to extra eccentricity of m3.
    If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
      soften binary to a_bin = a_bin + da_bin and
      new binary eccentricity e_bin = e_bin + de
      and take da_bin worth of binary energy from eccentricity of m3. 
    If binary is unbound ie GM_bin/a_bin << m3v_rel^2 then:
      remove binary from binary array
      add binary components m1,m2 back to singleton arrays with new orbital eccentricities e_1,e_2 from energy of encounter.
      Equipartition energy so m1v1^2 =m2 v_2^2 and 
      generate new individual orbital eccentricities e1=v1/v_kep_circ and e_2=v_2/v_kep_circ
      Take energy put into destroying binary from orb. eccentricity of m3.  
        Parameters
    ----------
    mass_smbh : float
        mass of supermassive black hole in units of solar masses
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar masses
    disk_surf_model : function
        returns AGN gas disk surface density in kg/m^2 given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    disk_aspect_ratio_model : function
        returns AGN gas disk aspect ratio given a distance from the SMBH in r_g
        can accept a simple float (constant), but this is deprecated
    bh_orb_ecc : float array
        orbital eccentricity of singleton BH     
    timestep : float
        size of timestep in years
        
        Returns
    -------
    bh_new_orb_ecc : float array
        updated orbital eccentricities damped by AGN gas
        """
    #Read in binary array properties.
    #Loop through number of binaries (given by bindex) and number of bin properties (given by integer_nbinprop)
    
    # Housekeeping
    # Mass of Sun in kg
    solar_mass = 2.e30
    # Magnitude of energy change to drive binary to merger in ~2 interactions in a strong encounter. Say de_strong=0.9
    de_strong =0.9

    number_of_binaries = bindex
    # set up 1-d arrays for bin com, masses, separations, velocities of com, orbit time (in yrs), orbits/timestep
    bin_coms = np.zeros(number_of_binaries)
    bin_masses = np.zeros(number_of_binaries)
    bin_separations = np.zeros(number_of_binaries)
    bin_eccentricities = np.zeros(number_of_binaries)
    bin_orbital_eccentricities = np.zeros(number_of_binaries)
    bin_velocities = np.zeros(number_of_binaries)
    bin_orbital_times = np.zeros(number_of_binaries)
    bin_orbits_per_timestep = np.zeros(number_of_binaries)
    bin_binding_energy = np.zeros(number_of_binaries)

    #Read in values of binaries at start of timestep
    for j in range(0, number_of_binaries-1):
        bin_coms[j] = bin_array[9,j]
        bin_masses[j] = bin_array[2,j] + bin_array[3,j]
        bin_separations[j] = bin_array[8,j]
        # Eccentricity of binary around its own center of mass
        bin_eccentricities[j] = bin_array[13,j]
        # Orbital Eccentricity of binary c.o.m. around SMBH
        bin_orbital_eccentricities[j] = bin_array[18,j] 
        # Keplerian binary velocity of c.o.m. around SMBH
        bin_velocities[j] = scipy.constants.c/np.sqrt(bin_coms[j])
        # binary orbital time around SMBH in yrs (3.15yrs at 1000r_g from 1e8 Msun SMBH)
        bin_orbital_times[j] = 3.15*(mass_smbh/1.e8)*(bin_coms[j]/1.e3)**(1.5)
        # number of orbits per timestep
        bin_orbits_per_timestep[j] = timestep/bin_orbital_times[j]
        # binary binding energy (GM1M2/Sep) in Joules where M1,M2 in Kg and Sep in meters
        rg_in_meters = scipy.constants.G*(solar_mass*mass_smbh)/(scipy.constants.c)**2.0
        bin_binding_energy[j] = scipy.constants.G*((solar_mass)**2)*bin_array[2,j]*bin_array[3,j]/(bin_separations[j]*rg_in_meters)


    #Assume all incoming eccentricities are prograde (for now)
    prograde_bh_orb_ecc = bh_orb_ecc
    #Find the e< crit_ecc population. These are the interlopers w. low encounter vel that can harden the circularized population
    circ_prograde_population = np.ma.masked_where(prograde_bh_orb_ecc > crit_ecc, prograde_bh_orb_ecc)
    #Find the indices of the e<crit_ecc population     
    circ_prograde_population_indices = np.ma.nonzero(circ_prograde_population)
    #Find their locations and masses
    circ_prograde_population_locations = prograde_bh_locations[circ_prograde_population_indices]
    circ_prograde_population_masses = prograde_bh_masses[circ_prograde_population_indices]
    circ_prograde_population_eccentricities = prograde_bh_orb_ecc[circ_prograde_population_indices]
    #Find min and max radii around SMBH for eccentric orbiters
    ecc_orb_min = prograde_bh_locations[circ_prograde_population_indices]*(1.0-prograde_bh_orb_ecc[circ_prograde_population_indices])
    ecc_orb_max = prograde_bh_locations[circ_prograde_population_indices]*(1.0+prograde_bh_orb_ecc[circ_prograde_population_indices])
    #print('min',ecc_orb_min)
    #print('max',ecc_orb_max)
    # Keplerian velocity of ecc prograde orbiter around SMBH (=c/sqrt(a/r_g))
    circ_velocities = scipy.constants.c/np.sqrt(circ_prograde_population_locations)

    #Set up number of interactions/encounters
    num_poss_ints = 0
    num_encounters = 0
    if number_of_binaries > 0:
            for i in range(0, number_of_binaries-1):    
                for j in range (0,len(circ_prograde_population_locations)):
                    #If binary com orbit lies inside circ orbit [min,max] radius
                    # i.e. does R_m3_minimum lie inside R_bin_maximum and does R_m3_max lie outside R_bin_minimum 
                    if (1.0-bin_orbital_eccentricities[i])*bin_coms[i] < ecc_orb_max[j] and (1.0+bin_orbital_eccentricities[i])*bin_coms[i] > ecc_orb_min[j]:
                        # Make a temporary Hill sphere treating binary + ecc interloper as a 'binary' = M_1+M_2+M_3
                        # r_h = a_circ1(temp_bin_mass/3mass_smbh)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                        temp_bin_mass = bin_masses[i] + circ_prograde_population_masses[j]
                        bh_smbh_mass_ratio = temp_bin_mass/(3.0*mass_smbh)
                        mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)                        
                        prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                        prob_enc_per_timestep = prob_orbit_overlap * bin_orbits_per_timestep[i]
                        # critical velocity for ionization given by
                        # v_crit =sqrt(GM_1M_2(M_1+M_2+M_3)/M_3(M_1+M_2)a_bin) =sqrt(GM_1M_2(M_bin+M_3)/M_3M_bin a_bin)
                        # & divide by 1.e3 to get v_crit in km/s
                        v_crit_kms = np.sqrt(scipy.constants.G*bin_array[2,i]*bin_array[3,i]*temp_bin_mass*solar_mass/(circ_prograde_population_masses[j]*bin_masses[i]*bin_separations[i]*rg_in_meters))/1.e3 

                        if prob_enc_per_timestep > 1:
                            prob_enc_per_timestep = 1
                        random_uniform_number = np.random.uniform(0,1)
                        if random_uniform_number < prob_enc_per_timestep:
                            #Perturb *this* ith binary depending on how hard it already is.
                            num_encounters = num_encounters + 1                            
                            #Line below is key. Right now it says only dynamically perturb bin if its circularized. If not circularized, leave alone.
                            #If comment out this line then all binaries are perturbed.
                            #if bin_orbital_eccentricities[i] <= crit_ecc:
                                # Find relative velocity of interloper in km/s so divide by 1.e3
                            rel_vel_kms = abs(bin_velocities[i] - circ_velocities[j])/1.e3
                            rel_vel_ms = abs(bin_velocities[i] - circ_velocities[j])
                                # K.E. of interloper
                            ke_interloper = 0.5*circ_prograde_population_masses[j]*solar_mass*(rel_vel_ms**2.0)
                            hard = bin_binding_energy[i] - ke_interloper                                
                            if hard > 0:
                                # Binary is hard w.r.t interloper
                                # Change binary parameters; decr separation, incr ecc around com and orb_ecc 
                                # print("HARDEN bin KE3,BEb, dr,dr(1-de),e_b,e_b(1+de),e_orb_bin,e_orb_bin(1+de)", ke_interloper, bin_binding_energy[i], bin_separations[i],bin_separations[i]*(1-de_strong),bin_eccentricities[i],bin_eccentricities[i]*(1+de_strong),bin_orbital_eccentricities[i],bin_orbital_eccentricities[i]*(1+de))
                                bin_separations[i] = bin_separations[i]*(1-de_strong)
                                bin_eccentricities[i] = bin_eccentricities[i]*(1+de_strong)
                                bin_orbital_eccentricities[i] = bin_orbital_eccentricities[i]*(1+de)
                                #Change interloper parameters; increase a_ecc, increase e_ecc
                                circ_prograde_population_locations[j] = circ_prograde_population_locations[j]*(1+de)
                                circ_prograde_population_eccentricities[j] = circ_prograde_population_eccentricities[j]*(1+de)

                            if hard < 0:
                                #Binary is soft w.r.t. interloper
                                #Check to see if binary is ionized
                                #if rel_vel_kms > v_crit_kms:
                                #    print("IONIZE bin, rel_vel, v_crit", rel_vel_kms, v_crit_kms)
                                    
                                # Change binary parameters; incr bin separation, decr ecc around com, incr orb_ecc
                                #print("SOFTEN KE3,BEb, bin dr,dr(1+de),e_b,e_b(1+de),e_orb_bin,e_orb_bin(1+de),rel_vel,v_crit", ke_interloper, bin_binding_energy[i], bin_separations[i],bin_separations[i]*(1+de),bin_eccentricities[i],bin_eccentricities[i]*(1-de),bin_orbital_eccentricities[i],bin_orbital_eccentricities[i]*(1+de),rel_vel_kms, v_crit_kms)
                                bin_separations[i] = bin_separations[i]*(1+de)
                                bin_eccentricities[i] = bin_eccentricities[i]*(1-de)
                                bin_orbital_eccentricities[i] = bin_orbital_eccentricities[i]*(1+de)
                                #Change interloper parameters; decrease a_ecc, decrease e_ecc
                                circ_prograde_population_locations[j] = circ_prograde_population_locations[j]*(1-de)
                                circ_prograde_population_eccentricities[j] = circ_prograde_population_eccentricities[j]*(1-de)


                                
                        num_poss_ints = num_poss_ints + 1
                #print("Num encounters",i,num_poss_ints,num_encounters)
            num_poss_ints = 0
            num_encounters = 0
        

            
    #Write new binary parameters (new separations, new orb. ecc around SMBH, new ecc around c.o.m.)
    for j in range(0,number_of_binaries-1):
        bin_array[8,j] = bin_separations[j]
        bin_array[18,j] = bin_orbital_eccentricities[j]
        bin_array[13,j] = bin_eccentricities[j]

    
    #TO DO: Also return array of modified circularized orbiters.

    return bin_array

def bin_spheroid_encounter(mass_smbh, timestep, bin_array, time_passed, bindex, mbh_powerlaw_index, mode_mbh_init):
    """ Use Leigh+18 to figure out the rate at which spheroid encounters happen to binaries embedded in the disk
    Binaries at small disk radii encounter spheroid objects at high rate, particularly early on in the disk lifetime
    However, orbits at those small radii get captured quickly by the disk.
     
    From Fig.1 in Leigh+18, Rate of sph. encounter = 20/Myr at t=0, normalized to a_bin=1AU, R_disk=10^3r_g or 0.2/10kyr timestep 
    Within 1Myr, for a dense model disk (e.g. Sirko & Goodman), most of those inner stellar orbits have been captured by the disk.
    So rate of sph. encounter ->0/Myr at t=1Myr since those orbits are gone (R<10^3r_g; assuming approx circular orbits!) for SG disk model
    For TQM disk model, rate of encounter slightly lower but non-zero.

    So, inside R_com<10^3r_g: 
    Assume: Rate of encounter = 0.2 (timestep/10kyr)^-1 (R_com/10^3r_g)^-1 (a_bin/1r_gM8)^-2
    Generate random number from uniform [0,1] distribution and if <0.2 (normalized to above condition) then encounter
    
    Encounter rt starts at = 0.2 (timestep/10kyr)^-1 (R_com/10^3r_g)^-1 (a_bin/1r_gM8)^-2 at t=0
    decreases to          = 0(timestep/10kyr)^-1 (R_com/10^3r_g)^-1 (a_bin/1r_gM8)^-2 (time_passed/1Myr)
    at R<10^3r_g.
    Outside: R_com>10^3r_g
    Normalize to rate at (R_com/10^4r_g) so that rate is non-zero at R_com=[1e3,1e4]r_g after 1Myr.
    Decrease rate with time, but ensure it goes to zero at R_com<1.e3r_g.

    So, rate of sph. encounter = 2/Myr at t=0, normalized to a_bin=1AU, R_disk=10^4r_g which is equivalently
    Encounter rate = 0.02 (timestep/10kyr)^-1 (R_com/10^4r_g)^-1 (a_bin/1r_gM8)^2
    Drop this by an order of magnitude over 1Myr.
    Encounter rate = 0.02 (timestep/10kyr)^-1 (R_com/10^4r_g)^-1 (a_bin/1r_gM8)^2 (time_passed/10kyr)^-1/2   
    so ->0.002 after a Myr
    For R_com < 10^3r_g:
        if time_passed <=1Myr
            Encounter rt = 0.2*(1-(1Myr/time_passed))(timestep/10kyr)^{-1}(R_com/10^3r_g)^-1 (a_bin/1r_gM8)^2
        if time_passed >1Myr
            Encounter rt = 0
    For R_com > 10^3r_g: 
        Encounter rt = 0.02 * (timestep/10kyr)^-1 (R_com/10^4r_g)^-1 (a_bin/1r_gM8)^2 (time_passed/10kyr)^-1/2 
    
    Return corrected binary with spin angles projected onto new L_bin.
    
    Assume typical interaction mass and inclination angle. 
    Binary orbital angular momentum is
        L_bin =M_bin*v_orb_bin X R_com
    Spheroid orbital angular momentum is
        L3=m3*v3 X R3 
    where m3,v3,R3 are the mass, velocity and semi-major axis of tertiary encounter. 
    Draw m3 from IMF random distrib. 
    Draw R3 from uniform distribution[100,2000]r_g say. v_3= c/sqrt(R_3)
    Ratio of L3/Lbin =(m3/M_bin)*sqrt(R3/R_com)

    """
    number_of_binaries = bindex
    # set up 1-d arrays for bin com, masses, separations
    bin_coms = np.zeros(number_of_binaries)
    bin_masses = np.zeros(number_of_binaries)
    bin_separations = np.zeros(number_of_binaries)   
    bin_velocities = np.zeros(number_of_binaries)

    #Units of r_g normalized to 1AU around a 10^8Msun SMBH
    dist_in_rg_m8 = 1.0*(1.0e8/mass_smbh)

    #Read in values of binaries at start of timestep
    for j in range(0, number_of_binaries-1):
        bin_coms[j] = bin_array[9,j]
        bin_masses[j] = bin_array[2,j] + bin_array[3,j]
        bin_separations[j] = bin_array[8,j]
        # Keplerian binary velocity of c.o.m. around SMBH
        bin_velocities[j] = scipy.constants.c/np.sqrt(bin_coms[j])

    for i in range(0,number_of_binaries-1):
        #Calculate encounter rate for each binary based on com location, binary size and time passed
        if bin_coms[i] < 1.e3:
            if time_passed <= 1.e6:
                enc_rate = 0.2*(1.0-(time_passed/1.e6))*(bin_separations[i]/dist_in_rg_m8)**(2.0)/((timestep/1.e4)*(bin_coms[i]/1.e3))
            if time_passed >1.e6:
                enc_rate = 0.0
        if bin_coms[i] > 1.e3:
                enc_rate = 0.02*(bin_separations[i]/dist_in_rg_m8)**(2.0)/((timestep/1.e4)*(bin_coms[i]/1.e4)*np.sqrt(time_passed/1.e4))

        #Based on est encounter rate, calculate if binary actually has a spheroid encounter
        random_uniform_number = np.random.uniform(0,1)
        if random_uniform_number < enc_rate:
            #print("SPHEROID INTERACTION!")
            #print("Enc rt., Rnd #, bin_com,time_passed:",enc_rate, random_uniform_number, bin_coms[i]/1.e4, bin_separations[i], dist_in_rg_m8, time_passed)  

            #Generate random interloper with semi-major axis btwn [100,2000]r_g
            interloper_outer_radius = 2000.0
            random_uniform_number2 = np.random.uniform(0,1)
            spheroid_bh_radius = interloper_outer_radius*random_uniform_number2
            #Generate random interloper mass from IMF
            spheroid_bh_mass = (np.random.pareto(mbh_powerlaw_index,1)+1)*mode_mbh_init
            #print("R3,m3",spheroid_bh_radius,spheroid_bh_mass)
            #Compare orbital angular momentum for Interloper and Binary
            #Ratio of L3/Lbin =(m3/M_bin)*sqrt(R3/R_com)
            L_ratio = (spheroid_bh_mass/bin_masses[i])*np.sqrt(spheroid_bh_radius/bin_coms[i])
            #print("L ratio",L_ratio)

    return bin_array