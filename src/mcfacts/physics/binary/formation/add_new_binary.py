import numpy as np
import scipy
from mcfacts.mcfacts_random_state import rng
from mcfacts.objects.agnobject import obj_to_binary_bh_array
from astropy import constants as const
from astropy import units as u
from astropy.units import cds

def add_to_binary_array_old(
        disk_bins_bhbh,
        disk_bh_pro_orbs_a,
        disk_bh_pro_masses,
        disk_bh_pro_spins,
        disk_bh_pro_spin_angles,
        disk_bh_pro_gens,
        disk_bin_bhbh_pro_indices,
        bindex,
        fraction_bin_retro,
        smbh_mass,
        agn_redshift
        ):
    """Create new BH binaries with appropriate parameters.
    
    We take the semi-maj axis, masses, spins, spin angles and generations
    from the relevant singletons, found in hillsphere.binary_check2, and sort
    those parameters into disk_bins_bhbh. We then ADD additional parameters
    relevant only for binaries, including semi-major axis of the binary,
    semi-major axis of the orbit of the center of mass of the binary around
    the SMBH, a flag to permit or suppress retrograde binaries, eventually
    eccentricity and inclination.

    Parameters
    ----------
    disk_bins_bhbh : [21, bindex] mixed array
        binary black hole array, multi-dimensional;
            [0,j]: float
                location of object 1 at formation of binary (distance from SMBH in
                r_g)
            [1,j]: float
                location of object 2 at formation of binary (distance from SMBH in
                r_g)
            [2,j]: float
                mass of obj 1 at time t in units of solar masses
            [3,j]: float
                mass of obj 2 at time t in units of solar masses
            [4,j]: float
                dimensionless spin magnitude of obj 1 at time t
            [5,j]: float
                dimensionless spin magnitude of obj 2 at time t
            [6,j]: float
                spin angle of obj 1 wrt disk gas in radians at time t
            [7,j]: float
                spin angle of obj 2 wrt disk gas in radians at time t
            [8,j]: float
                binary semi-major axis around c.o.m. in units of r_g of SMBH
            [9,j]: float
                binary center of mass location wrt SMBH in r_g
            [10,j]: float
                time to merger through GW alone (not set here)
            [11,j]: int
                merger flag = -2 if merging this timestep, else = 0 (not set here)
            [12,j]: float
                time of merger if binary has already merged (not set here)
            [13,j]: float
                binary eccentricity around binary center of mass
            [14,j]: int
                generation of obj 1 (1=natal black hole, no prior mergers)
            [15,j]: int
                generation of obj 2 (1=natal black hole, no prior mergers)
            [16,j]: int
                binary angular momentum switch +1/-1 for pro/retrograde
            [17,j]: float
                binary orbital inclination
            [18,j]: float
                binary orbital eccentricity of binary center of mass around SMBH
            [19,j]: float
                GW frequency of binary
                nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
            [20,j]: float
                GW strain of binary
                h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
                where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
                For local distances, approx 
                d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
                assume 1Mpc = 3.1e22m.
                From Ned Wright's calculator
                (https://www.astro.ucla.edu/~wright/CosmoCalc.html)
                (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
    disk_bh_pro_orbs_a : float array
        locations of prograde singleton BH at start of timestep in units of
        gravitational radii (r_g=GM_SMBH/c^2)
    disk_bh_pro_masses : float array
        mass of prograde singleton BH at start of timestep in units of solar
        masses
    disk_bh_pro_spins : float array
        dimensionless spin magnitude of prograde singleton BH at start of
        timestep
    disk_bh_pro_spin_angles : float array
       spin angle of prograde singleton BH wrt the gas disk angular momentum
       in radians at start of time step
    disk_bh_pro_gens : int array
        generation of prograde singleton BH (1=no previous merger history)
    disk_bin_bhbh_pro_indices : [2,N] int array
        array of indices corresponding to locations in disk_bh_pro_orbs_a,
        disk_bh_pro_masses, disk_bh_pro_spins, disk_bh_pro_spin_angles, and
        disk_bh_pro_gens which corresponds to binaries that form in this
        timestep.
        it has a length of the number of binaries to form (N) and a width of 2.
    bindex : int
        counter for length of disk_bins_bhbh before adding new binaries
    fraction_bin_retro : float
        fraction of binaries which form retrograde (wrt to the disk gas)
        around their own center of mass.
        = 0.0 turns all retrograde BBH at formation into prograde BBH. 
        = 0.5 half of the binaries will be retrograde
        = 1.0 all binaries will be retrograde.
    smbh_mass : float
        mass of supermassive black hole in units of solar masses
    agn_redshift : float
        redshift of the AGN, used to set d_obs

    Returns
    -------
    disk_bins_bhbh : [21, bindex+N] float array
        as for input, but updated (thus longer), to include newly formed
        binaries
    """
    # get length of current binary array before any additions
    # --we will need to do this, but for now we are using bindex bc too
    #   complex to fix now and appears many places
    # find number of new binaries based on indices from hillsphere.binary_check
    num_new_bins = np.shape(disk_bin_bhbh_pro_indices)[1]
    #print("num_new_bins",num_new_bins)

    redshift_d_obs_dict = {0.1: 421,
                           0.5: 1909}

    # If there are new binaries, actually form them!
    if num_new_bins > 0:
        # send close encounter indices to new array
        array_of_indices = disk_bin_bhbh_pro_indices
        bincount = 0
        # for all the new binaries that need to be created
        for j in range(bindex, bindex + num_new_bins):
            # for each member of the binary
            for i in range(0,2):
                # pick the [N,0] or [N,1] index for member 1 and member 2 of
                # binary N
                if num_new_bins == 1:
                    thing1 = array_of_indices[i]
                else:
                    thing1 = array_of_indices[i][bincount]
                
                disk_bins_bhbh[i,j] = disk_bh_pro_orbs_a[thing1]
                disk_bins_bhbh[i+2,j] = disk_bh_pro_masses[thing1]
                disk_bins_bhbh[i+4,j] = disk_bh_pro_spins[thing1]
                disk_bins_bhbh[i+6,j] = disk_bh_pro_spin_angles[thing1]
                # For new binary create initial binary semi-major axis
                temp_loc_1 = disk_bins_bhbh[0,j]
                temp_loc_2 = disk_bins_bhbh[1,j]
                temp_bin_separation = np.abs(temp_loc_1 - temp_loc_2)
                disk_bins_bhbh[8,j] = temp_bin_separation
                # Binary c.o.m.= location_1 + separation*M_2/(M_1+M_2)
                temp_mass_1 = disk_bins_bhbh[2,j]
                temp_mass_2 = disk_bins_bhbh[3,j]
                temp_bin_mass = temp_mass_1 + temp_mass_2
                disk_bins_bhbh[9,j] = temp_loc_1 + \
                    (temp_bin_separation*temp_mass_2/temp_bin_mass)
                # Set up binary eccentricity around its own center of mass.
                # Draw uniform value btwn [0,1]
                disk_bins_bhbh[13,j] = rng.random()
                # Set up binary member generations
                disk_bins_bhbh[i+14,j] = disk_bh_pro_gens[thing1]
                # Set up bin orb. ang. mom.
                # (randomly +1 (pro) or -1(retrograde))
                # random number between [0,1]
                random_uniform_number = rng.random()
                # Generate a random number between [-retro,1-retro] where
                # retro is the fraction of BBH that are retrograde.
                # For default retro =0.1, range is [-0.1,0.9] and 90% of BBH
                # are prograde.
                bh_initial_orb_ang_mom = random_uniform_number - \
                    fraction_bin_retro
                # If retro =0, range = [0,1] and set L_bbh = +1.
                if fraction_bin_retro == 0:
                    bh_initial_orb_ang_mom = 1
                # If retro =0.1 (default), range = [-0.1,0.9].
                # If range <0, L_BBH = -1; if range >0, L_BBH = +1
                if fraction_bin_retro > 0:
                    if bh_initial_orb_ang_mom < 0:
                        bh_initial_orb_ang_mom = -1
                    if bh_initial_orb_ang_mom > 0:
                        bh_initial_orb_ang_mom = 1

                disk_bins_bhbh[16,j] = bh_initial_orb_ang_mom                
                # Set up binary inclination (in units radians). Will want this
                # to be pi radians if retrograde.
                disk_bins_bhbh[17,j] = 0
                # Set up binary orbital eccentricity of com around SMBH.
                # Assume initially v.small (e~0.01)
                disk_bins_bhbh[18,j] = 0.01
                # Set up binary GW frequency
                # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
                #1rg =1AU=1.5e11m for 1e8Msun
                rg = 1.5e11*(smbh_mass/1.e8)
                bin_sep_meters = temp_bin_separation*rg
                temp_mass_1_kg = ((temp_mass_1*cds.Msun).to(u.kg)).value #2.e30*temp_mass_1
                temp_mass_2_kg = ((temp_mass_2*cds.Msun).to(u.kg)).value #2.e30*temp_mass_2
                temp_bin_mass_kg = ((temp_bin_mass*cds.Msun).to(u.kg)).value #2.e30*temp_bin_mass
                m_chirp = ((temp_mass_1_kg*temp_mass_2_kg)**(3./5.))/ \
                    (temp_bin_mass_kg**(1/5))
                rg_chirp = (scipy.constants.G * m_chirp)/ \
                    (scipy.constants.c**(2.0))
                if bin_sep_meters < rg_chirp:
                    bin_sep_meters = rg_chirp
                
                # compute GW frequency & strain for each binary
                nu_gw = (1.0/scipy.constants.pi)*np.sqrt(
                    temp_bin_mass_kg *
                    scipy.constants.G /
                    (bin_sep_meters**(3.0))
                    )
                disk_bins_bhbh[19,j] = nu_gw
                Mpc = 3.1e22 # This is terrible use astropy
                #d_obs = 421*Mpc
                redshift = redshift_d_obs_dict[agn_redshift]
                d_obs = (redshift*u.Mpc).to(u.meter).value  #1909*Mpc

                strain = (4/d_obs)*rg_chirp * \
                    (np.pi*nu_gw*rg_chirp/scipy.constants.c)**(2./3.)
                disk_bins_bhbh[20,j] = strain
            bincount = bincount + 1

    return disk_bins_bhbh


def add_to_binary_obj(blackholes_binary, blackholes_pro, bh_pro_id_num_binary, id_start_val, fraction_bin_retro, smbh_mass, agn_redshift):
    """_summary_

    Parameters
    ----------
    blackholes_binary : AGNBinaryBlackHole
        binary black holes, will add new binaries
    blackholes_pro : AGNBlackHole
        prograde black holes
    bh_pro_id_num_binary : numpy array of ints
        ID numbers for the prograde blackholes that will form binaries
    id_start_val : int
        starting value for the ID numbers (add 1 to ensure it's unique)
    fraction_bin_retro : float
        fraction of binaries which form retrograde (wrt to the disk gas)
        around their own center of mass.
        = 0.0 turns all retrograde BBH at formation into prograde BBH. 
        = 0.5 half of the binaries will be retrograde
        = 1.0 all binaries will be retrograde.
    smbh_mass : float
        mass of SMBH in units of Msun
    agn_redshift : float
        redshift of the AGN, used to set d_obs

    Returns
    -------
    blackholes_binary : AGNBinaryBlackHole
        binary black hole object with new binaries added
    id_nums : numpy array of ints
        ID numbers of the new binary black holes
    """

    redshift_d_obs_dict = {0.1: 421*u.Mpc,
                           0.5: 1909*u.Mpc}

    bin_num = bh_pro_id_num_binary.shape[1]
    id_nums = np.arange(id_start_val+1, id_start_val + 1 + bin_num, 1)
    #print("add_new_binary/id_start_val",id_start_val)
    #print("add_new_binary/id_nums",id_nums)
    orb_a_1 = np.zeros(bin_num)
    orb_a_2 = np.zeros(bin_num)
    mass_1 = np.zeros(bin_num)
    mass_2 = np.zeros(bin_num)
    spin_1 = np.zeros(bin_num)
    spin_2 = np.zeros(bin_num)
    spin_angle_1 = np.zeros(bin_num)
    spin_angle_2 = np.zeros(bin_num)
    bin_sep = np.zeros(bin_num)
    bin_orb_a = np.zeros(bin_num)
    time_to_merger_gw = np.zeros(bin_num)
    flag_merging = np.zeros(bin_num)
    time_merged = np.zeros(bin_num)
    # Set up binary eccentricity around its own center of mass.
    # Draw uniform value btwn [0,1]
    bin_ecc = rng.random(bin_num)
    gen_1 = np.zeros(bin_num)
    gen_2 = np.zeros(bin_num)
    bin_orb_ang_mom = np.zeros(bin_num)
    # Set up binary inclination (in units radians). Will want this
    # to be pi radians if retrograde.
    bin_orb_inc = np.zeros(bin_num)
    # Set up binary orbital eccentricity of com around SMBH.
    # Assume initially v.small (e~0.01)
    bin_orb_ecc = np.full(bin_num, 0.01)
    gw_freq = np.zeros(bin_num)
    gw_strain = np.zeros(bin_num)
    galaxy = np.zeros(bin_num)

    for i in range(bin_num):
        id_num_1 = bh_pro_id_num_binary[0,i]
        id_num_2 = bh_pro_id_num_binary[1,i]

        #print("add_new_binary.add_to_binary_obj/bh_pro_id_num_binary[0,i]",bh_pro_id_num_binary[0,i])
        #print("add_new_binary.add_to_binary_obj/bh_pro_id_num_binary[1,i]",bh_pro_id_num_binary[1,i])
        #print("add_new_binary.add_to_binary_obj/mass_1",blackholes_pro.at_id_num(id_num_1, "mass"))
        #print("add_new_binary.add_to_binary_obj/mass_2",blackholes_pro.at_id_num(id_num_2, "mass"))

        mass_1[i] = blackholes_pro.at_id_num(id_num_1, "mass")
        mass_2[i] = blackholes_pro.at_id_num(id_num_2, "mass")
        orb_a_1[i] = blackholes_pro.at_id_num(id_num_1, "orb_a")
        orb_a_2[i] = blackholes_pro.at_id_num(id_num_2, "orb_a")
        spin_1[i] = blackholes_pro.at_id_num(id_num_1, "spin")
        spin_2[i] = blackholes_pro.at_id_num(id_num_2, "spin")
        spin_angle_1[i] = blackholes_pro.at_id_num(id_num_1, "spin_angle")
        spin_angle_2[i] = blackholes_pro.at_id_num(id_num_2, "spin_angle")
        gen_1[i] = blackholes_pro.at_id_num(id_num_1, "gen")
        gen_2[i] = blackholes_pro.at_id_num(id_num_1, "gen")
        bin_sep[i] = np.abs(orb_a_1[i] - orb_a_2[i])
        galaxy[i] = blackholes_pro.at_id_num(id_num_1, "galaxy")

        # Binary c.o.m.= location_1 + separation*M_2/(M_1+M_2)
        bin_orb_a[i] = orb_a_1[i] + ((bin_sep[i] * mass_2[i]) / (mass_1[i] + mass_2[i]))

        gen_1[i] = blackholes_pro.at_id_num(id_num_1, "gen")
        gen_2[i] = blackholes_pro.at_id_num(id_num_2, "gen")

        # Set up bin orb. ang. mom.
        # (randomly +1 (pro) or -1(retrograde))
        # random number between [0,1]
        # If fraction_bin_retro =0, range = [0,1] and set L_bbh = +1.
        if fraction_bin_retro == 0:
            bin_orb_ang_mom[i] = 1.
        else:
            bin_orb_ang_mom[i] = (2.0*np.around(rng.random())) - 1.0

        # Calculate binary GW frequency
        # nu_gw = 1/pi *sqrt(GM_bin/a_bin^3)
        # 1rg =1AU=1.5e11m for 1e8Msun
        rg_to_meters = 1.5e11 * (smbh_mass / 1.e8) * u.meter
        bin_sep_meters = bin_sep[i] * rg_to_meters
        mass_1_units = (mass_1[i] * cds.Msun).to(u.kg)
        mass_2_units = (mass_2[i] * cds.Msun).to(u.kg)
        mass_bin = mass_1_units + mass_2_units
        mass_chirp = (np.power(mass_1_units * mass_2_units, 3./5.) / (np.power(mass_bin, 1./5.))).to(u.kg)
        rg_chirp = ((const.G * mass_chirp) / np.power(const.c, 2)).to(u.meter)

        if (bin_sep_meters < rg_chirp):
            bin_sep_meters = rg_chirp

        nu_gw = ((1.0/np.pi)*np.sqrt(
                    mass_bin *
                    const.G /
                    (bin_sep_meters**(3.0)))).to(u.Hz)

        gw_freq[i] = np.array([nu_gw.value])

        # Calculate GW strain
        # h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
        #         where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
        #         For local distances, approx 
        #         d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
        #         assume 1Mpc = 3.1e22m.
        #         From Ned Wright's calculator
        #         (https://www.astro.ucla.edu/~wright/CosmoCalc.html)
        #         (z=0.1)=421Mpc. (z=0.5)=1909 Mpc
        #d_obs = 421*u.Mpc
        d_obs = redshift_d_obs_dict[agn_redshift].to(u.meter)
        strain = (4/d_obs) * rg_chirp * np.power(np.pi * nu_gw * rg_chirp / const.c, 2./3.)

        gw_strain[i] = np.array([strain.value])

    #print(mass_1)

    blackholes_binary.add_binaries(new_orb_a_1=orb_a_1,
                                     new_orb_a_2=orb_a_2,
                                     new_mass_1=mass_1,
                                     new_mass_2=mass_2,
                                     new_spin_1=spin_1,
                                     new_spin_2=spin_2,
                                     new_spin_angle_1=spin_angle_1,
                                     new_spin_angle_2=spin_angle_2,
                                     new_bin_sep=bin_sep,
                                     new_bin_orb_a=bin_orb_a,
                                     new_time_to_merger_gw=time_to_merger_gw,
                                     new_flag_merging=flag_merging,
                                     new_time_merged=time_merged,
                                     new_bin_ecc=bin_ecc,
                                     new_gen_1=gen_1,
                                     new_gen_2=gen_2,
                                     new_bin_orb_ang_mom=bin_orb_ang_mom,
                                     new_bin_orb_inc=bin_orb_inc,
                                     new_bin_orb_ecc=bin_orb_ecc,
                                     new_gw_freq=gw_freq,
                                     new_gw_strain=gw_strain,
                                     new_id_num=id_nums,
                                     new_galaxy=galaxy)

    return (blackholes_binary, id_nums)
