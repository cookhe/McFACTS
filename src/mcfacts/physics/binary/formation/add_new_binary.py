import numpy as np
from mcfacts.mcfacts_random_state import rng
from mcfacts.physics.binary.evolve.evolve import gw_strain_freq
from astropy import constants as const
from astropy import units as u
from astropy.units import cds


def add_to_binary_obj(blackholes_binary, blackholes_pro, bh_pro_id_num_binary, id_start_val, fraction_bin_retro, smbh_mass, agn_redshift):
    """
    Create new BH binaries with appropriate parameters.
    
    We take the semi-maj axis, masses, spins, spin angles and generations
    from the relevant singletons, found in hillsphere.binary_check2, and sort
    those parameters into disk_bins_bhbh. We then ADD additional parameters
    relevant only for binaries, including semi-major axis of the binary,
    semi-major axis of the orbit of the center of mass of the binary around
    the SMBH, a flag to permit or suppress retrograde binaries, eventually
    eccentricity and inclination.

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
        id_num_1 = bh_pro_id_num_binary[0, i]
        id_num_2 = bh_pro_id_num_binary[1, i]

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
        d_obs = redshift_d_obs_dict[agn_redshift].to(u.meter)
        strain = (4/d_obs) * rg_chirp * np.power(np.pi * nu_gw * rg_chirp / const.c, 2./3.)

        gw_strain[i] = np.array([strain.value])

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
