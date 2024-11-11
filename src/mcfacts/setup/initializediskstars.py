from mcfacts.setup import setupdiskstars, setupdiskblackholes
from mcfacts.setup import diskstars_hillspheremergers
from mcfacts.objects.agnobject import AGNStar
import numpy as np
from mcfacts.mcfacts_random_state import rng


def init_single_stars(opts, id_start_val=None):

    # Generate initial number of stars
    star_num_initial = setupdiskstars.setup_disk_stars_num(
            opts.nsc_mass,
            opts.nsc_ratio_bh_num_star_num,
            opts.nsc_ratio_bh_mass_star_mass,
            opts.disk_star_scale_factor,
            opts.nsc_radius_outer,
            opts.nsc_density_index_outer,
            opts.smbh_mass,
            opts.disk_radius_outer,
            opts.disk_aspect_ratio_avg,
            opts.nsc_radius_crit,
            opts.nsc_density_index_inner,
        )
    print("num stars initial", star_num_initial)

    # Generate initial masses for the initial number of stars, pre-Hill sphere mergers
    masses_initial = setupdiskstars.setup_disk_stars_masses(star_num=star_num_initial,
                                                            disk_star_mass_min_init=opts.disk_star_mass_min_init,
                                                            disk_star_mass_max_init=opts.disk_star_mass_max_init,
                                                            nsc_imf_star_powerlaw_index=opts.nsc_imf_star_powerlaw_index)

    # Generating star locations in an x^2 distribution
    #x_vals = rng.uniform(low=0.0, high=1, size=star_num_initial)
    #r_locations_initial = np.sqrt(x_vals)
    #r_locations_initial_scaled = r_locations_initial*opts.disk_radius_trap

    r_locations_initial_scaled = setupdiskstars.setup_disk_stars_orb_a(star_num_initial, opts.disk_radius_outer, opts.disk_inner_stable_circ_orb)

    # Sort the mass and location arrays by the location array
    sort_idx = np.argsort(r_locations_initial_scaled)
    r_locations_initial_sorted = r_locations_initial_scaled[sort_idx]
    masses_initial_sorted = masses_initial[sort_idx]
    masses_stars, orbs_a_stars = diskstars_hillspheremergers.hillsphere_mergers(n_stars=star_num_initial,
                                                                                masses_initial_sorted=masses_initial_sorted,
                                                                                orbs_a_initial_sorted=r_locations_initial_sorted,
                                                                                min_initial_star_mass=opts.disk_star_mass_min_init,
                                                                                disk_radius=opts.disk_radius_trap,
                                                                                smbh_mass=opts.smbh_mass,
                                                                                P_m=1.35,
                                                                                P_r=1.)
    star_num = len(masses_stars)

    if (opts.flag_initial_stars_BH_immortal == 0):
        # Stars over disk_star_initial_mass_cutoff will be held at disk_star_initial_mass_cutoff and be immortal
        masses_stars[masses_stars > opts.disk_star_initial_mass_cutoff] = opts.disk_star_initial_mass_cutoff

    #star_radius = setupdiskstars.setup_disk_stars_radius(masses_stars)
    star_spin = setupdiskstars.setup_disk_stars_spins(star_num, opts.nsc_star_spin_dist_mu, opts.nsc_star_spin_dist_sigma)
    star_spin_angle = setupdiskstars.setup_disk_stars_spin_angles(star_num, star_spin)
    star_orb_inc = setupdiskstars.setup_disk_stars_inclination(star_num)
    #star_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(rng,star_num)
    star_orb_arg_periapse = setupdiskstars.setup_disk_stars_arg_periapse(star_num)
    if opts.flag_orb_ecc_damping == 1:
        star_orb_ecc = setupdiskstars.setup_disk_stars_eccentricity_uniform(star_num)
    else:
        star_orb_ecc = setupdiskstars.setup_disk_stars_circularized(star_num, opts.disk_bh_pro_orb_ecc_crit)

    star_X, star_Y, star_Z = setupdiskstars.setup_disk_stars_comp(star_num=star_num,
                                                                  star_ZAMS_metallicity=opts.nsc_star_metallicity_z_init,
                                                                  star_ZAMS_helium=opts.nsc_star_metallicity_y_init)

    stars = AGNStar(mass=masses_stars,
                    spin=star_spin,
                    spin_angle=star_spin_angle,
                    orb_a=orbs_a_stars,
                    orb_inc=star_orb_inc,
                    orb_ecc=star_orb_ecc,
                    #orb_ang_mom= star_orb_ang_mom,
                    orb_arg_periapse=star_orb_arg_periapse,
                    #radius=star_radius,
                    star_X=star_X,
                    star_Y=star_Y,
                    star_Z=star_Z,
                    galaxy=np.zeros(star_num),
                    time_passed=np.zeros(star_num),
                    smbh_mass=opts.smbh_mass,
                    id_start_val=id_start_val,
                    star_num=star_num)

    return (stars, star_num)
