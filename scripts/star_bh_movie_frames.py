import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath-snapshots",
                        default="gal000",
                        type=str, help="path to galaxy")
    parser.add_argument("--fname-stars-merge",
                        default="output_stars_merged.dat",
                        type=str, help="output merged stars file")
    parser.add_argument("--fname-stars-explode",
                        default="output_stars_exploded.dat",
                        type=str, help="output exploded stars file")
    parser.add_argument("--fname-stars-unbound",
                        default="output_stars_unbound.dat",
                        type=str, help="output unbound stars file")
    parser.add_argument("--fname-bh-unbound",
                        default="output_mergers_unbound.dat",
                        type=str, help="output unbound bh file")
    parser.add_argument("--num-timesteps",
                        default=60,
                        type=int, help="number of timesteps")
    parser.add_argument("--timestep-duration-yr",
                        default=10000,
                        type=int, help="timestep length in  years")
    parser.add_argument("--plots-directory",
                        default="gal000",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.fpath_snapshots)
    assert os.path.isdir(opts.fpath_snapshots)
    assert os.path.isdir(opts.plots_directory)
    return opts


immortal_star_cutoff = 298
# Higher zorder means on top
star_zorder = 5
bh_zorder = 10
bbh_zorder = 15


def plotting(stars_orba, stars_mass, mask_immortal, starsin_orba, starsin_mass, starsretro_orba, starsretro_mass,
             bh_orba, bh_mass, bhin_orba, bhin_mass, bhretro_orba, bhretro_mass, bbh_orba, bbh_mass,
             bbh_merge_orba, bbh_merge_mass, star_merge_orba, star_merge_mass, star_explode_orba, star_explode_mass,
             bh_unbound_orba, bh_unbound_mass, star_unbound_orb_a, star_unbound_mass,
             timestep, mask_label, nomask_label, bh_label, bbh_label, bbh_merge_label, star_merge_label, star_explode_label,
             bh_unbound_label, star_unbound_label, save_name):
    plt.scatter(stars_orba[~mask_immortal], stars_mass[~mask_immortal], marker="o", edgecolor='#DA627D', facecolor='None', zorder=star_zorder)
    plt.scatter(stars_orba[mask_immortal], stars_mass[mask_immortal], marker="o", edgecolor='#450920', facecolor='None', zorder=star_zorder)
    plt.scatter(starsin_orba, starsin_mass, marker="o", edgecolor='#DA627D', facecolor='None', zorder=star_zorder)
    plt.scatter(starsretro_orba, starsretro_mass, marker="o", edgecolor='#DA627D', facecolor='None', zorder=star_zorder)
    plt.scatter(bh_orba, bh_mass, marker="o", edgecolor="darkgoldenrod", facecolor="None", zorder=bh_zorder)
    plt.scatter(bhin_orba, bhin_mass, marker="o", edgecolor="darkgoldenrod", facecolor="None", zorder=bh_zorder)
    plt.scatter(bhretro_orba, bhretro_mass, marker="o", edgecolor="darkgoldenrod", facecolor="None", zorder=bh_zorder)
    plt.scatter(bbh_orba, bbh_mass, marker="o", edgecolors="#005f73", facecolor="None", zorder=bbh_zorder)
    plt.scatter(bbh_merge_orba, bbh_merge_mass, marker="D", edgecolor="k", facecolor="None", zorder=bbh_zorder)
    plt.scatter(star_merge_orba, star_merge_mass, marker="d", edgecolor="k", facecolor="None", zorder=star_zorder)
    plt.scatter(star_explode_orba, star_explode_mass, marker="X", edgecolor="k", facecolor="None", zorder=star_zorder)
    plt.scatter(bh_unbound_orba, bh_unbound_mass, marker="^", edgecolor="k", facecolor="None", zorder=bh_zorder)
    plt.scatter(star_unbound_orb_a, star_unbound_mass, marker=">", edgecolor="k", facecolor="None", zorder=star_zorder)
    plt.scatter(0, -10, label=nomask_label, color="#DA627D")
    plt.scatter(0, -10, label=mask_label, color="#450920")
    plt.scatter(0, -10, label=bh_label, color="darkgoldenrod")
    plt.scatter(0, -10, label=bbh_label, color="#005f73")
    plt.scatter(0, -10, marker="D", label=bbh_merge_label, color="k")
    plt.scatter(0, -10, marker="d", label=star_merge_label, color="k")
    plt.scatter(0, -10, marker="X", label=star_explode_label, color="k")
    plt.scatter(0, -10, marker="^", label=bh_unbound_label, color="k")
    plt.scatter(0, -10, marker=">", label=star_unbound_label, color="k")

    plt.vlines(700, -5, 500, colors='dimgrey', label="Trap radius", zorder=0)

    plt.legend(title=rf"{(int(timestep))/1e2} Myr", loc="upper left", frameon=False)
    plt.ylabel(r"Mass [$M_\odot$]")
    plt.xlabel(r"Semi-major axis [R$_{\mathrm {g, SMBH}}$]")
    plt.ylim(-5, 310)
    plt.xlim(20, 55000)

    plt.xscale("log")
    plt.savefig(save_name + "_log.png", dpi=300)

    plt.close()


def generate_plots(fpath, num_timesteps, timestep_duration_yr, bbh_merge_data, star_explode_data, star_merge_data, bh_unbound_data, star_unbound_data):
    bh_single_pro = []
    bh_inner = []
    bh_retro = []
    bh_binary = []
    star_pro = []
    star_inner = []
    star_retro = []
    bbh_merge = []
    star_merge = []
    star_explode = []
    bh_unbound = []
    star_unbound = []

    for i in range(0, num_timesteps):

        bh_single_pro.append(fpath + f"/output_bh_single_pro_{i}.dat")
        bh_inner.append(fpath + f"/output_bh_single_inner_disk_{i}.dat")
        bh_retro.append(fpath + f"/output_bh_single_retro_{i}.dat")
        bh_binary.append(fpath + f"/output_bh_binary_{i}.dat")

        star_pro.append(fpath + f"/output_stars_single_pro_{i}.dat")
        star_inner.append(fpath + f"/output_stars_single_inner_disk_{i}.dat")
        star_retro.append(fpath + f"/output_stars_single_retro_{i}.dat")

        bbh_merge.append(bbh_merge_data[bbh_merge_data[:, 2] == i * timestep_duration_yr])
        star_merge.append(star_merge_data[star_merge_data[:, 1] == i * timestep_duration_yr])
        star_explode.append(star_explode_data[star_explode_data[:, 1] == i * timestep_duration_yr])
        bh_unbound.append(bh_unbound_data[bh_unbound_data[:, 1] == i * timestep_duration_yr])
        star_unbound.append(star_unbound_data[star_unbound_data[:, 1] == i * timestep_duration_yr])

    zip_data = zip(enumerate(star_pro), enumerate(star_inner), enumerate(star_retro),
                   enumerate(bh_single_pro), enumerate(bh_inner), enumerate(bh_retro),
                   enumerate(bh_binary),
                   bbh_merge, star_merge, star_explode, bh_unbound, star_unbound)
    for (star_stuff, star_in_stuff, star_retro_stuff,
         bh_stuff, bh_in_stuff, bh_retro_stuff,
         bbh_stuff,
         bbh_merge_stuff, star_merge_stuff, star_explode_stuff,
         bh_unbound_stuff, star_unbound_stuff) in zip_data:

        t_star, fname_star = star_stuff
        i_bh, fname_bh = bh_stuff
        i_bbh, fname_bbh = bbh_stuff
        i_s, fname_star_inner = star_in_stuff
        i_s, fname_star_retro = star_retro_stuff
        i_b, fname_bh_inner = bh_in_stuff
        i_b, fname_bh_retro = bh_retro_stuff
        t_star = f"{t_star:02d}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            star = np.loadtxt(fname_star)
            star_in = np.loadtxt(fname_star_inner)
            star_retro = np.loadtxt(fname_star_retro)
            bh = np.loadtxt(fname_bh)
            bh_in = np.loadtxt(fname_bh_inner)
            bh_retro = np.loadtxt(fname_bh_retro)
            bbh = np.loadtxt(fname_bbh)
        mask_immortal = star[:, 2] == immortal_star_cutoff
        if bbh.size == 0:
            bbh_orba = None
            bbh_mass = None
        elif bbh.size == 22:
            bbh_orba = bbh[9]
            bbh_mass = bbh[2] + bbh[3]
        else:
            bbh_orba = bbh[:, 9]
            bbh_mass = bbh[:, 2] + bbh[:, 3]

        if star_in.size == 0:
            star_in_orba = None
            star_in_mass = None
        elif star_in.size == 16:
            star_in_orba = star_in[1]
            star_in_mass = star_in[2]
        else:
            star_in_orba = star_in[:, 1]
            star_in_mass = star_in[:, 2]

        if bh_in.size == 0:
            bh_in_orba = None
            bh_in_mass = None
        elif bh_in.size == 14:
            bh_in_orba = bh_in[1]
            bh_in_mass = bh_in[2]
        else:
            bh_in_orba = bh_in[:, 1]
            bh_in_mass = bh_in[:, 2]
        plotting(star[:, 1], star[:, 2], mask_immortal, star_in_orba, star_in_mass, star_retro[:, 1], star_retro[:, 2],
                 bh[:, 1], bh[:, 2], bh_in_orba, bh_in_mass, bh_retro[:, 1], bh_retro[:, 2],
                 bbh_orba, bbh_mass,
                 bbh_merge_stuff[:, 0], bbh_merge_stuff[:, 1],
                 star_merge_stuff[:, 2], star_merge_stuff[:, 3],
                 star_explode_stuff[:, 2], star_explode_stuff[:, 3],
                 bh_unbound_stuff[:, 2], bh_unbound_stuff[:, 3],
                 star_unbound_stuff[:, 2], star_unbound_stuff[:, 3],
                 t_star,
                 r"Immortal star ($298$ $M_\odot$)", "Single star", "Single BH", "BBH", "BBH merger", "Star merger", "Supernova", "Unbound BH", "Unbound star",
                 fpath + f"/orba_mass_movie_timestep_{t_star}")


def main():

    opts = arg()

    # Load BBH mergers, star mergers, star explosions
    # BBH cols are bin_orb_a, mass_final, time_merged
    bbh_merge = np.loadtxt(opts.fpath_snapshots + "/output_mergers.dat", usecols=(1, 2, 14))
    # star explode cols are galaxy, time_sn, orb_a_star, mass_star
    star_explode = np.loadtxt(opts.fname_stars_explode, usecols=(0, 1, 2, 3))
    # star merge cols are galaxy, time_merged, orb_a_final, mass_final
    star_merge = np.loadtxt(opts.fname_stars_merge, usecols=(0, 1, 2, 3))
    bh_unbound = np.loadtxt(opts.fname_bh_unbound, usecols=(0, 1, 2, 3))
    star_unbound = np.loadtxt(opts.fname_stars_unbound, usecols=(0, 1, 2, 3))

    # Cut out other galaxies
    star_explode = star_explode[star_explode[:, 0] == 0]
    star_merge = star_merge[star_merge[:, 0] == 0]
    bh_unbound = bh_unbound[bh_unbound[:, 0] == 0]
    star_unbound = star_unbound[star_unbound[:, 0] == 0]

    generate_plots(opts.fpath_snapshots, opts.num_timesteps, opts.timestep_duration_yr, bbh_merge, star_explode, star_merge, bh_unbound, star_unbound)


######## Execution ########
if __name__ == "__main__":
    main()
