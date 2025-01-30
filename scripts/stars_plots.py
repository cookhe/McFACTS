#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import glob as g
import os
from scipy.optimize import curve_fit
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data
from mcfacts.vis import plotting
from mcfacts.vis import styles

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

#apj_col or apj_page
figsize = "apj_page"

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-directory",
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--fname-stars",
                        default="output_mergers_stars.dat",
                        type=str, help="output_stars file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts


def make_gen_masks(table, col):
    """Create masks for retrieving different sets of a stellar population

    Parameters
    ----------
    table : numpy.ndarray
        Data
    col : int
        Column index where generation information is stored
    """

    # Column with generation data
    gen_obj = table[:, col]

    # Masks for hierarchical generations
    # g1 : 1g objects
    # g2 : 2g objects
    # g3 : >=3g
    # Pipe operator (|) = logical OR. (&)= logical AND.

    g1_mask = (gen_obj == 1)
    g2_mask = (gen_obj == 2)
    gX_mask = (gen_obj >= 3)

    return (g1_mask, g2_mask, gX_mask)


def main():
    opts = arg()

    stars = np.loadtxt(opts.fname_stars, skiprows=2)

    merger_g1_mask, merger_g2_mask, merger_gX_mask = make_gen_masks(stars, 11)

    # Ensure no union between sets
    assert all(merger_g1_mask & merger_g2_mask) == 0
    assert all(merger_g1_mask & merger_gX_mask) == 0
    assert all(merger_g2_mask & merger_gX_mask) == 0

    # Ensure no elements are missed
    assert all(merger_g1_mask | merger_g2_mask | merger_gX_mask) == 1

    # folders = (g.glob(opts.runs_directory + "gal*"))

    # data = np.loadtxt(folders[0] + "/initial_params_star.dat",skiprows=2)


    # ========================================
    # Stars initial mass
    # ========================================

    # fig = plt.figure(figsize=plotting.set_size(figsize))
    # bins = np.linspace(np.log10(data[:,2]).min(), np.log10(data[:,2]).max(),20)

    # plt.hist(np.log10(data[:,2]), bins=bins)
    # plt.xlabel('log initial mass [$M_\odot$]')

    # if figsize == 'apj_col':
    #     plt.legend(fontsize=6)
    # elif figsize == 'apj_page':
    #     plt.legend()

    # plt.savefig(opts.plots_directory + r"/stars_initial_mass.png",format="png")


    # ========================================
    # Number of Mergers vs Mass
    # ========================================

    # Plot intial and final mass distributions
    fig = plt.figure(figsize=plotting.set_size(figsize))
    counts, bins = np.histogram(stars[:, 3])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(stars[:, 3].min()), int(stars[:, 3].max()) + 2, 1)

    hist_data = [stars[:, 3][merger_g1_mask], stars[:, 3][merger_g2_mask], stars[:, 3][merger_gX_mask]]
    hist_label = ['1g', '2g', r'$\geq$3g']
    hist_color = [styles.color_gen1, styles.color_gen2, styles.color_genX]

    plt.hist(hist_data, bins=bins, align='left', color=hist_color, alpha=0.9, rwidth=0.8, label=hist_label, stacked=True)

    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Star Mass [$M_\odot$]')
    #plt.xscale('log')
    # plt.ylim(-5,max(counts))
    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    svf_ax.tick_params(axis='x', direction='out', which='both')
    #plt.grid(True, color='gray', ls='dashed')
    svf_ax.yaxis.grid(True, color='gray', ls='dashed')
    #plt.xticks(np.geomspace(int(stars[:, 3].min()), int(stars[:, 3].max()), 5).astype(int))
    #plt.xticks(np.geomspace(20, 200, 5).astype(int))

    #svf_ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    #svf_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.savefig(opts.plots_directory + r"/star_merger_mass.png", format='png')

    plt.close()



    # ========================================
    # Merger Mass vs Radius
    # ========================================

    # TQM has a trap at 500r_g, SG has a trap radius at 700r_g.
    # trap_radius = 500
    trap_radius = 700

    # Separate generational subpopulations
    gen1_orb_a = stars[:, 2][merger_g1_mask]
    gen2_orb_a = stars[:, 2][merger_g2_mask]
    genX_orb_a = stars[:, 2][merger_gX_mask]
    gen1_mass = stars[:, 3][merger_g1_mask]
    gen2_mass = stars[:, 3][merger_g2_mask]
    genX_mass = stars[:, 3][merger_gX_mask]

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(gen1_orb_a, gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g'
                )

    plt.scatter(gen2_orb_a, gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g'
                )

    plt.scatter(genX_orb_a, genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g'
                )

    plt.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} ' + r'$R_g$')

    plt.ylabel(r'Star Mass [$M_\odot$]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.ylim(0.4, 400)

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_mass_v_radius.png", format='png')
    plt.close()

    # ========================================
    # HRD
    # ========================================

    # Separate generational subpopulations
    gen1_teff = stars[:, 5][merger_g1_mask]
    gen2_teff = stars[:, 5][merger_g2_mask]
    genX_teff = stars[:, 5][merger_gX_mask]
    gen1_lum = stars[:, 6][merger_g1_mask]
    gen2_lum = stars[:, 6][merger_g2_mask]
    genX_lum = stars[:, 6][merger_gX_mask]


    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.scatter(gen1_teff, gen1_lum,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g'
                )

    plt.scatter(gen2_teff, gen2_lum,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g'
                )

    plt.scatter(genX_teff, genX_lum,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\geq$3g'
                )

    plt.gca().invert_xaxis()
    plt.ylabel(r"$\log L/L_\odot$")
    plt.xlabel(r"$\log T_\mathrm{eff}/\mathrm{K}$")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    svf_ax = plt.gca()
    svf_ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory + "/star_hrd.png", format='png')
    plt.close()



######## Execution ########
if __name__ == "__main__":
    main()
