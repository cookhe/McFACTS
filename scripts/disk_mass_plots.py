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
figsize = "apj_col"

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-directory",
                        default="runs",
                        type=str, help="folder with files for each run")
    parser.add_argument("--fname-disk",
                        default="output_diskmasscycled.dat",
                        type=str, help="diskmasscycled.dat file")
    parser.add_argument("--plots-directory",
                        default=".",
                        type=str, help="directory to save plots")
    opts = parser.parse_args()
    print(opts.runs_directory)
    assert os.path.isdir(opts.runs_directory)
    return opts

def main():
    opts = arg()

    data = np.loadtxt(opts.fname_disk, skiprows=2)

    # Get the average and std of mass gained/lost at each timestep

    mass_lost_avg = []
    mass_gain_avg = []
    mass_lost_std = []
    mass_gain_std = []

    for t in np.unique(data[:, 1]):
        mass_gain_avg.append(np.mean(data[:, 2][data[:, 1] == t]))
        mass_gain_std.append(np.mean(data[:, 2][data[:, 1] == t]))
        mass_lost_avg.append(np.mean(data[:, 3][data[:, 1] == t]))
        mass_lost_std.append(np.std(data[:, 3][data[:, 1] == t]))

    mass_gain_avg = np.array(mass_gain_avg)
    mass_gain_std = np.array(mass_gain_std)
    mass_lost_avg = np.array(mass_lost_avg)
    mass_lost_std = np.array(mass_lost_std)

    # Transform into Msun/yr. Hardcoding timestep as 10,000 years.
    timestep_duration_yr = 10_000
    mass_gain_avg_rate = mass_gain_avg / timestep_duration_yr
    mass_gain_std_rate = mass_gain_std / timestep_duration_yr
    mass_lost_avg_rate = mass_lost_avg / timestep_duration_yr
    mass_lost_std_rate = mass_lost_std / timestep_duration_yr

    # ========================================
    # Mass lost from the disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_duration_yr, -mass_lost_avg, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_duration_yr, -(mass_lost_avg - mass_lost_std), -(mass_lost_avg + mass_lost_std), alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [$10^3$ yr]")
    plt.ylabel(r"$M_{\rm disk}$ lost [$M_\odot$]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_duration_yr + 1, 6))

    plt.savefig(opts.plots_directory + r"/disk_mass_lost.png",format="png")
    plt.close()

    # ========================================
    # Mass gained to the disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_duration_yr, mass_gain_avg, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_duration_yr, mass_gain_avg - mass_gain_std, mass_gain_avg + mass_gain_std, alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [$10^3$ yr]")
    plt.ylabel(r"$M_{\rm disk}$ gained [$M_\odot$]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_duration_yr + 1, 6))

    plt.savefig(opts.plots_directory + r"/disk_mass_gain.png",format="png")
    plt.close()

    # ========================================
    # Mass accretion rate for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_duration_yr, mass_gain_avg_rate, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_duration_yr, mass_gain_avg_rate - mass_gain_std_rate, mass_gain_avg_rate + mass_gain_std_rate, alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [$10^3$ yr]")
    plt.ylabel(r"$M_{\rm disk}$ accretion [$M_\odot$/yr]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_duration_yr + 1, 6))

    plt.savefig(opts.plots_directory + r"/disk_mdot_gain.png",format="png")
    plt.close()

    # ========================================
    # Mass loss rate for disk
    # ========================================

    fig = plt.figure(figsize=plotting.set_size(figsize))

    plt.plot(np.unique(data[:, 1])/timestep_duration_yr, -mass_lost_avg_rate, label='Mean value')
    plt.fill_between(np.unique(data[:, 1])/timestep_duration_yr, -(mass_lost_avg_rate - mass_lost_std_rate), -(mass_lost_avg_rate + mass_lost_std_rate), alpha=0.2, label='Standard deviation')

    plt.xlabel("Time [$10^3$ yr]")
    plt.ylabel(r"$M_{\rm disk}$ los [$M_\odot$/yr]")

    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()

    plt.xticks(np.linspace(0,data[:, 1].max()/timestep_duration_yr + 1, 6))

    plt.savefig(opts.plots_directory + r"/disk_mdot_loss.png",format="png")
    plt.close()



######## Execution ########
if __name__ == "__main__":
    main()
