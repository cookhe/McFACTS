#!/usr/bin/env python
'''Plotting script for mcfacts quantities'''
######## Imports ########
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from os.path import join, basename
#### Local ####
from mcfacts.outputs.mcfacts_handler import McfactsHandler

######## Plots ########
def plot_bbh_vs_smbh_mass(
                          run_directory,
                          fname_out,
                          title=None,
                          early=False,
                          late=False,
                         ):
    '''Plot smbh mass'''
    # Check inputs 
    assert type(early) == bool, "early should be True/False"
    assert type(late) == bool, "late should be True/False"
    
    # Load mcfacts data
    MHe = McfactsHandler.from_runs(join(run_directory, "early"))
    MHl = McfactsHandler.from_runs(join(run_directory, "late"))
    # Initialize style
    plt.style.use('bmh')
    # Initialize plot
    width = 2*3.375
    height = width*0.75
    fig, axes = plt.subplots(nrows=1,sharex='col', figsize=(width, height))
    # Identify zero BBH points
    early_bbh_mask = MHe.avg_systems > 0
    late_bbh_mask = MHl.avg_systems > 0
    # Get dx
    x = np.log10(MHe.mass_smbh.value)
    dx = x[1]-x[0]
    # Plot things
    axes.scatter(np.log10(MHe.mass_smbh.value), np.log10(MHe.avg_systems),label='early-type')
    axes.errorbar(
                  np.log10(MHe.mass_smbh.value[~early_bbh_mask]) + 0.07*dx,
                  np.ones(np.sum(~early_bbh_mask))*np.log10(1./min(MHe.n_iterations)),
                  yerr=0.3,
                  uplims=True,
                  marker='none',
	              linestyle='none',
                  linewidth=1.8,
                 )
    axes.scatter(np.log10(MHl.mass_smbh.value), np.log10(MHl.avg_systems),label='late-type')
    axes.errorbar(
                  np.log10(MHl.mass_smbh.value[~late_bbh_mask]) - 0.08*dx,
                  np.ones(np.sum(~late_bbh_mask))*np.log10(1./min(MHl.n_iterations)),
                  yerr=0.3,
                  uplims=True,
                  marker='none',
	              linestyle='none',
                  linewidth=1.8,
                 )
    #axes[1].scatter(np.log10(MH.mass_smbh.value), MH.avg_systems)
    axes.set_xlabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{SMBH}}) [\mathrm{M}_{\odot}]$",
        fontsize=18)
    axes.set_ylabel(r"$\mathrm{log}_{10}(\mu_{\mathrm{BBH}})$",
        fontsize=24)
    #axes[1].set_ylabel(r"$\mu_{\mathrm{BBH}}$")
    axes.set_ylim([
                   np.log10(1./min(MHe.n_iterations)) - 0.5,
                   np.log10(max(max(MHe.avg_systems),max(MHl.avg_systems))) + 1,
                  ])
    #axes[1].set_ylim(0, axes[1].get_ylim()[1])
    axes.set_xlim(np.log10([min(MHe.mass_smbh.value),max(MHe.mass_smbh.value)]))
    # Check title
    axes.set_title(title, fontsize=20)
    # legend
    plt.legend(fontsize=12)
    # Tight layout
    plt.tight_layout()
    # savefig
    plt.savefig(fname_out)
    # Close plt
    plt.close()
    
def plot_nsc_mass_vs_smbh_mass(run_directory, fname_out, title=None):
    '''Plot smbh mass'''
    # Load mcfacts data
    MH = McfactsHandler.from_runs(run_directory)
    # Initialize style
    plt.style.use('bmh')
    # Initialize plot
    fig, ax = plt.subplots()
    # Plot things
    ax.scatter(np.log10(MH.mass_smbh.value), np.log10(MH.batch_param('M_nsc').value))
    ax.set_xlabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{SMBH}}) [\mathrm{M}_{\odot}]$")
    ax.set_ylabel(r"$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{NSC}}) [\mathrm{M}_{\odot}]$")
    # Check title
    if not (title is None):
        fig.suptitle(title)
    # savefig
    plt.savefig(fname_out)
    # Close plt
    plt.close()
    
######## Arguments for default ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=str,
        help="Runs for McFacts")
    parser.add_argument("--early", action='store_true',
        help="Plot early-type quantities?")
    parser.add_argument("--late", action='store_true',
        help="Plot late-type quantities?")
    opts = parser.parse_args()
    return opts

######## main ########
def main():
    opts = arg()
    plot_bbh_vs_smbh_mass(
                          opts.run_directory,
                          join(opts.run_directory, "bbh_vs_smbh_mass.png"),
                          title=basename(opts.run_directory),
                          early=opts.early,
                          late=opts.late,
                         )
    plot_nsc_mass_vs_smbh_mass(
                               join(opts.run_directory, "early"),
                               join(opts.run_directory, "nsc_mass_vs_smbh_mass_early.png"),
                               title=basename(opts.run_directory),
                              )
    plot_nsc_mass_vs_smbh_mass(
                               join(opts.run_directory, "late"),
                               join(opts.run_directory, "nsc_mass_vs_smbh_mass_late.png"),
                               title=basename(opts.run_directory),
                              )
    return

######## Execution ########
if __name__ == "__main__":
    main()
