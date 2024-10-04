#!/usr/bin/env python3

######## Globals ########
COLUMN_NAMES = "iter CM M chi_eff a_tot spin_angle m1 m2 a1 a2 theta1 theta2 gen1 gen2 t_merge chi_p"

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import os
from scipy.optimize import curve_fit
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data
from mcfacts.vis import plotting
from mcfacts.vis import styles

# Use the McFACTS plot style
plt.style.use("mcfacts.vis.mcfacts_figures")

figsize = "apj_col"

######## Arg ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname-emris",
        default="output_mergers_emris.dat",
        type=str, help="output_emris file")
    parser.add_argument("--fname-mergers",
        default="output_mergers_population.dat",
        type=str, help="output_mergers file")
    parser.add_argument("--plots-directory",
        default=".",
        type=str, help="directory to save plots")
    parser.add_argument("--fname-lvk",
        default="output_mergers_lvk.dat",
        type=str, help="output_lvk file")
    opts = parser.parse_args()
    print(opts.fname_mergers)
    assert os.path.isfile(opts.fname_mergers)
    assert os.path.isfile(opts.fname_emris)
    assert os.path.isfile(opts.fname_lvk)
    return opts

######## Main ########
def main():
    #plt.style.use('seaborn-v0_8-poster')

    # need section for loading data
    opts = arg()
    mergers = np.loadtxt(opts.fname_mergers, skiprows=2)
    emris = np.loadtxt(opts.fname_emris, skiprows=2)
    lvk = np.loadtxt(opts.fname_lvk,skiprows=2)

    # exclude all rows with NaNs or zeros in the final mass column
    mask = (np.isfinite(mergers[:,2])) & (mergers[:,2] != 0)
    mergers = mergers[mask]

    # Generations of mass 1 and 2 in the binary merger
    gen_obj1 = mergers[:,12]
    gen_obj2 = mergers[:,13]

    # Masks for hierarchical generations
    # g1 : all 1g-1g mergers
    # g2 : 2g-1g and 2g-2g mergers
    # g3 : >=3g-Ng (first object at least 3rd gen; second object any gen)
    # Pipe operator (|) = logical OR. (&)= logical AND.
    g1_conditions = (gen_obj1 == 1) & (gen_obj2 ==1)
    g2_conditions = ((gen_obj1 == 2) | (gen_obj2 == 2)) & ((gen_obj1 <= 2) & (gen_obj2 <= 2))
    gX_conditions = (gen_obj1 >= 3) | (gen_obj2 >= 3)

    # Ensure no union between sets
    assert all(g1_conditions & g2_conditions) == 0
    assert all(g1_conditions & gX_conditions) == 0
    assert all(g2_conditions & gX_conditions) == 0
    # Ensure no elements are missed
    assert all(g1_conditions | g2_conditions | gX_conditions) == 1


    # ------------ HISTORGRAM ---------------
    # Plot intial and final mass distributions
    fig = plt.figure(figsize=plotting.set_size(figsize))
    counts, bins = np.histogram(mergers[:,2])
    # plt.hist(bins[:-1], bins, weights=counts)
    bins = np.arange(int(mergers[:,2].min()), int(mergers[:,2].max())+2, 1)
    plt.hist(mergers[:,2], bins=bins, align='left', color='rebeccapurple', alpha=0.9, rwidth=0.8)
    
    plt.ylabel('Number of Mergers')
    plt.xlabel(r'Remnant Mass [$M_\odot$]')
    plt.xscale('log')
    # plt.ylim(-5,max(counts))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', direction='out', which='both')
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory+r"/merger_remnant_mass.png", format='png')
    plt.close()

    # TQM has a trap at 245r_g, SG has a trap radius at 700r_g.
    #trap_radius = 245
    trap_radius = 700
    
    #plt.title('Migration Trap influence')
    for i in range(len(mergers[:,1])):
        if mergers[i,1] < 10.0:
            mergers[i,1] = 10.0

    # Separate generational subpopulations
    gen1_orb_a = mergers[:,1][g1_conditions]
    gen2_orb_a = mergers[:,1][g2_conditions]
    genX_orb_a = mergers[:,1][gX_conditions]
    gen1_mass = mergers[:,2][g1_conditions]
    gen2_mass = mergers[:,2][g2_conditions]
    genX_mass = mergers[:,2][gX_conditions]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    plt.scatter(gen1_orb_a, gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolors="none",
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )
    plt.scatter(gen2_orb_a, gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolors="none",
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    plt.scatter(genX_orb_a, genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolors="none",
                alpha=styles.markeralpha_genX,
                label=r'$\ge$3g-Ng'
                )
    plt.axvline(trap_radius, color='k', linestyle='--', zorder=0,
                label=f'Trap Radius = {trap_radius} '+r'$R_g$')
    #plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
    plt.ylabel(r'Remnant Mass [$M_\odot$]')
    plt.xlabel(r'Radius [$R_g$]')
    plt.xscale('log')
    plt.yscale('log')
    if figsize == 'apj_col':
        plt.legend(fontsize=6)
    elif figsize == 'apj_page':
        plt.legend()
    plt.ylim(18,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.savefig(opts.plots_directory+"/merger_mass_v_radius.png", format='png')
    plt.close()



    # retrieve component masses and mass ratio
    m1 = np.zeros(mergers.shape[0])
    m2 = np.zeros(mergers.shape[0])
    mass_ratio = np.zeros(mergers.shape[0])
    for i in range(mergers.shape[0]):
        if mergers[i,6] < mergers[i,7]:
            m1[i] = mergers[i,7]
            m2[i] = mergers[i,6]
            mass_ratio[i] = mergers[i,6] / mergers[i,7]
        else:
            mass_ratio[i] = mergers[i,7] / mergers[i,6]
            m1[i] = mergers[i,6]
            m2[i] = mergers[i,7]

    # (q,X_eff) Figure details here:
    # Want to highlight higher generation mergers on this plot
    chi_eff = mergers[:,3]

    # Get 1g-1g population
    gen1_chi_eff      = chi_eff[g1_conditions]
    gen1_mass_ratio   = mass_ratio[g1_conditions]
    # 2g-1g and 2g-2g population
    gen2_chi_eff     = chi_eff[g2_conditions]
    gen_mass_ratio  = mass_ratio[g2_conditions]
    # >=3g-Ng population (i.e., N=1,2,3,4,...)
    genX_chi_eff    = chi_eff[gX_conditions]
    genX_mass_ratio = mass_ratio[gX_conditions]
    # all 2+g mergers; H = hierarchical
    genH_chi_eff = chi_eff[(g2_conditions + gX_conditions)]
    genH_mass_ratio = mass_ratio[(g2_conditions + gX_conditions)]

    # function for a line used when fitting to the data.
    def linefunc(x, m):
        """Model for a line passing through (x,y)=(0,1)."""
        return m*(x-1)
    # points for plotting line fit
    x = np.linspace(-1,1, num=2)

    # fit the hierarchical mergers (any binaries with 2+g) to a line passing through 0,1
    # popt contains the model parameters, pcov the covariances
    # poptHigh, pcovHigh = curve_fit(linefunc, high_gen_mass_ratio, high_gen_chi_eff)

    # plot the 1g-1g population
    fig = plt.figure(figsize=(plotting.set_size(figsize)[0],2.8))
    ax2 = fig.add_subplot(111)
    # 1g-1g mergers
    ax2.scatter(gen1_chi_eff, gen1_mass_ratio,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )
    # plot the 2g+ mergers
    ax2.scatter(gen2_chi_eff, gen_mass_ratio,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    # plot the 3g+ mergers
    ax2.scatter(genX_chi_eff, genX_mass_ratio,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\ge$3g-Ng'
                )
    if len(genH_chi_eff) > 0:
        poptHier, pcovHier = curve_fit(linefunc, genH_mass_ratio, genH_chi_eff)
        errHier = np.sqrt(np.diag(pcovHier))[0]
        # plot the line fitting the hierarchical mergers
        ax2.plot(linefunc(x,*poptHier), x,
                ls='dashed',
                lw=1,
                color='gray',
                zorder=3,
                label=r'$d\chi/dq(\geq$2g)=' +
                    f'{poptHier[0]:.2f}' +
                    r'$\pm$' + f'{errHier:.2f}'
                )
        #         #  alpha=linealpha,
    if len(chi_eff) > 0:
        poptAll, pcovAll = curve_fit(linefunc, mass_ratio, chi_eff)
        errAll = np.sqrt(np.diag(pcovAll))[0]
        ax2.plot(linefunc(x,*poptAll), x,
                ls='solid',
                lw=1,
                color='black',
                zorder=3,
                label=r'$d\chi/dq$(all)=' +
                    f'{poptAll[0]:.2f}' +
                    r'$\pm$' + f'{errAll:.2f}'
                )
                #  alpha=linealpha,
    ax2.set(
        ylabel=r'$q = M_2 / M_1$', #  ($M_1 > M_2$)')
        xlabel=r'$\chi_{\rm eff}$',
        ylim=(0,1),
        xlim=(-1,1),
        axisbelow=True
        )
    if figsize == 'apj_col':
        ax2.legend(loc='lower left', fontsize=6)
    elif figsize == 'apj_page':
        ax2.legend(loc='lower left')
    ax2.grid('on', color='gray', ls='dotted')
    plt.savefig(opts.plots_directory+"./q_chi_eff.png", format='png')#,
                # dpi=600)
    plt.close()
    
    #Figure of Disk radius vs Chi_p follows.
    # Can break out higher mass Chi_p events as test/illustration.
    #Set up default arrays for high mass BBH (>40Msun say) to overplot vs chi_p. 
    chi_p = mergers[:,15]
    gen1_chi_p = chi_p[g1_conditions]
    gen2_chi_p = chi_p[g2_conditions]
    genX_chi_p = chi_p[gX_conditions]

    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax1 = fig.add_subplot(111)
    ax1.scatter(np.log10(gen1_orb_a), gen1_chi_p,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g')
    # plot the 2g+ mergers
    ax1.scatter(np.log10(gen2_orb_a), gen2_chi_p,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g')
    # plot the 3g+ mergers
    ax1.scatter(np.log10(genX_orb_a), genX_chi_p,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\ge$3g-Ng')
    #plt.title("In-plane effective Spin vs. Merger radius")
    ax1.set(
        ylabel=r'$\chi_{\rm p}$',
        xlabel=r'$\log_{10} (R)$ [$R_g$]',
        ylim=(0,1),
        axisbelow=True)
    ax1.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        ax1.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax1.legend()
    plt.savefig(opts.plots_directory+"./r_chi_p.png", format='png')
    plt.close()
    
    # plt.figure()
    # index = 2
    # mode = 10
    # pareto = (np.random.pareto(index, 1000) + 1) * mode

    # x = np.linspace(1,100)
    # p = index*mode**index / x**(index+1)

    # # count, bins, _ = plt.hist(pareto, 100)
    # plt.plot(x, p)
    # plt.xlim(0,100)
    # plt.show()

    all_time = mergers[:,14]
    gen1_time = all_time[g1_conditions]
    gen2_time = all_time[g2_conditions]
    genX_time = all_time[gX_conditions]
    
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax3 = plt.subplot()
    #plt.title("Time of Merger after AGN Onset")
    # ax3.scatter(mergers[:,14]/1e6, mergers[:,2], s=pointsize_merge_time, color='darkolivegreen')
    ax3.scatter(gen1_time/1e6, gen1_mass,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )
    # plot the 2g+ mergers
    ax3.scatter(gen2_time/1e6, gen2_mass,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    # plot the 3g+ mergers
    ax3.scatter(genX_time/1e6, genX_mass,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\ge$3g-Ng'
                )
    ax3.set(
        xlabel='Time [Myr]',
        ylabel=r'Remnant Mass [$M_\odot$]',
        yscale="log",
        axisbelow=True
        )
    plt.grid(True, color='gray', ls='dashed')
    if figsize == 'apj_col':
        ax3.legend(fontsize=6)
    elif figsize == 'apj_page':
        ax3.legend()
    plt.savefig(opts.plots_directory+'/time_of_merger.png', format='png')
    plt.close()

    # Sort Objects into Mass 1 and Mass 2 by generation
    mass_mask_g1 = mergers[g1_conditions,6] > mergers[g1_conditions,7]
    gen1_mass_1 = np.zeros(np.sum(g1_conditions))
    gen1_mass_1[mass_mask_g1] = mergers[g1_conditions,6][mass_mask_g1]
    gen1_mass_1[~mass_mask_g1] = mergers[g1_conditions,7][~mass_mask_g1]
    gen1_mass_2 = np.zeros(np.sum(g1_conditions))
    gen1_mass_2[~mass_mask_g1] = mergers[g1_conditions,6][~mass_mask_g1]
    gen1_mass_2[mass_mask_g1] = mergers[g1_conditions,7][mass_mask_g1]

    mass_mask_g2 = mergers[g2_conditions,6] > mergers[g2_conditions,7]
    gen2_mass_1 = np.zeros(np.sum(g2_conditions))
    gen2_mass_1[mass_mask_g2] = mergers[g2_conditions,6][mass_mask_g2]
    gen2_mass_1[~mass_mask_g2] = mergers[g2_conditions,7][~mass_mask_g2]
    gen2_mass_2 = np.zeros(np.sum(g2_conditions))
    gen2_mass_2[~mass_mask_g2] = mergers[g2_conditions,6][~mass_mask_g2]
    gen2_mass_2[mass_mask_g2] = mergers[g2_conditions,7][mass_mask_g2]

    mass_mask_gX = mergers[gX_conditions,6] > mergers[gX_conditions,7]
    genX_mass_1 = np.zeros(np.sum(gX_conditions))
    genX_mass_1[mass_mask_gX] = mergers[gX_conditions,6][mass_mask_gX]
    genX_mass_1[~mass_mask_gX] = mergers[gX_conditions,7][~mass_mask_gX]
    genX_mass_2 = np.zeros(np.sum(gX_conditions))
    genX_mass_2[~mass_mask_gX] = mergers[gX_conditions,6][~mass_mask_gX]
    genX_mass_2[mass_mask_gX] = mergers[gX_conditions,7][mass_mask_gX]

    # Check that there aren't any zeros remaining.
    assert (gen1_mass_1 > 0).all()
    assert (gen1_mass_2 > 0).all()
    assert (gen2_mass_1 > 0).all()
    assert (gen2_mass_2 > 0).all()
    assert (genX_mass_1 > 0).all()
    assert (genX_mass_2 > 0).all()    

    pointsize_m1m2 = 5
    fig = plt.figure(figsize=plotting.set_size(figsize))
    ax4 = plt.subplot()

    # plt.scatter(m1, m2, s=pointsize_m1m2, color='k')
    ax4.scatter(gen1_mass_1, gen1_mass_2,
                s=styles.markersize_gen1,
                marker=styles.marker_gen1,
                edgecolor=styles.color_gen1,
                facecolor='none',
                alpha=styles.markeralpha_gen1,
                label='1g-1g'
                )
    # plot the 2g+ mergers
    ax4.scatter(gen2_mass_1, gen2_mass_2,
                s=styles.markersize_gen2,
                marker=styles.marker_gen2,
                edgecolor=styles.color_gen2,
                facecolor='none',
                alpha=styles.markeralpha_gen2,
                label='2g-1g or 2g-2g'
                )
    # plot the 3g+ mergers
    ax4.scatter(genX_mass_1, genX_mass_2,
                s=styles.markersize_genX,
                marker=styles.marker_genX,
                edgecolor=styles.color_genX,
                facecolor='none',
                alpha=styles.markeralpha_genX,
                label=r'$\ge$3g-Ng'
                )
    ax4.set(
        xlabel=r'$M_1$ [$M_\odot$]',
        ylabel=r'$M_2$ [$M_\odot$]',
        xscale='log',
        yscale='log',
        axisbelow=(True),
        # aspect=('equal')
        )
    ax4.legend(fontsize=6)
    # plt.grid(True, color='gray', ls='dotted')
    plt.savefig(opts.plots_directory+'/m1m2.png', format='png')
    plt.close()

    # GW strain figure: 
    #make sure LISA.py and PhenomA.py in /vis directory
    #Using https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/blob/master/LISA.py 
    # and same location for PhenomA.py
    #Also make sure LIGO sensitivity curves are in /vis directory

    #--------------Here we're pulling the LIGO data from the released files and setting them to H1 and L1 values-------------
    # READ LIGO O3 Sensitivity data (from https://git.ligo.org/sensitivity-curves/o3-sensitivity-curves)
    H1 = impresources.files(data) / \
        'O3-H1-C01_CLEAN_SUB60HZ-1262197260.0_sensitivity_strain_asd.txt'  
    L1 = impresources.files(data) / \
        'O3-L1-C01_CLEAN_SUB60HZ-1262141640.0_sensitivity_strain_asd.txt'
    # Adjust sep according to your delimiter (e.g., '\t' for tab-delimited files)
    dfh1 = pd.read_csv(H1, sep='\t', header=None)  # Use header=None if the file doesn't contain header row
    dfl1 = pd.read_csv(L1, sep='\t', header=None)

    # Access columns as df[0], df[1], ...
    f_H1 = dfh1[0]
    h_H1 = dfh1[1]

    #H - hanford 
    #L - Ligvston

    # create LISA object
    lisa = li.LISA() 

    # Plot LISA's sensitivity curve
    #-------------- f is the frequency (x-axis) being created -------------
    #-------------- Sn is the sensitivity curve of LISA -------------
    f  = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
    Sn = lisa.Sn(f)


    fig, ax = plt.subplots(1, figsize=(plotting.set_size(figsize)[0],2.9))
    
    ax.set_xlabel(r'f [Hz]')#, fontsize=20, labelpad=10)
    ax.set_ylabel(r'${\rm h}_{\rm char}$')#, fontsize=20, labelpad=10)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    ax.set_xlim(1.0e-7, 1.0e+4)
    ax.set_ylim(1.0e-24, 1.0e-15)

    #----------Finding the rows in which EMRIs signals are either identical or zeroes and removing them----------
    identical_rows_emris = np.where( emris[:,5] == emris[:,6])
    zero_rows_emris = np.where(emris[:,6] == 0)    
    emris = np.delete(emris,identical_rows_emris,0)
    #emris = np.delete(emris,zero_rows_emris,0)
    emris[~np.isfinite(emris)] = 1.e-40

    #----------Finding the rows in which LVKs signals are either identical or zeroes and removing them----------
    identical_rows_lvk = np.where(lvk[:,5] == lvk[:,6])
    zero_rows_lvk = np.where(lvk[:,6] == 0)
    lvk = np.delete(lvk,identical_rows_lvk,0)
    #lvk = np.delete(lvk,zero_rows_lvk,0)
    lvk[~np.isfinite(lvk)] = 1.e-40

    #----------Setting the values for the EMRIs and LVKs signals and inverting them----------
    inv_freq_emris = 1/emris[:,6]
    inv_freq_lvk = 1/lvk[:,6]
    #ma_freq_emris = np.ma.where(freq_emris == 0)
    #ma_freq_lvk = np.ma.where(freq_lvk == 0)
    #indices_where_zeros_emris = np.where(freq_emris = 0.)
    #freq_emris = freq_emris[freq_emris !=0]
    #freq_lvk = freq_lvk[freq_lvk !=0]

    #inv_freq_emris = 1.0/ma_freq_emris
    #inv_freq_lvk = 1.0/ma_freq_lvk
    # timestep =1.e4yr
    timestep = 1.e4
    strain_per_freq_emris = emris[:,5]*inv_freq_emris/timestep
    strain_per_freq_lvk = lvk[:,5]*inv_freq_lvk/timestep
    # plot the characteristic detector strains
    ax.loglog(f, np.sqrt(f*Sn),
              label='LISA Sensitivity',
            #   color='darkred',
              zorder=0)
    ax.loglog(f_H1, h_H1,
              label='LIGO O3, H1 Sensitivity',
            #   color='darkblue',
              zorder=0)
    ax.scatter(emris[:,6], strain_per_freq_emris,
               s=0.4*styles.markersize_gen1,
            #    edgecolor='red',
            #    facecolor='none',
               alpha=styles.markeralpha_gen1
               )
    ax.scatter(lvk[:,6], strain_per_freq_lvk,
               s=0.4*styles.markersize_gen1,
            #    edgecolor='blue',
            #    facecolor='none',
               alpha=styles.markeralpha_gen1
               )
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.loglog(f_L1, h_L1,label = 'LIGO O3, L1 Sensitivity') # plot the characteristic strain

    #ax.loglog(f_gw,h,color ='black', label='GW150914')

    if figsize == 'apj_col':
        plt.legend(fontsize=7)
    elif figsize == 'apj_page':
        plt.legend()
    # ax.legend()
    ax.set_xlabel(r'$\nu_{\rm GW}$ [Hz]')#, fontsize=20, labelpad=10)
    ax.set_ylabel(r'$h_{\rm char}/\nu_{\rm GW}$')#, fontsize=20, labelpad=10)
    plt.savefig(opts.plots_directory+'./gw_strain.png', format='png')
    plt.close()

######## Execution ########
if __name__ == "__main__":
    main()
