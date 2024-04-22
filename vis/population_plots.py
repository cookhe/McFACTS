import sys
import matplotlib.pyplot as plt
import numpy as np
import LISA as li
import PhenomA as pa
import pandas as pd

#plt.style.use('seaborn-v0_8-poster')

# load data
fname = sys.argv[1] # McFACTS populations data file.
mergers = np.loadtxt(fname, skiprows=2)
column_names = "iter CM M chi_eff a_tot spin_angle m1 m2 a1 a2 theta1 theta2 gen1 gen2 t_merge"

# plt.figure(figsize=(10,6))
# plt.figure()

# Plot intial and final mass distributions
counts, bins = np.histogram(mergers[:,2])
# plt.hist(bins[:-1], bins, weights=counts)
bins = np.arange(int(mergers[:,2].min()), int(mergers[:,2].max())+2, 1)
plt.hist(mergers[:,2], bins=bins, align='left', color='rebeccapurple', alpha=0.9, rwidth=0.8)

# plt.hist(bh_masses_by_sorted_location, bins=numbins, align='left', label='Final', color='purple', alpha=0.5)
# plt.hist(binary_bh_array[2:4,:bin_index], bins=numbins, align='left', label='Final', color=['purple'], alpha=0.5)
plt.title(f'Black Hole Merger Renmant Masses\n'+
            f'Number of Mergers: {mergers.shape[0]}')
plt.ylabel('Number')
plt.xlabel(r'Mass ($M_\odot$)')

ax = plt.gca()
ax.set_axisbelow(True)
plt.grid(True, color='gray', ls='dashed')
plt.savefig("./merger_remnant_mass.png", format='png')
plt.tight_layout()
plt.close()




trap_radius = 700
plt.figure()
plt.title('Migration Trap influence')
for i in range(len(mergers[:,1])):
    if mergers[i,1] < 10.0:
        mergers[i,1] = 10.0

plt.scatter(mergers[:,1], mergers[:,2], color='teal')
plt.axvline(trap_radius, color='grey', linewidth=20, zorder=0, alpha=0.6, label=f'Radius = {trap_radius} '+r'$R_g$')
plt.text(650, 602, 'Migration Trap', rotation='vertical', size=18, fontweight='bold')
plt.ylabel(r'Mass ($M_\odot$)')
plt.xlabel(r'Radius ($R_g$)')
plt.xscale('log')
plt.legend(frameon=False)
plt.ylim(10,1000)

ax = plt.gca()
ax.set_axisbelow(True)
plt.grid(True, color='gray', ls='dashed')
plt.tight_layout()
plt.savefig("./merger_mass_v_radius.png", format='png')
plt.close()




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

chi_eff = mergers[:,3]
plt.scatter(chi_eff, mass_ratio, color='darkgoldenrod')
plt.title("Mass Ratio vs. Effective Spin")
plt.ylabel(r'$q = M_1 / M_2$ ($M_1 > M_2$)')
plt.xlabel(r'$\chi_{\rm eff}$')
plt.ylim(0,1.05)
plt.xlim(-1,1)
ax = plt.gca()
ax.set_axisbelow(True)
plt.grid(True, color='gray', ls='dashed')
plt.tight_layout()
plt.savefig("./q_chi_eff.png", format='png')
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


plt.figure()
plt.title("Time of Merger after AGN Onset")
plt.scatter(mergers[:,14]/1e6, mergers[:,2], color='darkolivegreen')
plt.xlabel('Time (Myr)')
plt.ylabel(r'Mass ($M_\odot$)')
# plt.xscale("log")
ax = plt.gca()
ax.set_axisbelow(True)
plt.grid(True, color='gray', ls='dashed')
plt.tight_layout()
plt.savefig('./time_of_merger.png', format='png')
plt.close()




plt.figure()
plt.scatter(m1, m2, color='k')
plt.xlabel(r'$M_1$ ($M_\odot$)')
plt.ylabel(r'$M_2$ ($M_\odot$)')
ax = plt.gca()
ax.set_aspect('equal')
ax.set_axisbelow(True)
plt.grid(True, color='gray', ls='dashed')
plt.tight_layout()
plt.savefig('./m1m2.png', format='png')

#GW strain figure: 
#make sure LISA.py and PhenomA.py in /vis directory
#Using https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/blob/master/LISA.py 
# and same location for PhenomA.py
#Also make sure LIGO sensitivity curves are in /vis directory

# READ LIGO O3 Sensitivity data (from https://git.ligo.org/sensitivity-curves/o3-sensitivity-curves)
H1 = 'O3-H1-C01_CLEAN_SUB60HZ-1262197260.0_sensitivity_strain_asd.txt'  
L1 = 'O3-L1-C01_CLEAN_SUB60HZ-1262141640.0_sensitivity_strain_asd.txt'
# Adjust sep according to your delimiter (e.g., '\t' for tab-delimited files)
dfh1 = pd.read_csv(H1, sep='\t', header=None)  # Use header=None if the file doesn't contain header row
dfl1 = pd.read_csv(L1, sep='\t', header=None)

# Access columns as df[0], df[1], ...
f_H1 = dfh1[0]
h_H1 = dfh1[1]
#f_L1 = dfl1[0]
#h_L1 = dfl1[1]

#f_H1 = dfh1[0].to_numpy()
#h_H1 = dfh1[1].to_numpy()
#f_L1 = dfl1[0].to_numpy()
#h_L1 = dfl1[1].to_numpy()

# create LISA object
lisa = li.LISA() 

# Plot LISA's sensitivity curve
f  = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
Sn = lisa.Sn(f)


fig, ax = plt.subplots(1, figsize=(8,6))
plt.tight_layout()

ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
ax.set_ylabel(r'h', fontsize=20, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=20)

ax.set_xlim(1.0e-5, 1e4)
ax.set_ylim(1.0e-24, 1.0e-15)

ax.loglog(f, np.sqrt(f*Sn),label = 'LISA Sensitivity') # plot the characteristic strain
ax.loglog(f_H1, h_H1,label = 'LIGO O3, H1 Sensitivity') # plot the characteristic strain
#ax.loglog(f_L1, h_L1,label = 'LIGO O3, L1 Sensitivity') # plot the characteristic strain

#ax.loglog(f_gw,h,color ='black', label='GW150914')

ax.legend()
plt.savefig('./gw_strain.png', format='png')