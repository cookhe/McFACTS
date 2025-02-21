import numpy as np

def bin_recapture(bindex,bin_array,timestep):
    """ Recapture BBH that has orbital inclination >0 post spheroid encounter. 
    From Fabj+20, if i<5deg (=(5deg/180deg)*pi=0.09rad), time to recapture a BH in SG disk is 1Myr (M_b/10Msun)^-1(R/10^4r_g)
    if i=[5,15]deg =(0.09-0.27rad), time to recapture a BH in SG disk is 50Myrs(M_b/10Msun)^-1 (R/10^4r_g)
    For now, ignore if i>15deg (>0.27rad)

    """
    number_of_binaries = bindex
    # set up 1-d arrays for bin orbital inclinations
    bin_orbital_inclinations = np.zeros(number_of_binaries)
    bin_masses = np.zeros(number_of_binaries)
    bin_coms = np.zeros(number_of_binaries)
    #Critical inclinations (5deg,15deg for SG disk model)
    crit_inc1 = 0.09
    crit_inc2 = 0.27
    
    for j in range(0, number_of_binaries-1):
        # Read in bin masses (in units solar masses) and bin orbital inclinations (in units radians)
        bin_coms[j] = bin_array[9,j]
        bin_masses[j] = bin_array[2,j] + bin_array[3,j]
        bin_orbital_inclinations[j] = bin_array[17,j]
        # Check if bin orbital inclinations are >0    
        if bin_orbital_inclinations[j] > 0:
            print("i0", bin_orbital_inclinations[j])
            # is bin orbital inclination <5deg in SG disk?
            if bin_orbital_inclinations[j] <crit_inc1:
                bin_orbital_inclinations[j] = bin_orbital_inclinations[j]*(1.0 - ((timestep/1.e6)*(bin_masses[j]/10.0)*(bin_coms[j]/1.e4)))

            if bin_orbital_inclinations[j] >crit_inc1 and bin_orbital_inclinations[j] < crit_inc2:
                bin_orbital_inclinations[j] = bin_orbital_inclinations[j]*(1.0 - ((timestep/5.e7)*(bin_masses[j]/10.0)*(bin_coms[j]/1.e4)))
            print("i1", bin_orbital_inclinations[j])
        #Update bin orbital inclinations
        bin_array[17,j] = bin_orbital_inclinations[j]

    return bin_array

def secunda20():
    """ Generate disk capture BH
    Should return one additional BH to be appended to disk location inside 1000r_g every 0.1Myr.
    Draw parameters from initial mass,spin distribution. Assume prograde (Retrograde capture is separate problem).
"""
    return