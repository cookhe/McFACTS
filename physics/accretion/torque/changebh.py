import numpy as np

def change_spin_magnitudes(prograde_bh_spins, frac_Eddington_ratio, spin_torque_condition, timestep, prograde_bh_orb_ecc, e_crit):
    """Update the spin magnitude of the embedded black holes based on their accreted mass
        in this timestep.

    Parameters
    ----------
    prograde_bh_spins : float array
        initial spins of black holes in prograde orbits around SMBH
    frac_Eddington_ratio : float
        user chosen input set by input file; Accretion rate of fully embedded stellar mass 
        black hole in units of Eddington accretion rate. 1.0=embedded BH accreting at Eddington.
        Super-Eddington accretion rates are permitted.
    spin_torque_condition : float
        user chosen input set by input file; fraction of initial mass required to be 
        accreted before BH spin is torqued fully into alignment with the AGN disk. 
        We don't know for sure but Bogdanovic et al. says between 0.01=1% and 0.1=10% 
        is what is required
    timestep : float
        length of timestep in units of years
    prograde_bh_orb_ecc : float array
        orbital eccentricity of BH in prograde orbits around SMBH
    e_crit : float
        critical value of orbital eccentricity below which prograde accretion (& migration & binary formation) occurs
    Returns
    -------
    bh_new_spins : float array
        spin magnitudes of black holes after accreting at prescribed rate for one timestep
    """
    #A retrograde BH a=-1 will spin down to a=0 when it accretes a factor sqrt(3/2)=1.22 in mass (Bardeen 1970).
    # Since M_edd/t = 2.3 e-8 M0/yr or 2.3e-4M0/10kyr then M(t)=M0*exp((M_edd/t)*f_edd*time)
    # so M(t)~1.2=M0*exp(0.2) so in 10^7yr, spin should go a=-1 to a=0. Or delta a ~ 10^-3 every 10^4yr.

    #def change_spin_magnitudes(bh_spins,prograde_orb_ang_mom_indices,frac_Eddington_ratio,spin_torque_condition,mass_growth_Edd_rate,timestep):
    #bh_new_spins=bh_spins
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1
   
    #I think this should be 1.e-3! See argument above.
    spin_iteration = (1.e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    #print("Spin Iteration", spin_iteration)
    #spin_iteration = (4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)

    bh_new_spins = prograde_bh_spins
    #Singleton BH with orb ecc > e_crit will spin down b/c accrete retrograde
    prograde_bh_spin_down = np.ma.masked_where(prograde_bh_orb_ecc <= e_crit, prograde_bh_orb_ecc)
    #Singleton BH with orb ecc < e_crit will spin up b/c accrete prograde
    prograde_bh_spin_up = np.ma.masked_where(prograde_bh_orb_ecc >e_crit, prograde_bh_orb_ecc)
    #Indices of singleton BH with orb ecc > e_crit
    indices_bh_spin_down = np.ma.nonzero(prograde_bh_spin_down) 
    #print(indices_bh_spin_down)
    #Indices of singleton BH with orb ecc < e_crit
    indices_bh_spin_up = np.ma.nonzero(prograde_bh_spin_up)
    #bh_new_spins[prograde_orb_ang_mom_indices]=bh_new_spins[prograde_orb_ang_mom_indices]+(4.4e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    bh_new_spins[indices_bh_spin_up] = prograde_bh_spins[indices_bh_spin_up] + spin_iteration
    #print('BH spin up', bh_new_spins[indices_bh_spin_up])
    #Spin down BH with orb ecc > e_crit
    bh_new_spins[indices_bh_spin_down] = prograde_bh_spins[indices_bh_spin_down] - spin_iteration
    #print('BH spin down', bh_new_spins[indices_bh_spin_down])
    # TO DO: Include a condition to keep a maximal (a=+0.98) spin BH at that value once it reaches it
    #Housekeeping:
    bh_max_spin = 0.98
    bh_min_spin = -0.98
    #print("OLD/NEW SPINs",prograde_bh_spins,bh_new_spins)
    #for i in range(len(prograde_bh_spins)):
    #    if bh_new_spins[i] < bh_min_spin:
    #        bh_new_spins[i] = bh_min_spin

    #    if bh_new_spins[i] > bh_max_spin:
    #        bh_new_spins[i] = bh_max_spin      
    #bh_new_spins = np.where(bh_new_spins < bh_min_spin, bh_new_spins, bh_min_spin)
    #bh_new_spins = np.where(bh_new_spins > bh_max_spin, bh_new_spins, bh_max_spin)
    #Return updated new spins    
    return bh_new_spins


def change_spin_angles(prograde_bh_spin_angles, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep, prograde_bh_orb_ecc, e_crit):
    """_summary_

    Parameters
    ----------
    prograde_bh_spin_angles : float array
        _description_
    frac_Eddington_ratio : float
        _description_
    spin_torque_condition : _type_
        _description_
    spin_minimum_resolution : _type_
        _description_
    timestep : float
        _description_
    prograde_bh_orb ecc : float array
        orbital eccentricity of BH around SMBH
    e_crit : float
        critical eccentricity of BH below which prograde accretion & spin torque into disk alignment else retrograde accretion
    Returns
    -------
    bh_new_spin_angles : float array
        Iterated BH spin angles w.r.t disk orbital angular momentum. 0= complete alignment.
    """
    #Calculate change in spin angle due to accretion during timestep
    normalized_Eddington_ratio = frac_Eddington_ratio/1.0
    normalized_timestep = timestep/1.e4
    normalized_spin_torque_condition = spin_torque_condition/0.1

    spin_torque_iteration = (6.98e-3*normalized_Eddington_ratio*normalized_spin_torque_condition*normalized_timestep)
    
    #print("Initital spin angles",prograde_bh_spin_angles)
    #Assume same angles as before to start
    bh_new_spin_angles = prograde_bh_spin_angles
    #Singleton BH with orb ecc > e_crit will spin down b/c accrete retrograde
    prograde_bh_spin_down = np.ma.masked_where(prograde_bh_orb_ecc <= e_crit, prograde_bh_orb_ecc)
    #Singleton BH with orb ecc < e_crit will spin up b/c accrete prograde
    prograde_bh_spin_up = np.ma.masked_where(prograde_bh_orb_ecc >e_crit, prograde_bh_orb_ecc)
    #Indices of singleton BH with orb ecc > e_crit
    indices_bh_spin_down = np.ma.nonzero(prograde_bh_spin_down) 
    #print(indices_bh_spin_down)
    #Indices of singleton BH with orb ecc < e_crit
    indices_bh_spin_up = np.ma.nonzero(prograde_bh_spin_up)

    # Spin up BH are torqued towards zero (ie alignment with disk, so decrease mag of spin angle)
    bh_new_spin_angles[indices_bh_spin_up] = prograde_bh_spin_angles[indices_bh_spin_up] - spin_torque_iteration
    #Spin down BH with orb ecc > e_crit are torqued toward anti-alignment with disk, incr mag of spin angle.
    bh_new_spin_angles[indices_bh_spin_down] = prograde_bh_spin_angles[indices_bh_spin_down] + spin_torque_iteration
    #print(bh_new_spin_angles[indices_bh_spin_down])
    
    #TO DO: Include a condition to keep spin angle at or close to zero once it gets there
    #Return new spin angles
    #Housekeeping
    # Max bh spin angle in rads (pi rads = anti-alignment)
    bh_max_spin_angle = 3.10
    bh_new_spin_angles[bh_new_spin_angles<spin_minimum_resolution] = 0.0
    bh_new_spin_angles[bh_new_spin_angles > bh_max_spin_angle] = bh_max_spin_angle
    #print("Final spin angles",bh_new_spin_angles)
    return bh_new_spin_angles


