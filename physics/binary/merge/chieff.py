import numpy as np


def chi_effective(mass_1, mass_2, spin_1, spin_2, spin_angle1, spin_angle2, bin_ang_mom):
    # Calculate the chi_effective of a merger
    #chi_eff=;chi=(m1*a1*cos(theta_1) + m2*a2*cos(theta_2))/(mbin)
    total_mass  =mass_1 + mass_2
    angle_1 = spin_angle1
    angle_2 = spin_angle2
    abs_spin1 = np.abs(spin_1)
    abs_spin2 = np.abs(spin_2)

# If L_b=-1 then need to measure angles wrt to 3.1415rad not 0 rad.
    if not(isinstance(bin_ang_mom, np.ndarray)):
           if bin_ang_mom == -1:
               angle_1 = np.pi - spin_angle1
               angle_2 = np.pi - spin_angle2
    else:
        indx_swap = np.where(bin_ang_mom == -1)[0]
        angle_1[indx_swap] = np.pi - spin_angle1[indx_swap]
        angle_2[indx_swap] = np.pi - spin_angle2[indx_swap]



# Calculate each component of chi_effective
    chi_factor1 = (mass_1/total_mass)*abs_spin1*np.cos(angle_1)
    chi_factor2 = (mass_2/total_mass)*abs_spin2*np.cos(angle_2)

    chi_eff = chi_factor1 + chi_factor2

    return chi_eff

def chi_p(mass_1, mass_2, spin_1, spin_2, spin_angle1, spin_angle2, bin_ang_mom):
    # Calculate the chi_p associated with a merger
    # chi_p = max[spin_1_perp, (q(4q+3)/(4+3q))* spin_2_perp]
    # where spin_1_perp = spin1*sin(spin_angle1) and q=mass_2/mass_1 where mass_2< mass_1
    #chi_eff=;chi=(m1*a1*cos(theta_1) + m2*a2*cos(theta_2))/(mbin)
    total_mass  = mass_1 + mass_2
    #Convert spin angle from radians to degrees. Nope! Don't need to, numpy.cos(angle), angle in rads.
    #spin_angle1_deg = spin_angle1*(180.0/np.pi)
    #spin_angle2_deg = spin_angle2*(180.0/np.pi)
    # If mass1 is the dominant binary partner
    #Define default mass ratio of 1, otherwise choose based on masses
    q = 1.0
    #Define default spins
    spin_1_perp = spin_1*np.sin(spin_angle1)
    spin_2_perp = spin_2*np.sin(spin_angle2)
    
    if mass_1 > mass_2:
         q=mass_2/mass_1
         spin_1_perp = spin_1*np.sin(spin_angle1)
         spin_2_perp = spin_2*np.sin(spin_angle2)
    # If mass2 is the dominant binary partner
    if mass_2 > mass_1:
         q=mass_1/mass_2
         spin_1_perp = spin_2*np.sin(spin_angle2)
         spin_2_perp = spin_1*np.sin(spin_angle1)     

    q_factor = q*((4.0*q) + 3.0)/(4.0+(3.0*q))
    
    #Assume spin_1_perp is dominant source of chi_p
    chi_p = spin_1_perp
    #if not then change chi_p definition and output
    if chi_p < q_factor*spin_2_perp:
         chi_p = q_factor*spin_2_perp

    return chi_p
