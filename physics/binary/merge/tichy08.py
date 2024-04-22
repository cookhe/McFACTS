
import numpy as np

def merged_mass(mass_1,mass_2,spin_1,spin_2):
# Calculate the final mass of a merged binary
# Only depends on M1,M2,a1,a2
# Using approximations from Tichy \& Maronetti (2008) where
# m_final=(M1+M2)*[1.0-0.2nu-0.208nu^2(a1+a2)]
# where nu is the symmetric mass ratio or nu=q/((1+q)^2)
    # These are arrays
    #mass_1 = -1
    #mass_2 = -1
    #if isinstance(m1, np.ndarray):
    # if len(m1.shape) >0:
    #    mass_1 = m1[0]
    #    mass_2 = m2[0]
    #    total_spin = spin_1[0] + spin_2[0]
    #else:
    #    mass_1 = m1
    #    mass_2 = m2
    #    total_spin = spin_1 + spin_2
    
    primary = np.max(np.c_[mass_1, mass_2])
    secondary = np.min(np.c_[mass_1, mass_2])

    #If issue with primary mass, print!
    if primary < 1.0:
        print("primary,secondary=", primary, secondary)
        mass_ratio = 1.0
    if primary > 1.0:    
        mass_ratio = secondary/primary
    
    total_mass = primary + secondary
    total_spin = spin_1 + spin_2
    inv_mass_ratio = (1.0/mass_ratio)
    nu_factor = (1.0+mass_ratio)**2.0
    nu = mass_ratio/nu_factor
    nusq = nu*nu
    #print(" Mass array ",total_mass)
    nu =  (primary* secondary)/ (total_mass)**2 # mass_ratio/nu_factor
    nusq = nu*nu

    #mass_1 = -1
    #mass_2 = -1
    #if isinstance(m1, np.ndarray):
    # if len(m1.shape) >0:
    #    mass_1 = m1[0]
    #    mass_2 = m2[0]
    #    total_spin = spin_1[0] + spin_2[0]
    #else:
    #    mass_1 = m1
    #    mass_2 = m2
    #    total_mass = mass_1 + mass_2
    #    total_spin=spin_1 + spin_2

    #print(" Mass array ",mass_1 + mass_2)
    #nu =  (mass_1* mass_2)/ (mass_1+mass_2)**2 # mass_ratio/nu_factor
    #nusq = nu*nu

    mass_factor = 1.0-(0.2*nu)-(0.208*nusq*total_spin)
    merged_mass = (total_mass)*mass_factor
    return merged_mass


def merged_spin(mass_1, mass_2, spin_1, spin_2, bin_ang_mom):
# Calculate the spin magnitude of a merged binary.
# Only depends on M1,M2,a1,a2 and the binary ang mom around its center of mass
# Using approximations from Tichy \& Maronetti(2008) where
# a_final=0.686(5.04nu-4.16nu^2) +0.4[a1/((0.632+1/q)^2)+ a2/((0.632+q)^2)]
# where q=m_2/m_1 and nu=q/((1+q)^2)
    # if mass_1 >= mass_2:
    #     primary = mass_1
    #     secondary = mass_2
    # else:
    #     primary = mass_2
    #     secondary = mass_1
    primary = np.max(np.c_[mass_1, mass_2])
    secondary = np.min(np.c_[mass_1, mass_2])

    mass_ratio = secondary/primary
    inv_mass_ratio = (1.0/mass_ratio)
    nu_factor = (1.0+mass_ratio)**2.0
    nu = mass_ratio/nu_factor
    nusq = nu*nu
    spin1_factor = (0.632+inv_mass_ratio)**(2.0)
    spin2_factor = (0.632+mass_ratio)**2.0
    merged_spin = 0.686*((5.04*nu)-(4.16*nusq))+(0.4*((spin_1/spin1_factor)+(spin_2/spin2_factor)))
    #print("MERGER props",primary,secondary,spin_1,spin_2,bin_ang_mom, merged_spin)
    return merged_spin

