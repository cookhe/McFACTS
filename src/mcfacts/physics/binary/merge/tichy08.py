
import numpy as np


def merged_mass(masses_1, masses_2, spins_1, spins_2):
    """
    Calculate the final mass of a merged binary.

    Using approximations from Tichy \& Maronetti (2008) where
    m_final=(M1+M2)*[1.0-0.2nu-0.208nu^2(a1+a2)]
    where nu is the symmetric mass ratio or nu=q/((1+q)^2)

    Parameters
    ----------
    masses_1 : numpy array
        Masses of objects 1.
    masses_2 : numpy array
        Masses of objects 2.
    spins_1 : numpy array
        Spin magnitudes of objects 1.
    spins_2 : numpy array
        Spin magnitudes of objects 2.

    Returns
    -------
    merged_masses: numpy array
        Final mass of merger remnant.
    """

    mass_ratios = np.ones(masses_1.size)
    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_1 < masses_2] = masses_1[masses_1 < masses_2] / masses_2[masses_1 < masses_2]

    total_masses = masses_1 + masses_2
    total_spins = spins_1 + spins_2
    nu_factors = np.power(1.0 + mass_ratios, 2)
    nu = mass_ratios / nu_factors
    nu_squared = nu * nu

    mass_factors = 1.0 - (0.2 * nu) - (0.208 * nu_squared * total_spins)
    merged_masses = total_masses*mass_factors

    return (merged_masses)


def merged_spin(masses_1, masses_2, spins_1, spins_2):
    """
    Calculate the spin magnitude of a merged binary.

    Only depends on M1,M2,a1,a2 and the binary ang mom around its center of mass.
    Using approximations from Tichy \& Maronetti(2008) where
    a_final=0.686(5.04nu-4.16nu^2) +0.4[a1/((0.632+1/q)^2)+ a2/((0.632+q)^2)]
    where q=m_2/m_1 and nu=q/((1+q)^2)

    Parameters
    ----------
    masses_1 : numpy array
        Masses of objects 1.
    masses_2 : numpy array
        Masses of objects 2.
    spins_1 : numpy array
        Spin magnitudes of objects 1.
    spins_2 : numpy array
        Spin magnitudes of objects 2.

    Returns
    -------
    merged_spins : numpy array
        Final spin magnitude of merger remnant.
    """

    mass_ratios = np.ones(masses_1.size)
    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_1 < masses_2] = masses_1[masses_1 < masses_2] / masses_2[masses_1 < masses_2]

    mass_ratios_inv = 1.0 / mass_ratios

    nu_factors = np.power(1.0 + mass_ratios, 2)
    nu = mass_ratios / nu_factors
    nu_squared = nu * nu

    spin_factors_1 = np.power(0.632 + mass_ratios_inv, 2)
    spin_factors_2 = np.power(0.632 + mass_ratios, 2)

    merged_spins = 0.686 * ((5.04 * nu) - (4.16 * nu_squared)) + (0.4 * ((spins_1 / spin_factors_1) + (spins_2 / spin_factors_2)))

    return (merged_spins)
