import numpy as np


def chi_effective(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2, bin_ang_mom):
    """
    Calculate the effective spin :math:`\chi_{\rm eff}` associated with a merger.

    The measured effective spin of a merger is calculated as

    .. math:: \chi_{\rm eff}=\frac{m_1*\chi_1*\cos(\theta_1) + m_2*\chi_2*\cos(\theta_2)}/{m_{\rm bin}}
    
    Parameters
    ----------
    masses_1 : numpy array
        Mass of object 1.
    masses_2 : numpy array
        Mass of object 2.
    spins_1 : numpy array
        Dimensionless spin magnitude of object 1.
    spins_2 : numpy array
        Dimensionless spin magnitude of object 2.
    spin_angles_1 : numpy array
        Dimensionless spin angle of object 1.
    spin_angles_2 : numpy array
        Dimensionless spin angle of object 2.
    bin_ang_mom : int/ndarray
        Magnitude of the binary's mutual angular momentum. If 1, the binary
        is prograde (aligned with disk angular momentum). If -1, the binary
        is retrograde (anti-aligned with disk angular momentum).

    Returns
    -------
    chi_eff : numpy array
        The effective spin value for these object(s).
    """

    total_masses = masses_1 + masses_2
    spins_1 = np.abs(spins_1)
    spins_2 = np.abs(spins_2)

    spin_angles_1[bin_ang_mom < 0] = np.pi - spin_angles_1[bin_ang_mom < 0]
    spin_angles_2[bin_ang_mom < 0] = np.pi - spin_angles_2[bin_ang_mom < 0]

    spin_factors_1 = (masses_1 / total_masses) * spins_1 * np.cos(spin_angles_1)
    spin_factors_2 = (masses_2 / total_masses) * spins_2 * np.cos(spin_angles_2)

    chi_eff = spin_factors_1 + spin_factors_2

    return (chi_eff)


def chi_p(masses_1, masses_2, spins_1, spins_2, spin_angles_1, spin_angles_2, bin_orbs_inc):
    """
    Calculate the precessing spin component :math:`\chi_p` associated with a merger.

    chi_p = max[spin_1_perp, (q(4q+3)/(4+3q))* spin_2_perp]

    where

    spin_1_perp = spin_1 * sin(spin_angle_1)
    spin_2_perp = spin_2 * sin(spin_angle_2)

    are perpendicular to `spin_1

    and :math:`q=M_2/M_1` where :math:`M_2 < M_1`.

    Parameters
    ----------
    masses_1 : numpy array
        Mass of object 1.
    masses_2 : numpy array
        Mass of object 2.
    spins_1 : numpy array
        Dimensionless spin magnitude of object 1.
    spins_2 : numpy array
        Dimensionless spin magnitude of object 2.
    spin_angles_1 : numpy array
        Dimensionless spin angle of object 1.
    spin_angles_2 : numpy array
        Dimensionless spin angle of object 2.
    bin_orbs_inc : numpy array
        Angle of inclination of the binary with respect to the disk.

    Returns
    -------
    chi_p : numpy array
        precessing spin component for these objects
    """

    # If mass_1 is the dominant binary component
    # Define default mass ratio of 1.0, otherwise choose based on masses
    mass_ratios = np.ones(masses_1.size)

    # Define spin angle to include binary inclination wrt disk (units of radians)
    spin_angles_1 = spin_angles_1 + bin_orbs_inc
    spin_angles_2 = spin_angles_2 + bin_orbs_inc

    # Make sure angles are < pi radians
    spin_angles_1_diffs = spin_angles_1 - np.pi
    spin_angles_2_diffs = spin_angles_2 - np.pi

    spin_angles_1[spin_angles_1_diffs > 0] = spin_angles_1[spin_angles_1_diffs > 0] - spin_angles_1_diffs[spin_angles_1_diffs > 0]
    spin_angles_2[spin_angles_2_diffs > 0] = spin_angles_2[spin_angles_2_diffs > 0] - spin_angles_2_diffs[spin_angles_2_diffs > 0]

    # Define default spins
    spins_1_perp = np.abs(spins_1) * np.sin(spin_angles_1)
    spins_2_perp = np.abs(spins_2) * np.sin(spin_angles_2)

    mass_ratios[masses_1 > masses_2] = masses_2[masses_1 > masses_2] / masses_1[masses_1 > masses_2]
    mass_ratios[masses_2 > masses_1] = masses_1[masses_2 > masses_1] / masses_2[masses_2 > masses_1]

    #spins_1_perp[masses_1 > masses_2] = np.abs(spins_1[masses_1 > masses_2]) * np.sin(spin_angles_1[masses_1 > masses_2])
    spins_1_perp[masses_2 > masses_1] = np.abs(spins_2[masses_2 > masses_1]) * np.sin(spin_angles_2[masses_2 > masses_1])

    #spins_2_perp[masses_1 > masses_2] = np.abs(spins_2[masses_1 > masses_2]) * np.sin(spin_angles_2[masses_1 > masses_2])
    spins_2_perp[masses_2 > masses_1] = np.abs(spins_1[masses_2 > masses_1]) * np.sin(spin_angles_1[masses_2 > masses_1])

    mass_ratio_factors = mass_ratios * ((4.0 * mass_ratios) + 3.0) / (4.0 + (3.0 * mass_ratios))

    # Assume spins_1_perp is dominant source of chi_p
    chi_p = spins_1_perp
    # If not then change chi_p definition and output
    chi_p[chi_p < (mass_ratio_factors * spins_2_perp)] = mass_ratio_factors[chi_p < (mass_ratio_factors * spins_2_perp)] * spins_2_perp[chi_p < (mass_ratio_factors * spins_2_perp)]

    return (chi_p)
