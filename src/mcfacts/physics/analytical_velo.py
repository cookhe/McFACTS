"""
Module for calculating the recoil (kick) velocity of a remnant BH.
"""
import numpy as np
from mcfacts.mcfacts_random_state import rng

def analytical_kick_velocity(
        mass_1,
        mass_2,
        spin_1,
        spin_2,
        spin_angle_1,
        spin_angle_2):
    """
    Compute the analytical gravitational wave recoil (kick) velocity for merging black hole binaries
    as in Akiba et al. 2024 (arXiv:2410.19881).

    Parameters
    ----------
    mass_1 : numpy.ndarray
        Mass [M_sun] of object 1 with :obj:`float` type
    mass_2 : numpy.ndarray
        Mass [M_sun] of object 2 with :obj:`float` type
    spin_1 : numpy.ndarray
        Spin magnitude [unitless] of object 1 with :obj:`float` type
    spin_2 : numpy.ndarray
        Spin magnitude [unitless] of object 2 with :obj:`float` type
    spin_angle_1 : numpy.ndarray
        Spin angle [radian] of object 1 with :obj:`float` type
    spin_angle_2 : numpy.ndarray
        Spin angle [radian] of object 2 with :obj:`float` type

    Returns
    -------
    v_kick : np.ndarray
        Kick velocity [km/s] of the remnant BH with :obj:`float` type
    """
    mass_1 = np.array(mass_1)
    mass_2 = np.array(mass_2)
    spin_1 = np.array(spin_1)
    spin_2 = np.array(spin_2)
    spin_angle_1 = np.array(spin_angle_1)
    spin_angle_2 = np.array(spin_angle_2)

    # As in Akiba et al, mass_2 should be the more massive BH in the binary.
    mask = mass_1 <= mass_2

    m_1_new = np.where(mask, mass_1, mass_2)
    m_2_new = np.where(mask, mass_2, mass_1)
    spin_1_new = np.where(mask, spin_1, spin_2)
    spin_2_new = np.where(mask, spin_2, spin_1)
    spin_angle_1_new = np.where(mask, spin_angle_1, spin_angle_2)
    spin_angle_2_new = np.where(mask, spin_angle_2, spin_angle_1)

    # "perp" and "par" refer to components perpendicular and parallel to the orbital angular momentum axis, respectively.
    # Orbital angular momentum axis of binary is aligned with the disk angualr momentum.
    # Find the perp and par components of spin:
    spin_1_par = spin_1_new * np.cos(spin_angle_1_new)
    spin_1_perp = spin_1_new * np.sin(spin_angle_1_new)
    spin_2_par = spin_2_new * np.cos(spin_angle_2_new)
    spin_2_perp = spin_2_new * np.sin(spin_angle_2_new)

    # Find the mass ratio q and asymmetric mass ratio eta:
    q = m_1_new / m_2_new
    eta = q / (1 + q)**2

    # As defined in Akiba et al. 2024:
    S = (2 * (spin_1_new + q**2 * spin_2_new)) / (1 + q)**2

    xi = np.radians(145)
    A = 1.2e4
    B = -0.93
    H = 6.9e3
    V_11, V_A, V_B, V_C = 3678, 2481, 1793, 1507

    angle = rng.uniform(0.0, 2*np.pi, size=len(mass_1))

    v_m = A * eta**2 * np.sqrt(1 - 4 * eta) * (1 + B * eta)
    v_perp = (H * eta**2 / (1 + q)) * (spin_2_par - q * spin_1_par)

    v_par = ((16 * eta**2) / (1 + q)) * (V_11 + (V_A * S) + (V_B * S**2) + (V_C * S**3)) * \
            np.abs(spin_2_perp - q * spin_1_perp) * np.cos(angle)

    v_kick = np.sqrt((v_m + v_perp * np.cos(xi))**2 +
                     (v_perp * np.sin(xi))**2 +
                     v_par**2)
    v_kick = np.array(v_kick)
    assert np.all(v_kick > 0), \
        "v_kick has values <= 0"
    assert np.isfinite(v_kick).all(), \
        "Finite check failure: v_kick"


    return v_kick
