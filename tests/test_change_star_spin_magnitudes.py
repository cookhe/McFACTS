import numpy as np
import pytest
from mcfacts.physics.accretion import change_star_spin_magnitudes

"""
Cases to test:

(2) spin < 0, e <= e_crit -> spin up
(1) spin < 0, e > e_crit  -> spin down
(2) spin > 0, e <= e_crit -> spin up
(1) spin > 0, e > e_crit  -> spin down

(1) change spin from neg to pos
(1) change spin from pos to neg
(2) catch spin outside min/max values
    - spin(e>e_crit) = -0.975 -> -0.985 -> -0.98
    - spin(e<e_crit) = 0.975 -> 0.985 -> 0.98

spins = [ -0.3, -0.3,  -0.3,   0.3,  0.3,   0.3, -0.005, 0.005, -0.975, 0.975]
ecc   = [0.005, 0.01, 0.015, 0.005, 0.01, 0.015,  0.005, 0.015,  0.015, 0.005]


"""

# The following choices equates to changes in spin of 0.01.
#    If you change these such that the change decreases, you
#    may need to adjust the spin values at indices 6 and 7 to
#    ensure they still change sign.
disk_star_eddington_ratio = 1.
disk_star_torque_condition = 0.1
timestep_duration_yr = 1.e5
disk_star_pro_orbs_ecc_crit = 0.01

old_spin_magnitudes = np.array([ -0.3, -0.3,  -0.3,   0.3,  0.3,   0.3, -0.005, 0.005, -0.975, 0.975,    0.,   0.,    0.])
pro_ecc = np.array([0.005, 0.01, 0.015, 0.005, 0.01, 0.015,  0.005, 0.015,  0.015, 0.005, 0.005, 0.01, 0.015])



def test_change_star_spin_magnitudes():
    """Test that star spin magnitudes change correctly based on their
    orbital eccentricity relative to the critical eccentricity.
    """
    new_spin_magnitudes = change_star_spin_magnitudes(old_spin_magnitudes, disk_star_eddington_ratio, disk_star_torque_condition, timestep_duration_yr, pro_ecc, disk_star_pro_orbs_ecc_crit)
    delta = new_spin_magnitudes - old_spin_magnitudes

    # Mask of BHs with circular obits (i.e. eccentricity less than the critical value)
    mask_where_circular = pro_ecc <= disk_star_pro_orbs_ecc_crit

    # BHs with ecc < ecc_crit spin up : positive delta
    assert np.all(delta[mask_where_circular] >= 0)
    # BHs with ecc > ecc_crit spin down : negative delta
    assert np.all(delta[~mask_where_circular] < 0)

    # No spins should be less than -0.98 or greater than 0.98
    assert np.all(new_spin_magnitudes >= -0.98)
    assert np.all(new_spin_magnitudes <= 0.98)

    # Check that the spins that started at zero are no longer zero
    assert np.all(new_spin_magnitudes[np.where(old_spin_magnitudes == 0)] != 0)


print("ecc:", pro_ecc)
print("ecc <= crit?", pro_ecc <= disk_star_pro_orbs_ecc_crit)
print("pre spin mags:", old_spin_magnitudes)

new_spins = change_star_spin_magnitudes(old_spin_magnitudes, disk_star_eddington_ratio, disk_star_torque_condition, timestep_duration_yr, pro_ecc, disk_star_pro_orbs_ecc_crit)

print("post spin mags:", new_spins)
print("delta", new_spins - old_spin_magnitudes)
