import numpy as np
import fit_modeler
import evolve_binary

import time

if __name__ == "__main__":
    mass_1 = 1.8
    mass_2 = 1.2
    spin_1_mag = 0.7
    spin_2_mag = 0.2
    spin_angle_1 = np.pi / 3
    spin_angle_2 = np.pi / 2
    phi_12 = np.pi / 4
    # This should be in units of mass_1 + mass_2
    bin_sep = 10
    bin_inc = [0, 0, 1]
    bin_phase = 0
    # These next three are used to correct the remnant velocity;
    # If they are None, no correction is applied.
    bin_orb_a = None
    mass_SMBH = None
    spin_SMBH = None

    surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")
    
    print(mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate)

    M_f, spin_f, v_f = evolve_binary.evolve_binary(
        mass_1,
        mass_2,
        spin_1_mag,
        spin_2_mag,
        spin_angle_1,
        spin_angle_2,
        phi_12,
        bin_sep,
        bin_inc,
        bin_phase,
        bin_orb_a,
        mass_SMBH,
        spin_SMBH,
        surrogate,
        verbose=True,
    )

    print("M_f = ", M_f)
    print("spin_f = ", spin_f)
    print("v_f = ", v_f)
