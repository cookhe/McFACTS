"""
Module to process binary black hole mergers using the surfinBH surrogate model.
"""

import juliacall
import numpy as np
from mcfacts.external.evolve_binary import fit_modeler
from mcfacts.external.evolve_binary import evolve_binary

import pandas as pd
import time, os
from astropy import constants as const

#surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")

def surrogate(mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate):

    #print(mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate)
    
    mass_1 = mass_1[0]
    mass_2 = mass_2[0]
    spin_1_mag = spin_1_mag[0]
    spin_2_mag = spin_2_mag[0]
    spin_angle_1 = spin_angle_1[0]
    spin_angle_2 = spin_angle_2[0]
    phi_12 = phi_12[0]
    
    #print(mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate)
    
    start = time.time()
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
    
    end = time.time()
    
    run_time = end - start
    print("Merger took ", run_time, " seconds")
    
    spin_f_mag = np.linalg.norm(spin_f)
    v_f_mag = np.linalg.norm(v_f) * const.c.value / 1000
    
    #print(M_f, spin_f_mag, v_f_mag)
    
    M_f = np.array([M_f])
    spin_f_mag = np.array([spin_f_mag])
    v_f_mag = np.array([v_f_mag])
    
    #print(M_f, spin_f_mag, v_f_mag)
    
    print("M_f = ", M_f)
    print("spin_f = ", spin_f_mag)
    print("v_f = ", v_f_mag)
    
    return M_f, spin_f_mag, v_f_mag