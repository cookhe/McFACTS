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

def surrogate(m1, m2, s1m, s2m, sa1, sa2, p12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate):

    #print(m1, m2, s1m, s2m, sa1, sa2, p12)
    mass_final, spin_final, kick_final = [], [], []
    mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12 = [], [], [], [], [], [], []
    
    for value in m1:
        mass_1.append(value)
    for value in m2:
        mass_2.append(value)
    for value in s1m:
        spin_1_mag.append(value)
    for value in s2m:
        spin_2_mag.append(value)
    for value in sa1:
        spin_angle_1.append(value)
    for value in sa2:
        spin_angle_2.append(value)
    for value in p12:
        phi_12.append(value)

    for i in range(len(mass_1)):
        #print(mass_1, mass_2, spin_1_mag, spin_2_mag, spin_angle_1, spin_angle_2, phi_12, bin_sep, bin_inc, bin_phase, bin_orb_a, mass_SMBH, spin_SMBH, surrogate)
        
        start = time.time()
        M_f, spin_f, v_f = evolve_binary.evolve_binary(
            mass_1[i],
            mass_2[i],
            spin_1_mag[i],
            spin_2_mag[i],
            spin_angle_1[i],
            spin_angle_2[i],
            phi_12[i],
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
        
        mass_final.append(float(M_f))
        spin_final.append(float(spin_f_mag))
        kick_final.append(float(v_f_mag))
    
    #print(M_f, spin_f_mag, v_f_mag)
    
    print("M_f = ", mass_final)
    print("spin_f = ", spin_final)
    print("v_f = ", kick_final)
    
    return np.array(mass_final), np.array(spin_final), np.array(kick_final)