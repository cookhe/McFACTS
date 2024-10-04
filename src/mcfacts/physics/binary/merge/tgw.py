import astropy.constants as const
import astropy.units as u
from mcfacts.physics.dynamics.point_masses import time_of_orbital_shrinkage
from mcfacts.physics.dynamics.point_masses import orbital_separation_evolve
from mcfacts.physics.dynamics.point_masses import orbital_separation_evolve_reverse
from mcfacts.physics.dynamics.point_masses import time_of_orbital_shrinkage
from mcfacts.physics.dynamics.point_masses import si_from_r_g, r_g_from_units

def normalize_tgw(smbh_mass, inner_disk_outer_radius):
    """Gravitational wave timescale normalization.

    Calculate the normalization for timescale of merger (in yrs) due to GW emission.
    From Peters(1964):
    
    t_gw approx (5/256)* c^5/G^3 *a_b^4/(M_{b}^{2}mu_{b}) assuming ecc=0.0.
    
    For a_b in units of r_g=GM_smbh/c^2 we find
    
    t_gw=(5/256)*(G/c^3)*(a/r_g)^{4} *(M_s^4)/(M_b^{2}mu_b)

    Put bin_mass_ref in units of 10Msun is a reference mass.
    reduced_mass in units of 2.5Msun.
    
    Parameters
    ----------
    smbh_mass : float
        Mass of the supermassive black hole.
    inner_disk_outer_radius : float
        Outer radius of the inner disk (r_g)

    Returns
    -------
    float
        normalization to gravitational wave timescale
    """

    bin_mass_ref = 10.0
    '''
    G = const.G
    c = const.c
    mass_sun = const.M_sun
    year = 3.1536e7
    reduced_mass = 2.5
    norm = (5.0/256.0)*(G/(c**(3)))*(smbh_mass**(4))*mass_sun/((bin_mass_ref**(2))*reduced_mass)
    time_gw_normalization = norm/year
    '''
    time_gw_normalization = time_of_orbital_shrinkage(
        smbh_mass * u.solMass,
        bin_mass_ref * u.solMass,
        si_from_r_g(smbh_mass * u.solMass, inner_disk_outer_radius),
        0 * u.m,
    )
    return time_gw_normalization.si.value

