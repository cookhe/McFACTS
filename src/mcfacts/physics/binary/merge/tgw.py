import astropy.constants as const
import astropy.units as u

def normalize_tgw(smbh_mass):
    """Gravitational wave timescale normalization.

    Calculate the normalization for timescale of merger (in yrs) due to GW emission.
    From Peters(1964):
    
    t_gw approx (5/128)* c^5/G^3 *a_b^4/(M_{b}^{2}mu_{b}) assuming ecc=0.0.
    
    For a_b in units of r_g=GM_smbh/c^2 we find
    
    t_gw=(5/128)*(G/c^3)*(a/r_g)^{4} *(M_s^4)/(M_b^{2}mu_b)

    Put bin_mass_ref in units of 10Msun is a reference mass.
    reduced_mass in units of 2.5Msun.
    
    Parameters
    ----------
    smbh_mass : float
        Mass of the supermassive black hole.

    Returns
    -------
    float
        normalization to gravitational wave timescale
    """

    G = const.G
    c = const.c
    mass_sun = const.M_sun
    year = 3.1536e7
    bin_mass_ref = 10.0
    reduced_mass = 2.5
    norm = (5.0/128.0)*(G/(c**(3)))*(smbh_mass**(4))*mass_sun/((bin_mass_ref**(2))*reduced_mass)
    time_gw_normalization = norm/year

    return time_gw_normalization.si.value
