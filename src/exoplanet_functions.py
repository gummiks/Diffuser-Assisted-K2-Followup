import numpy as np
import pandas as p
import pandas as pd
import astropy.constants as aconst
from skyfield.api import load, Star, Angle
import astropy.units as u
import mcmc_utils

def calc_radec_w_proper_motion(ra,de,pmra,pmde,epoch=2018.):
    planets = load('/Users/gks/Dropbox/mypylib/notebooks/GIT/HETobs/de421.bsp')
    ts = load.timescale()
    earth = planets['earth']
    t = ts.utc(epoch)
    t_2000 = ts.utc(2000)
    ra_ang = Angle(degrees=ra)
    de_ang = Angle(degrees=de)
    star_obj = Star(ra=ra_ang,dec=de_ang,ra_mas_per_year=pmra,dec_mas_per_year=pmde)
    astrometric = earth.at(t).observe(star_obj)
    ra_out,dec_out,dist_out = astrometric.radec()
    ra_new = ra_out.to(u.degree).value
    de_new = dec_out.to(u.degree).value
    print("Old:",ra,de)
    print("New:",ra_new,de_new)
    return ra_new, de_new


def teff_v_k(v_minus_k,size=10000,return_dist = False):
    """
    Calculate Teff according to the relation in Boyajian et al. 2013
    See also Yee et al. 2017 (Specmatch-emp paper)
    
    INPUT:
        v_minus_k - V-K in magnitudes
        size - size of the array for the distribution
    
    EXAMPLE:
        V = 12.806 #+- 0.046
        K = 9.496  #+-0.017
        v_minus_k = V-K
        df, teff = teff_v_k(v_minus_k,return_dist=True)
    """
    a0 = (8649,28)
    a1 = (-2503,35)
    a2 = (442,12)
    a3 = (-31.7,1.5)
    constants = [a0,a1,a2,a3]
    a = [np.random.normal(a[0],a[1],size=size) for a in constants]
    
    teff = a[0] + a[1]*v_minus_k + a[2]*v_minus_k**2. + a[3]*v_minus_k**3.
    df = pd.DataFrame(teff,columns=["teff"])
    df_medvals = mcmc_utils.calc_medvals2(df)
    
    if return_dist:
        return df_medvals, teff
    else:
        return df_medvals

def teq(teff,aRs,a=0.3):
    """
    Calculate the planet effective temperature, Teq in K.
    
    INPUT:
        teff - effective temperature of star in K
        a    - albedo (=0 for blackbody)
        aRs  - a/R*s (orbital semi-major axis)
        
    OUTPUT:
        Teq in K
    
    Notes:
        https://en.wikipedia.org/wiki/Planetary_equilibrium_temperature
    """
    return teff*(1.-a)*np.sqrt(1./(2.*aRs))

def calc_transit_duration(P,RpRs,aRs,i):
    """
    A function to calculate the transit duration

    INPUT:
    P - days
    RpRs - 
    aRs - 
    i - in radians

    OUTPUT:
    transit duration in days
    
    NOTES:
    See Seager et al. 2002
    """
    aa = (1.+RpRs)**2.
    bb = (aRs*np.cos(i))**2.
    cc = 1.-(np.cos(i))**2.
    dd = np.sqrt((aa-bb)/cc)
    
    return (P/(np.pi))*np.arcsin((1./aRs)*dd)

def calc_impact_parameter(aRs,i):
    """
    INPUT:
    aRs -
    i - in degrees
    NOTES:
    See Seager et al. 2002
    """
    return aRs*np.cos(np.deg2rad(i))

def calc_logg_from_r_m(r,m):
    """
    INPUT:
    - r in solar radii
    - m in solar masses
    
    OUTPUT:
    - logg (log10(cgs gravity))
    """
    rad = r*aconst.R_sun.value
    mass = m*aconst.M_sun.value
    logg = np.log10((aconst.G.value*mass/(rad**2.))*100.)
    return logg

def rho_star_from_P_and_aRs(P,aRs,return_cgs=True):
    """
    Density of the star from period of planet P, and aRs from transit observations.
    
    INPUT:
    - P in days
    - aRs - unitless

    OUTPUT:
    density of the star
    FLAG: return_cgs:
    -- if true: return in g/cm^3
    -- else: return in solar densities
    
    NOTE:
    - Assumes that eccentricity of the planet is 0. Otherwise not valid
    - Uses equation from Seager et al. 2003
    
    EXAMPLE:
        rho_star_from_P_and_aRs(0.2803,2.3,return_cgs=True) # Should be 2.92
        
        P = np.random.normal(0.2803226,0.0000013,size=1000)
        a = np.random.normal(2.31,0.08,size=1000)
        dens = rho_star_from_P_and_aRs(P,a,return_cgs=True)
        df = pd.DataFrame(dens)
        mcmc_utils.calc_medvals2(df)
    """
    per = P*24.*60.*60 # Change to seconds
    rho_sun = aconst.M_sun.value/(4.*np.pi*(aconst.R_sun.value**3.)/3.) # kg/m^3

    # Density from transit, see equation in Seager et al. 2003
    rho_star = 3.*np.pi*(aRs**3.)/(aconst.G.value*(per**2.)) # kg/m^3
    if return_cgs:
        return rho_star * (1000./(100.*100.*100.)) # g/cm^3
    else:
        return rho_star/rho_sun

def aRs_from_rho_and_P(rho,P):
    """
    Calculate aRs from the stellar density, rho
    
    INPUT:
    - rho, density of star in g/cm^3
    - P, period of planet in days
    
    OUTPUT:
    - a/Rs - unitless semimajor axis
    
    EXAMPLE:
    aRs_from_rho_and_P(2.98,0.2803)# - should be 2.31
    """
    per = P*24.*60.*60 # Change to seconds
    rho_star = rho*(100.*100.*100.)/(1000.) # g/cm^3 -> kg/m^3
    G = aconst.G.value #
    aRs = ((rho_star*G*(per**2.))/(3.*np.pi))**(1./3.)
    return aRs
    
def rho_from_Ms_Rs(Ms,Rs,return_cgs=True):
    """
    Simple mean density of star from mass and radius
    
    INPUT:
    - Ms - mass of star in solar masses
    - Rs - radius of star in solar radii
    
    OUTPUT:
    density of the star
    FLAG: return_cgs:
    -- if true: return in g/cm^3
    -- else: return in solar densities
    
    EXAMPLE:
    rho_from_Ms_Rs(0.662,0.674) # should be 3.048 in cgs
    """
    _Ms = Ms*aconst.M_sun.value # kg
    _Rs = Rs*aconst.R_sun.value # m
    
    rho_star = _Ms/(4.*np.pi*_Rs**3./3.) # kg/m^3
    rho_sun = aconst.M_sun.value/(4.*np.pi*(aconst.R_sun.value**3.)/3.) # kg/m^3

    if return_cgs:
        return rho_star * (1000./(100.*100.*100.)) # g/cm^3
    else:
        return rho_star/rho_sun

def logg_from_Ms_Rs(Ms,Rs):
    """
    INPUT:
    - Ms in units of solar masses
    - Rs in units of solar radii
    
    OUTPUT:
    - log(g) where g is in cgs units (g/s^2)
    
    EXAMPLE:
        logg_from_Ms_Rs(0.662,0.674) # should be 4.60

        M = np.random.normal(0.662,0.022,size=1000)
        R = np.random.normal(0.674,0.039,size=1000)
        df = pd.DataFrame(logg_from_Ms_Rs(M,R))
        mcmc_utils.calc_medvals2(df)
    """
    _Ms = Ms*aconst.M_sun.value # kg
    _Rs = Rs*aconst.R_sun.value # m
    g  = aconst.G.value*_Ms/(_Rs**2.)*100. # SI units, # m/s^2->cm/s^2
    return np.log10(g)


def insolation_flux_from_aRs_and_Teff(aRs,Teff):
    """
    Return the insolation flux in Earth Insolation fluxes
    Earth insolation flux is: 1361 W/m^2    
   
    INPUT:
    - aRs - unitless
    - Teff - effective temperature of the host star
    
    OUTPUT:
    - F in units of Earth's insolation flux (i.e., divided by 1361 W/m^2)
    
    EXAMPLE:
    insolation_flux_from_aRs_and_Teff(aconst.au.value/aconst.R_sun.value,5770.)
    - should be close to 1.
    """
    flux = aconst.sigma_sb.value*Teff**4./(aRs**2.) # W/K^4 /m^2
    earth_insoliation_flux = aconst.L_sun.value/(4.*np.pi*aconst.au.value**2.) # W / m^2
    return flux/earth_insoliation_flux
