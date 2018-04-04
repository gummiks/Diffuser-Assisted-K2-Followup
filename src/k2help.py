from __future__ import print_function
import numpy as np
import everest
from astroquery.simbad import Simbad
import k2plr
import nexopl
import sys
import pandas as pd
import astropy
import glob
import os
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import matplotlib.pyplot as plt
import utils
import wget
import os
import glob
import re
import scipy.signal

KEPLER_JD_OFFSET = 2454833.0

def median_filter_and_sigma_clip_noflat(t,f,window=49,sigma_upper=15.,sigma=15.,return_mask=True,return_flat=True):
    """
    Use a median filter to flatten transit data and then sigma clip bad points. Returns mask

    INPUT:
        t - array of times
        f - array of fluxes (normalized)
        window - window for median filter
        sigma_upper -
        sigma -

    OUTPUT:
        t - clipped times
        f - flattened and clipped fluxes
        m - mask

    EXAMPLE:
        t = EP.star.time
        f = EP.star.flux/np.nanmedian(EP.star.flux)
        tm,fm,m = median_filter_and_sigma_clip(t,f,sigma=15.,sigma_upper=3.)

        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(tm,fm)
    """
    f_filt = f-scipy.signal.medfilt(f,window)+1.
    data = astropy.stats.sigma_clipping.sigma_clip(f_filt,sigma_upper=sigma_upper,sigma=sigma)
    m = np.where(data.mask)
    if return_mask:
        return np.delete(t,m), np.delete(f_filt,m), m[0], data.mask
    else:
        return np.delete(t,m), np.delete(f_filt,m)

def median_filter_and_sigma_clip(t,f,window=49,sigma_upper=4.,sigma=15.,return_mask=True,return_flat=True):
    """
    Use a median filter to flatten transit data and then sigma clip bad points. Returns mask

    INPUT:
        t - array of times
        f - array of fluxes (normalized)
        window - window for median filter
        sigma_upper -
        sigma -

    OUTPUT:
        t - clipped times
        f - flattened and clipped fluxes
        m - mask

    EXAMPLE:
        t = EP.star.time
        f = EP.star.flux/np.nanmedian(EP.star.flux)
        tm,fm,m = median_filter_and_sigma_clip(t,f,sigma=15.,sigma_upper=3.)

        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(tm,fm)
    """
    f_filt = f-scipy.signal.medfilt(f,window)+1.
    data = astropy.stats.sigma_clipping.sigma_clip(f_filt,sigma_upper=sigma_upper,sigma=sigma)
    m = np.where(data.mask)
    if return_mask:
        return np.delete(t,m), np.delete(f_filt,m), m[0], data.mask
    else:
        return np.delete(t,m), np.delete(f_filt,m)

def get_epic_numbers_from_list(myarray):
    """
    Return a list of epic numbers

    EXAMPLE: 
        k2help.get_epic_numbers_from_list(df_all._pl_name)
    """
    return [int(re.findall(r'\d{9}',name)[0]) for name in myarray if len(re.findall(r'\d{9}',name))>0]


def make_dir(dirname,verbose=True):
    try:
        os.makedirs(dirname)
        if verbose==True: print("Created folder:",dirname)
    except OSError:
        if verbose==True: print(dirname,"already exists. Skipping")

def get_k2_star(epicid):
    """
    Uses k2plr
    """
    client = k2plr.API()
    return client.k2_star(epicid)

def get_k2_star_ra_dec(epicid):
    k2star = get_k2_star(epicid)
    return k2star.k2_ra, k2star.k2_dec

def get_epic_info(epiclist,verbose=True):
    """
    Return a dataframe for a list of epic numbers
    """
    df = pd.DataFrame()
    for i,epicname in enumerate(epiclist):
        if verbose: print(i,epicname)
        k2star = k2help.get_k2_star(epicname)
        df.set_value(i,"epic",epicname)
        df.set_value(i,"pl_name","epic"+str(int(epicname)))
        df.set_value(i,"st_rad",k2star.rad)
        df.set_value(i,"st_mass",k2star.mass)
        df.set_value(i,"st_optmag",  k2star.kp)
        df.set_value(i,"st_teff",k2star.teff)
        df.set_value(i,"st_vj",k2star.vmag)
        df.set_value(i,"st_ic",k2star.imag)
        df.set_value(i,"st_h",k2star.hmag)
        df.set_value(i,"st_k",k2star.kmag)
        df.set_value(i,"st_j",k2star.jmag)
        df.set_value(i,"ra",k2star.k2_ra)
        df.set_value(i,"dec",k2star.k2_dec)
        ra_str, dec_str = utils.radecDeg2hourangleHMS(k2star.k2_ra,k2star.k2_dec)
        df.set_value(i,"ra_str",ra_str)
        df.set_value(i,"dec_str",dec_str)
        df.set_value(i,"st_pmra",k2star.pmra)
        df.set_value(i,"st_pmdec",k2star.pmdec)
    return df

def get_transit_epochs(time,t0,period):
    """
    Get when a planet is transiting inside an array called *time*

    INPUT:
        time
        t0
        period

    OUTPUT:
        Array of transit midpoints within the array time

    NOTES:
    """
    transit_number = int(np.ceil((max(time) - min(time)) / period))
    print(transit_number,"transits in data")
    #print(transit_number)
    n = np.arange(0.,transit_number)
    #print(n)
    transit_epochs = t0 + n*period
    return transit_epochs


def expand_epic_file(filename,sort_column="imag"):
    """
    A function to get more data on k2 target files
    """
    fp = filepath.FilePath(filename)
    fp.add_suffix("_expanded")
    savename = str(fp)

    df = pd.read_csv(filename,sep=",")
    for i, epicnumber in enumerate(df.epic.values):
        k2star = get_k2_star(epicnumber)
        df.set_value(i,"vmag",k2star.vmag)
        df.set_value(i,"imag",k2star.imag)
        df.set_value(i,"jmag",k2star.jmag)
        df.set_value(i,"hmag",k2star.hmag)
        df.set_value(i,"kmag",k2star.kmag)
        df.set_value(i,"teff",k2star.teff)
        df.set_value(i,"mass",k2star.mass)
        df.set_value(i,"rad",k2star.rad)
        df.set_value(i,"pmra",k2star.pmra)
        df.set_value(i,"omdec",k2star.pmdec)
        print(i,epicnumber,k2star.imag,k2star.jmag,k2star.hmag,k2star.teff,k2star.rad)
    df = df.sort_values(sort_column).reset_index(drop=True)
    df.to_csv(savename,index=False)

def get_epic_campaign(epicname):
    return everest.missions.k2.Season(epicname)

def get_epic_id(k2name):
    """
    Put in K2-name, and get epicID
    """
    query = Simbad.query_objectids(k2name)
    try:
        epic_id = [i for i in np.array(query["ID"]) if "EPIC" in i][0][5:]
        epic_id = get_epic_numbers_from_list([epic_id])[0]
        return int(epic_id)
    except TypeError:
        print("No epic ID found")
        return np.nan

def deg2hourangle(deg):
    return (deg*u.deg).to(u.hourangle)

class K2Star(object):
    """
    
    EXAMPLE:
        ss = K2Star(246485787)
        planet = ss.get_nexopl()
        planet.plot_planet_param_table()
    """

    def __init__(self,epicid):
        client = k2plr.API()
        self.s = client.k2_star(epicid)

    def get_nexopl(self):
        #df = pd.read_csv("/Users/gks/Dropbox/mypylib/k2star2nexopl.csv")
        #attributes = df.nexopl.values
        attributes = nexopl.NExSciEphem().exo_attributes.values
        df_exo = pd.DataFrame(columns=attributes)
        
        # Populating df
        df_exo.set_value(0,"pl_hostname","EPIC "+str(self.s.id))
        df_exo.set_value(0,"pl_name","EPIC "+str(self.s.id))
        #df_exo.set_value(0,"ra_str",utils.radeg2hourangle(self.s.k2_ra))
        df_exo.set_value(0,"ra_str",utils.radecDeg2hourangleHMS(self.s.k2_ra,self.s.k2_dec,sep_doubledots=False)[0])
        #df_exo.set_value(0,"dec_str",self.s.k2_dec)
        df_exo.set_value(0,"dec_str",utils.radecDeg2hourangleHMS(self.s.k2_ra,self.s.k2_dec,sep_doubledots=False)[1])
        df_exo.set_value(0,"st_mass",self.s.mass)
        #df_exo.set_value(0,"st_masserr1",self.s.e_mass)
        #df_exo.set_value(0,"st_masserr2",self.s.e_mass)
        df_exo.set_value(0,"st_rad",self.s.rad)
        #df_exo.set_value(0,"st_raderr1",self.s.e_rad)
        #df_exo.set_value(0,"st_raderr2",self.s.e_rad)
        df_exo.set_value(0,"st_teff",self.s.teff)
        #df_exo.set_value(0,"st_tefferr1",self.s.e_teff)
        #df_exo.set_value(0,"st_tefferr2",self.s.e_teff)     
        df_exo.set_value(0,"st_metfe",self.s.feh)
        #df_exo.set_value(0,"st_metfeerr1",self.s.e_feh)
        #df_exo.set_value(0,"st_metfeerr2",self.s.e_feh) 
        df_exo.set_value(0,"st_logg",self.s.logg)
        #df_exo.set_value(0,"st_loggerr1",self.s.e_logg)
        #df_exo.set_value(0,"st_loggerr2",self.s.e_logg)
        df_exo.set_value(0,"st_uj",self.s.umag)
        df_exo.set_value(0,"st_bj",self.s.bmag)
        df_exo.set_value(0,"st_vj",self.s.vmag)
        df_exo.set_value(0,"st_rc",self.s.rmag)
        df_exo.set_value(0,"st_ic",self.s.imag)
        df_exo.set_value(0,"st_j",self.s.jmag)
        df_exo.set_value(0,"st_h",self.s.hmag)
        df_exo.set_value(0,"st_k",self.s.kmag)

        return nexopl.NExopl(attributes,df_exo)


class EverestGK(everest.Everest):

    def __init__(self,ID):
        everest.Everest.__init__(self,ID)

    def get_flattened_flux(self,plot=True,win=49):
        self.flux_savgol = everest.math.SavGol(np.delete(self.flux,self.mask),win=win)
        self.flux_savgol_norm = self.flux_savgol/np.median(self.flux_savgol)
        if plot==True:
            self.fig, self.ax = plt.subplots()
            self.ax.plot(np.delete(self.time,self.mask),self.flux_savgol_norm,lw=0,marker="o",markersize=3,alpha=0.5)
            for label in self.ax.get_xticklabels():
                label.set_fontsize(12)
            for label in self.ax.get_yticklabels():
                label.set_fontsize(12)
            
    def plot_folded2(self,t0,period,dur=0.2,title=""):
        time = np.delete(self.time,self.mask)
        flux = self.flux_savgol_norm
        tfold = (time - t0 - period / 2.) % period - period / 2.
        inds = np.where(np.abs(tfold) < 2 * dur)[0]
        self.time_folded = tfold[inds]
        self.flux_folded = flux[inds]
        fig, ax = plt.subplots()
        ax.plot(self.time_folded,self.flux_folded,marker="o",markersize=3,alpha=0.3,lw=0)
        ax.set_title(title,y=1.03)
        #ax.set_ylim(0.97,1.02)
        #ax.set_title("TRAPPIST-1b, Vanderburg quicklook data",y=1.03)
        #ax.set_xlabel("BJD")
        #ax.set_ylabel("Relative Flux")

    #def get_whitened_flux(self,t0,period,dur=0.2,sigma=7.):
    def get_whitened_flux(self):
        """
        To get the whitened flux, i.e., before folding
        """
        # Mask the planet
        #self.mask_planet(t0, period, dur)
        # For some reason this is bad. Jan 8th 2018

        # Whiten
        gp = everest.gp.GP(self.kernel, self.kernel_params, white = False)
        gp.compute(self.apply_mask(self.time), self.apply_mask(self.fraw_err))
        med = np.nanmedian(self.apply_mask(self.flux))
        y, _ = gp.predict(self.apply_mask(self.flux) - med, self.time)
        self.fwhite = (self.flux - y)
        self.fwhite /= np.nanmedian(self.fwhite)
        return self.time, self.fwhite

    def get_masked_indices_in_transit(self,t, t0, period, dur = 0.2):
        '''
        Mask all of the transits/eclipses of a given planet/EB. After calling
        this method, you must re-compute the model by calling :py:meth:`compute`
        in order for the mask to take effect.
        
        :param float t0: The time of first transit (same units as light curve)
        :param float period: The period of the planet in days
        :param foat dur: The transit duration in days. Default 0.2
        
        '''
        mask = []
        t0 += np.ceil((t[0] - dur - t0) / period) * period
        for time in np.arange(t0, t[-1] + dur, period):
          mask.extend(np.where(np.abs(t - time) < dur / 2.)[0])
        return mask

    def get_folded_transit(self, t0, period, dur = 0.2,sigma=7.,plot=True,ax=None,sort_time=True):
        '''
        Extending plot_folded() from the original EVEREST package.
        
        INPUT:
            t0 - the midpoint of the transit
            period - period of the transit in days
            dur - duration of window to use in Everest.compute()
            sigma - numbers of sigma to clip (outliers)
            plot - if == True, plot a plot_folded plot

        OUTPUT:
            self.time_phased - the phased time
            self.flux_phased - the phased flux
            
        NOTES:
        Can access:
        self.time_phased, self.flux_phased - the time and flux (phased)
        '''
        # Mask the planet
        self.mask_planet(t0, period, dur)

        # Whiten
        gp = everest.gp.GP(self.kernel, self.kernel_params, white = False)
        gp.compute(self.apply_mask(self.time), self.apply_mask(self.fraw_err))
        med = np.nanmedian(self.apply_mask(self.flux))
        y, _ = gp.predict(self.apply_mask(self.flux) - med, self.time)
        fwhite = (self.flux - y)
        fwhite /= np.nanmedian(fwhite)

        # Fold
        tfold = (self.time - t0 - period / 2.) % period - period / 2. 

        # Crop
        inds = np.where(np.abs(tfold) < 2 * dur)[0]
        
        #self.time_masked, self.flux_flattened_masked - the time and flux_flattened (not phased)
        #m = astropy.stats.sigma_clip(fwhite,sigma=sigma).mask
        #self.time_masked = self.time[~m]
        #self.flux_flattened_masked = fwhite[~m]
        # OLD:
        self.time_phased = tfold[inds]
        self.flux_phased = fwhite[inds]
        
        # sigma clip
        m = astropy.stats.sigma_clipping.sigma_clip(self.flux_phased,sigma=sigma).mask
        self.flux_phased = self.flux_phased[~m]
        self.time_phased = self.time_phased[~m]


        if sort_time:
            df = pd.DataFrame(zip(self.time_phased,self.flux_phased),columns=["time","flux"]).sort_values("time").reset_index(drop=True)
            self.time_phased = df.time.values
            self.flux_phased = df.flux.values
        
        if plot:
            # Plot
            if ax ==None:
                fig, ax = plt.subplots(1, figsize = (9, 5))
            ax.plot(self.time_phased, self.flux_phased, 'k.', alpha = 0.5)

            # Get ylims
            yfin = np.delete(self.flux_phased, np.where(np.isnan(self.flux_phased)))
            lo, hi = yfin[np.argsort(yfin)][[3,-3]]
            pad = (hi - lo) * 0.1
            ylim = (lo - pad, hi + pad)
            ax.set_ylim(ylim[0],ylim[1])

            # Appearance
            ax.set_xlabel(r'Time (days)')
            ax.set_ylabel(r'Normalized Flux')
            ax.set_title("EPIC "+str(self.ID))
            ax.minorticks_on()

        return self.time_phased, self.flux_phased

def get_epic_detrended_list(path="/storage/work/gws5257/K2Detrending/everest2/k2/c13"):
    regex = path+"/*/*/*dvs.pdf"
    detrended = glob.glob(regex)
    epic_list = [int(filename.split(os.sep)[-1].split("_")[4][0:9]) for filename in detrended]
    return epic_list

def fold_transit_everest(tt,flux,t0,period,dur=0.2):
    """
    Uses the everest method
    """
    tfold = (tt - t0 - period / 2.) % period - period / 2. 
    # Crop
    inds = np.where(np.abs(tfold) < 2 * dur)[0]
    x = tfold[inds]
    y = flux[inds]
    df = pd.DataFrame(zip(x,y),columns=["time","flux"]).sort_values("time").reset_index(drop=True)
    return df.time.values, df.flux.values
    #plt.plot(df.time,df.flux)

def mask_planet_everest(tt, t0, period, dur = 0.2):
    '''
    Mask all of the transits/eclipses of a given planet/EB. After calling
    this method, you must re-compute the model by calling :py:meth:`compute`
    in order for the mask to take effect.

    :param float t0: The time of first transit (same units as light curve)
    :param float period: The period of the planet in days
    :param foat dur: The transit duration in days. Default 0.2

    '''
    mask = []
    t0 += np.ceil((tt[0] - dur - t0) / period) * period
    for t in np.arange(t0, tt[-1] + dur, period):
        mask.extend(np.where(np.abs(tt - t) < dur / 2.)[0])
    return mask

def savgol_sigma_clip(x,sigma_upper=3,sigma_lower=3,sigma=7,win=23):
    """
    Sigma clip by savgol-flattening first.
    
    INPUT:
     x - input array
    
    OUTPUT:
     x - the sigma-clipped array
     m - the mask
     
    NOTES:
     x, m = savgol_sigma_clip(f,sigma=6,win=23)
     plt.plot(t,f)
     plt.plot(t[~m],x)
    """
    x_savgol = everest.math.SavGol(x,win=win)
    data = astropy.stats.sigma_clipping.sigma_clip(x_savgol,sigma_upper=sigma_upper,sigma=sigma,sigma_lower=sigma_lower)
    m = data.mask
    return x[~m], m
