from pytransit import MandelAgol as MA
from scipy.optimize import minimize
import matplotlib as mpl
from matplotlib.pyplot import figure, subplots, subplot
from matplotlib.backends.backend_pdf import PdfPages
import math as mt
from exotk.utils.orbits import as_from_rhop, i_from_baew
from exotk.utils.likelihood import ll_normal_es_py, ll_normal_ev_py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import k2help
import blsMOD
import exoplanet_functions
from matplotlib.gridspec import GridSpec

import k2help
import astropy
import everest
import utils

PW,PH = 12., 12.
mpl.rc('axes', labelsize=7, titlesize=8)
mpl.rc('font', size=6)
mpl.rc('xtick', labelsize=7)
mpl.rc('ytick', labelsize=7)
mpl.rc('lines', linewidth=1)

class EPICBLSMULTI(object):
    KEPLER_JD_OFFSET = 2454833.0
    
    limbdark = [0.4,0.1]
    mask = []
    
    def __init__(self,epicname=None,time=None,flux=None):
        self.epicname = epicname
        self.time = time
        self.flux = flux
        if epicname is not None:
            self.star = k2help.EverestGK(epicname)
            self.star.get_flattened_flux(plot=False)
            #self.flux_savgol = everest.math.SavGol(np.delete(self.star.flux,self.star.mask),win=win)
            #self.flux_savgol_norm = self.flux_savgol/np.median(self.flux_savgol)
        if self.time is None and self.flux is None:
            self.star.transitmask = np.append(self.star.transitmask,self.mask)
            self.t    = np.delete(self.star.time,self.star.mask)
            self.f    = self.star.flux_savgol_norm
        else:
            self.t    = self.time#np.delete(self.star.time,self.star.mask)
            self.f    = self.flux#self.star.flux_savgol_norm
        
    def __call__(self,  qmi=0.02,
                        qma = 0.25,
                        df = 0.0001,
                        nf = 10000,
                        nb = 1000,
                        fmin = 1./50.,
                        dur=0.2,
                        period_range=None,
                        plot=False,
                        win=49,
                        period_fit_prior=None,
                        t0_fit_prior=None):
        self.bb = blsMOD.blsMOD(self.t,self.f, nf, fmin, df, nb, qmi, qma,period_range)
        print("Computing bls...")
        self.bb.compute()
        self.pv_bls = [self.bb._best_period,self.bb._epoch,np.sqrt(self.bb._depth),as_from_rhop(2.5,self.bb._best_period),0.1]
        self.fit_transit(pv_init=self.pv_bls,time=self.t,flux=self.f,period_fit_prior=period_fit_prior,t0_fit_prior=t0_fit_prior)
        print("Per=",self.pv_min[0],"epoch=",self.pv_min[1])
        if plot:
            self.fig, self.ax = plt.subplots(nrows=4)
            self.bb.plot_power_spectrum(inverse=True,ax=self.ax.flat[0])
            self.bb.plot_folded(ax=self.ax.flat[1])
            self.bb.plot_transits(ax=self.ax.flat[2])
            
            if self.time is not None:            
                self.t_folded,self.f_folded = k2help.fold_transit_everest(self.t,self.f,t0=self.bb._epoch,period=self.bb._best_period,dur=dur)
                self.ax.flat[3].plot(self.t_folded,self.f_folded,"k.",label="BLS",alpha=0.5)
                self.t_folded,self.f_folded = k2help.fold_transit_everest(self.t,self.f,t0=self.pv_min[1],period=self.pv_min[0],dur=dur)
                self.ax.flat[3].plot(self.t_folded,self.f_folded,"r.",label="Best Fit",markersize=5)
                self.ax.flat[3].legend(loc="lower left",fontsize=8)
            else:
                self.t_folded,self.f_folded = self.star.get_folded_transit(t0=self.bb._epoch,period=self.bb._best_period,dur=dur,plot=False);
                self.ax.flat[3].plot(self.t_folded,self.f_folded,"k.",label="BLS",alpha=0.5)
                self.t_folded,self.f_folded = self.star.get_folded_transit(t0=self.pv_min[1],period=self.pv_min[0],dur=dur,plot=False);
                self.ax.flat[3].plot(self.t_folded,self.f_folded,"r.",label="Best Fit",markersize=5)
                self.ax.flat[3].legend(loc="lower left",fontsize=8)
            
    
    def fit_transit(self,pv_init,time,flux,period_fit_prior=None,t0_fit_prior=None):
        """
        Fit folded transit
        """
        pbls = pv_init[0]
        t0bls = pv_init[1]

        self.tm = MA(supersampling=30, nthr=1) 
        #self.time_fitting = self.t#self.df_folded.time
        #self.flux_fitting = self.f#self.df_folded.flux

        def minfun(pv):
            if any(pv<=0) or (pv[3] <= 1) or (pv[4] > 1) or not (0.75<pv[0]<80.) or abs(pv[0]-pbls) > 0.1: return np.inf
            if period_fit_prior is not None:
                if (pv[0] < period_fit_prior[0]) or (pv[0] > period_fit_prior[1]):
                    return np.inf
            if t0_fit_prior is not None:
                if (pv[1] < t0_fit_prior[0]) or (pv[1] > t0_fit_prior[1]):
                    return np.inf
            elif abs(pv[1]-t0bls) > 0.4: return np.inf
            return -ll_normal_ev_py(flux, self.transit_model(pv,time), 1e-5)
        
        self.mr = minimize(minfun, pv_init, method='Nelder-Mead')
        mr = self.mr
        #mr = minimize(minfun, self.pv_bls, method='powell')
        self.pv_min = self.mr.x
        print(mr.message)
        lnlike, x = -mr.fun, mr.x

    def mask_planet(self,planet=None,P=None,T0=None,dur=0.2):
        print("Masking planet with:")
        if planet is not None:
            print("Using planet",planet._pl_name,"to mask")
            P = planet._pl_orbper
            T0 = planet._pl_tranmid-k2help.KEPLER_JD_OFFSET
        print("T0=",T0)
        print("P=",P)
        self.mask = k2help.mask_planet_everest(self.t,T0,P,dur=dur)
        self.t = np.delete(self.t,self.mask)
        self.f = np.delete(self.f,self.mask)

    def get_nexoplanet(self,pv=None,epicname=None):
        if pv is None:
            pv = self.pv_min
        self._pl_orbper  = pv[0]
        self._pl_tranmid = pv[1]+self.KEPLER_JD_OFFSET
        self._pl_ratror  = pv[2]
        self._pl_ratdor  = pv[3]
        self._pl_orbincl = np.rad2deg(mt.acos(pv[4]/pv[3]))
        self._pl_trandur = exoplanet_functions.calc_transit_duration(self._pl_orbper,
                                                      self._pl_ratror,
                                                      self._pl_ratdor,
                                                      np.deg2rad(self._pl_orbincl))

        if epicname is not None:
            self.k2star = k2help.K2Star(epicname)
            self.planet = self.k2star.get_nexopl()
            self.planet._pl_tranmid = self._pl_tranmid 
            self.planet._pl_orbper  = self._pl_orbper
            self.planet._pl_trandur = self._pl_trandur
            self.planet._pl_ratror  = self._pl_ratror
            self.planet._pl_ratdor  = self._pl_ratdor
            self.planet._pl_orbincl = self._pl_orbincl
            self.planet._pl_orbeccen = 0.
            self.planet._pl_trandep = self.planet._pl_ratror**2.
            return self.planet
        else:
            print("pl_orbper",self._pl_orbper)
            print("pl_tranmid",self._pl_tranmid)
            print("pl_ratror",self._pl_ratror)
            print("pl_ratdor",self._pl_ratdor)
            print("pl_orbincl",self._pl_orbincl)
            print("pl_trandur",self._pl_trandur)  

    def transit_model(self, pv, time):
        """
        Define the transit model for a given pv vector

        INPUT:
            pv[0] - period
            pv[1] - epoch (0)
            pv[2] - Rp/Rs (sqrt(depth))
            pv[3] - a/Rs (estimate from stellar density)
            pv[4] - impact param
        """
        #time = self.time_fitting if time is None else time
        #time
        p,t0,RpRs,aRs,dur = pv

        _i = mt.acos(pv[4]/pv[3])
        return self.tm.evaluate(time, RpRs, self.limbdark, t0, p, aRs, _i,e=0.0, w=0.0, c=0.0)

    def plot_lc_simple(self,pv=None,time=None,flux=None,ax=None,OFFSET=0.99):
        if pv is None:
            pv = self.pv_min
        if time is None or flux is None:
            time = self.t
            flux = self.f
        if ax==None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        self.ax.plot(time,flux,label="Data")
        self.ax.plot(time,flux-self.transit_model(pv,time)+OFFSET,label="Residual")
        self.ax.plot(time,self.transit_model(pv,time),label="Best fit model")
        self.ax.legend(loc="lower left",fontsize=9)

    def plot_lc(self,pv,time=None,flux=None,ax=None,OFFSET=0.99,p_fold=None,t0_fold=None,dur_fold=0.2):
        if time is None or flux is None:
            time = self.t
            flux = self.f
        if ax==None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        if p_fold is not None and t0_fold is not None:
            time, flux = k2help.fold_transit_everest(time,flux,t0=t0_fold,period=p_fold,dur=dur_fold)
            pv[1] = 0.
        tt = np.linspace(time.min(),time.max(),2000)
        self.ax.plot(time,flux,label="Data")
        self.ax.plot(time,flux-self.transit_model(pv,time)+OFFSET,label="Residual")
        self.ax.plot(time,self.transit_model(pv,time),label="Best fit model")
        self.ax.legend(loc="lower left",fontsize=9)
        self.ax.grid(lw=0.5,alpha=0.3)

    def plot_lc_min(self,time=None,flux=None,ax=None,OFFSET=0.99,fold_best=True,dur_fold=0.2):
        if fold_best:
            p_fold=self.pv_min[0]
            t0_fold=self.pv_min[1]
        else:
            p_fold=None
            t0_fold=None
        self.plot_lc(self.pv_min,time,flux,ax=ax,OFFSET=OFFSET,p_fold=p_fold,t0_fold=t0_fold,dur_fold=dur_fold)



class K2SFFBLS(object):
    """
    "/Users/gks/Dropbox/mypylib/notebooks/OBS_PLANNING/epic_catalogue/DETRENDED_FROM_CLUSTER/hlsp_k2sff_k2_lightcurve_247773181-c13_kepler_v1_llc-default-aper.txt",sigma=10
    """
    def __init__(self,filename,epicname=None,sigma_upper=4.,sigma=7.,plot_sigma_clip=False,):
        self.filename = filename
        self.df = pd.read_csv(self.filename,skiprows=0)
        self.t = self.df.ix[:,0].index.values
        self.f = self.df.ix[:,0].values
        self.epicname = k2help.get_epic_numbers_from_list([self.filename])[0]
        self.sigma_clip_fluxes(sigma_upper,sigma,plot_sigma_clip=plot_sigma_clip)
        self.EP = EPICBLSMULTI(epicname=None,time=self.t_flat,flux=self.f_flat)
        
    def sigma_clip_fluxes(self,sigma_upper=3.,sigma=6.,plot_sigma_clip=False):
        self.data = astropy.stats.sigma_clipping.sigma_clip(everest.math.SavGol(self.f),sigma_upper=sigma_upper,sigma=sigma)
        m = self.data.mask
        self.t_flat = self.t[~m]
        self.f_flat = everest.math.SavGol(self.f)[~m]
        if plot_sigma_clip:
            plt.plot(self.t,everest.math.SavGol(self.f))
            plt.plot(self.t_flat,self.f_flat)

    def __call__(self,plot_fit_panel=True,plot_folded_panel=True,use_BLS_values=False,period_range=(1.,50.),period_fit_prior=None,t0_fit_prior=None,dur=0.2):
        self.EP(period_range=period_range,nf=10000,plot=plot_fit_panel,period_fit_prior=period_fit_prior,t0_fit_prior=t0_fit_prior,dur=dur)
        self.planet = self.EP.get_nexoplanet(epicname=self.epicname)
        #if plot_folded_panel:
        #    fig, ax = plt.subplots()
        #    ax.plot(*gkastro.fold_data(self.t_flat,self.f_flat,self.EP.bb._epoch,self.EP.bb._best_period),marker="o",lw=0,markersize=3,label="BLS values")
        #    ax.plot(*gkastro.fold_data(self.t_flat,self.f_flat,self.EP._pl_tranmid,self.EP._pl_orbper),marker="o",lw=0,markersize=3,label="Best fit values")
        #    ax.legend(loc="lower right",fontsize=9)
        if use_BLS_values == True:
            self.planet._pl_tranmid = self.EP.bb._epoch + k2help.KEPLER_JD_OFFSET
            self.planet._pl_orbper  = self.EP.bb._best_period
        return self.planet

class EVERESTBLS(object):
    """
    EXAMPLE:
    E = EVERESTBLS(201498078,outmask=[],badmask=[],sigma_clipping=True)
    planet = E(period_range=(1.,50.))
    E.plot_folded()
    E.EP.plot_lc_min()

    win is not used if using GP flatten
    """
    def __init__(self,epicname,sigma_clipping=True,sigma_upper=4.,sigma=15.,plot_sigma_clip=True,win=49,badmask=None,outmask=None,flatten_method="GP",transitmask=None):
        self.epicname = epicname
        self.star = k2help.EverestGK(self.epicname)
        if transitmask is not None:
            self.star.transitmask = transitmask
        print("Transitmask:",self.star.transitmask)
        if badmask==[]:
            # Not really recommended
            self.star.badmask = badmask
        if badmask is not None and len(badmask)>0:
            print("Adding to badmask array")
            self.star.badmask = np.append(self.star.badmask,badmask)
            print(self.star.badmask)
        if outmask==[]:
            # Not really recommended
            self.star.outmask = outmask
        if flatten_method=="GP":
            print("Using *GP* to flatten (can be 'GP', 'Median', or default to SavGol)")
            print("there can still be leftover bad points, so sigma clipping is recommended")
            print("Also might be good to rerun with transitmask set")
            self.t, self.f = self.star.get_whitened_flux()
        elif flatten_method=="Median":
            print("Using *Median filtering* with Window {} to flatten (can be 'GP', 'Median', or default to SavGol)".format(win))
            self.t, self.f, mask = k2help.median_filter_and_sigma_clip(self.star.time,self.star.flux/np.nanmedian(self.star.flux),sigma_upper=sigma_upper,sigma=sigma,window=win)
            print("Appended",mask,"to star.badmask")
            self.star.badmask = np.append(self.star.badmask,mask)
        else:
        # UNCOMMENT TO GO BACK TO OLD SAVGOL
            print("Using old SavGold to flatten (not recommended)")
            self.t = self.star.apply_mask(self.star.time)
            flux = everest.math.SavGol(self.star.apply_mask(self.star.flux),win=win)
            self.flux_norm = flux/np.nanmedian(flux)        
            self.f = self.flux_norm
        if sigma_clipping:
            self.t_flat, self.f_flat = self.sigma_clip_fluxes(sigma_upper,sigma,plot_sigma_clip=plot_sigma_clip)
        else:
            self.t_flat, self.f_flat = self.t, self.f
        self.EP = EPICBLSMULTI(epicname=None,time=self.t_flat,flux=self.f_flat)
        
    def sigma_clip_fluxes(self,sigma_upper=3.,sigma=6.,plot_sigma_clip=False):
        self.data = astropy.stats.sigma_clipping.sigma_clip(self.f,sigma_upper=sigma_upper,sigma=sigma)
        m = self.data.mask
        t_flat = self.t[~m]
        f_flat = self.f[~m]
        print(len(self.t))
        print(len(self.t[~m]))
        if plot_sigma_clip:
            plt.plot(self.t,self.f)
            plt.plot(t_flat,f_flat)
        return t_flat,f_flat

    def __call__(self,plot_fit_panel=True,plot_folded_panel=True,use_BLS_values=False,period_range=(1.,50.),period_fit_prior=None,t0_fit_prior=None,dur=0.2):
        self.dur = dur
        self.EP(period_range=period_range,nf=10000,plot=plot_fit_panel,period_fit_prior=period_fit_prior,t0_fit_prior=t0_fit_prior,dur=dur)
        self.planet = self.EP.get_nexoplanet(epicname=self.epicname)
        if use_BLS_values == True:
            self.planet._pl_tranmid = self.EP.bb._epoch + k2help.KEPLER_JD_OFFSET
            self.planet._pl_orbper  = self.EP.bb._best_period
        self.plot_folded(dur=dur)
        return self.planet

    def plot_summary(self,savedir="bls_plots/",plot_aperture=True):
        self.savedir = savedir
        self.plot_aperture = plot_aperture
	utils.make_dir(self.savedir)

        #if mask_t0 is not None and mask_per is not None and mask_dur is not None:
        #    print("Masking planet")
        #    self.star.mask_planet(mask_t0,mask_per,mask_dur)
        #    self.savename = self.savedir + "bls_epic_"+str(epicname)+"_masked.pdf"
        #else:
        self.savename = self.savedir + "bls_epic_"+str(self.epicname)+".pdf"

        fig = plt.figure(figsize=(PW,PH))
        gs1 = GridSpec(3,3)
        gs1.update(top=0.98, bottom = 2/3.*1.03,hspace=0.07,left=0.07,right=0.96)
        gs = GridSpec(4,3)
        gs.update(top=2/3.*0.96,bottom=0.04,hspace=0.35,left=0.07,right=0.96)
	gs2 = GridSpec(4,3)
        gs.update(top=2/3.*0.96,bottom=0.04,hspace=0.35,left=0.07,right=0.96)

        ax_lcpos = subplot(gs1[0,:])
        ax_lctime = subplot(gs1[1,:],sharex=ax_lcpos)
        ax_lcwhite = subplot(gs1[2,:],sharex=ax_lcpos)
        ax_lcfold = subplot(gs[2,1:])
        ax_lnlike = subplot(gs[1,2])
        ax_lcoe   = subplot(gs[0,1])
        ax_lcoe2  = subplot(gs[0,2])
        ax_sde    = subplot(gs[3,1:])
        ax_transits = subplot(gs[1:,0])
        ax_info = subplot(gs[0,0])
        ax_ec = subplot(gs[1,1])

        
        ax_lcpos.plot(self.star.time,self.star.fraw,"k.",markersize=3)
        ax_lcpos.plot(self.star.time,self.star.fcor,"r.",markersize=3)
        ax_lctime.plot(self.EP.t,self.EP.f,"k.",markersize=3)
        #ax_lctime.plot(self.EP.t,self.EP.fcor,"k.",markersize=3)
        #ax_lctime.plot(self.EP.t,self.EP.flux,"r.",markersize=3)
        
        # Overview of transits with ticks
        self.EP.bb.plot_transits(ax=ax_lcwhite)

        # BLS power spectrum
        self.EP.bb.plot_power_spectrum(inverse=True,ax=ax_sde)

        # BLS fold
        self.EP.bb.plot_folded(ax=ax_lcfold)

        # Best fit
        self.EP.plot_lc_min(dur_fold=self.dur,ax=ax_lnlike)

        try:
            self.planet.plot_planet_param_table(ax=ax_info,linespacing=0.09)
        except:
            print("Error in plotting info")

        # Everest fold - for some reason this has to be at the end
        #self.plot_folded(dur=self.dur,ax=ax_lcoe)

        # Saving figure
        pp = PdfPages(self.savename)
        fig.savefig(pp,format='pdf')

        if self.plot_aperture:
            # Plotting aperture
            print("\n#################")
            print("Plotting Aperture")
            fig_ap,_ = self.star.plot_aperture(show=False)
            fig_ap.savefig(pp,format="pdf")
        # Best fit
        #fig.tight_layout()

        pp.close()
        print("Saved",self.savename)
    
    def plot_folded(self,dur=0.2):
        self.star.plot_folded(self.planet._pl_tranmid-k2help.KEPLER_JD_OFFSET,self.planet._pl_orbper,dur=dur)

    def get_cutout_phased_df(self,t0=None,P=None,dur=0.2,sigma=None,plot=True,use_median_filter=True):
        """
        A function to get both transit cutout data, and also the final folded transit.
        
        INPUT:
        - t0
        - P
        - dur is the total duration of the data (i.e. if 0.2, then goes from -0.1 to 0.1)
        - sigma 
        
        OUTPUT:
         df with columns:
         - x     the phased folded time
         - y     flux value
         - time  original time value before folding
         - index index of the original value before folding
         
        EXAMPLE:
        df = EP.get_cutout_phased_df(dur=0.4,sigma=3)
        _, mm = k2help.savgol_sigma_clip(df.y,win=21,sigma_upper=3)
        plt.plot(df.x, df.y,"r.")
        dfmm = df[~mm]
        t_fold_final = dfmm.x
        f_fold_final = dfmm.y
        plt.plot(t_fold_final,f_fold_final)

        dfmm_s = dfmm.sort_index()
        plt.plot(dfmm_s.x,dfmm_s.y,"r.")
        plt.plot(dfmm.x,dfmm.y,"k.")
        """
        if t0 is None and P is None:
            t0 = self.planet._pl_tranmid - k2help.KEPLER_JD_OFFSET
            P = self.planet._pl_orbper
            print("Using planet with t0=",t0,"and P=",P)

        if use_median_filter:
            print("Using median filtered data -- assumes you have run that from beginning!")
            t, f = self.t, self.f
        else:
            print("Using whitened flux")
            t, f = self.star.get_whitened_flux()

        m = np.isnan(f)
        t, f = t[~m],f[~m]
        
        ind = self.star.get_masked_indices_in_transit(t,t0,P,dur=dur)
        
        t_unfolded, f_unfolded = t[ind], f[ind]
        
        df = utils.fold_data(t_unfolded,f_unfolded,t0,P,dur=dur,sort=True)

        if sigma is not None:
            S=astropy.stats.sigma_clipping.sigma_clip(df.y.values)
            dfm = df[~S.mask]

        if plot:
            fig, ax = plt.subplots()
            ax.plot(df.x,df.y,"r.",label="folded data")
            if sigma is not None:
                ax.plot(dfm.x,dfm.y,"k.",label="sigma clipped")
            ax.legend(loc="upper right",fontsize=12)

        if sigma is not None:
            return dfm
        else:
            return df
