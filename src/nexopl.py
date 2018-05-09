from __future__ import print_function
import matplotlib.gridspec as gridspec
import os
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from astropy.time import Time
import batman
import mcmc_utils
import sys
#import mr_forecast
#from forecaster import mr_forecast
from astropy import constants as aconst
import utils 
import astropy
import k2help
from astropy import units as u
from scipy.stats import truncnorm 
import corner
import os
MASTER_DIR = os.path.dirname(os.path.abspath(__file__))
import exoplanet_functions
from matplotlib import rcParams

compactString = lambda string: string.replace(' ', '').replace('-', '').lower()
FINALDATE = Time("2020-01-01 00:00:00.0",format="iso").jd
#import het_config

class NExopl(object):
    """
    An exoplanet object. Autopopulates class attributes from df_exo.
    """
    pl_parameters = ["pl_name", "pl_hostname", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_trandur", "pl_trandurerr1", "pl_trandurerr2", "pl_tranmid", "pl_tranmiderr1", "pl_tranmiderr2", "ra_str", "dec_str", "pl_ratror", "pl_ratrorerr1", "pl_ratrorerr2", "pl_ratdor", "pl_ratdorerr1", "pl_ratdorerr2", "pl_orbincl", "pl_orbinclerr1", "pl_orbinclerr2", "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_masse", "pl_masseerr1", "pl_masseerr2", "pl_orblper", "pl_orblpererr1", "pl_orblpererr2", "st_mass", "st_masserr1", "st_masserr2", "st_rad", "st_raderr1", "st_raderr2", "st_teff", "st_tefferr1", "st_tefferr2", "st_metfe", "st_metfeerr1", "st_metfeerr2", "st_logg", "st_loggerr1", "st_loggerr2", "st_uj", "st_bj", "st_vj", "st_rc", "st_ic", "st_j", "st_h", "st_k"]
    def __init__(self,attributes,df_exo):
        for name in attributes:
            # Only one value
            #setattr(self, "_"+name, df_exo[name].values[0])
            setattr(self, "_"+ name, df_exo[name].values[0])
        setattr(self,"_pl_depth",(self._pl_ratror**2.)*1000.)
        #if (np.isfinite(self._pl_tranmid)) & (np.isfinite(self._pl_orbper)):
        #    setattr(self,"num_transits_to_FINALDATE",int((FINALDATE - self._pl_tranmid)/self._pl_orbper)+1)
        #else:
        #    setattr(self,"num_transits_to_FINALDATE",0)
        self.planet_table_parameters =  ['pl_tranmid',
                                         'pl_orbper',
                                         'pl_ratror',
                                         'pl_rade',
                                         'pl_radj',
                                         'pl_trandep',
                                         'pl_ratdor',
                                         'pl_orbsmax',
                                         'pl_orbincl',
                                         'pl_imppar',
                                         'pl_orbeccen',
                                         'pl_orblper',
                                         'pl_eqt',
                                         'pl_trandur']
        self.latex_labels_arr = np.array([['$T_{0}$ $(\\mathrm{BJD_{TDB}})$', 'Transit Midpoint'],
                                          ['$P$ (days)', 'Orbital period'],
                                          ['$R_p/R_*$', 'Radius ratio'],
                                          ['$R_p (R_\\oplus)$', 'Planet radius'],
                                          ['$R_p (R_J)$', 'Planet radius'],
                                          ['$\\delta$', 'Transit depth'],
                                          ['$a/R_*$', 'Normalized orbital radius'],
                                          ['$a$ (AU)', 'Semi-major axis'],
                                          ['$i$ $(^{\\circ})$', 'Transit inclination'],
                                          ['$b$', 'Impact parameter'],
                                          ['$e$', 'Eccentricity'],
                                          ['$\\omega$ $(^{\\circ})$', 'Argument of periastron'],
                                          ['$T_{\\mathrm{eq}}$(K)', 'Equilibrium temperature'],
                                          ['$T_{14}$ (days)', 'Transit duration']])
        self.planet_stellar_parameters = ['st_teff',
                                          'st_metfe',
                                          'st_logg']
        self.planet_stellar_mag        = ['st_uj',
                                          'st_bj',
                                          'st_vj',
                                          'st_rc',
                                          'st_ic',
                                          'st_j',
                                          'st_h',
                                          'st_k']
	if self._st_rad is None:
		print("WARNING, NO STAR RADIUS, ASSUMING 1R_sun")
		self._st_rad = 1.0
        self._pl_a = (self._pl_orbsmax*aconst.au)/(self._st_rad*aconst.R_sun)
        self._pl_a = self._pl_a.value

        self.planet_table_parameters_latex = pd.DataFrame(zip(self.planet_table_parameters,self.latex_labels_arr))

    def append_planet_to_database(self,prefix="Candidate:",suffix="b"):
        """
        """
        df_cand = pd.read_csv(MASTER_DIR+"/data/nexscidatabase/additional_candidates.csv")
        values = [getattr(self,"_"+attr) for attr in self.pl_parameters]
        df = pd.DataFrame(pd.Series(values,index=self.pl_parameters)).T
        planet_save_name = prefix+df["pl_name"].values[0]+suffix
        df.set_value(0,"pl_name",planet_save_name)
        df_final = pd.concat([df_cand,df]).reset_index(drop=True)
        df_final.to_csv(MASTER_DIR+"/data/nexscidatabase/additional_candidates.csv",index=False)
        print("Appended ",planet_save_name,"to",MASTER_DIR+"/data/nexscidatabase/additional_candidates.csv")

    @property
    def num_transits_to_FINALDATE(self):
        if (np.isfinite(self._pl_tranmid)) & (np.isfinite(self._pl_orbper)):
            return int((FINALDATE - self._pl_tranmid)/self._pl_orbper)+1
        else:
            return 0
        #return int((FINALDATE - self._pl_tranmid)/self._pl_orbper)+1

    def next_transits_fast(self,startdate=None,stopdate=None,location="Apache Point Observatory"):
        jd_start = Time(startdate,format="iso").jd
        jd_stop  = Time(stopdate, format="iso").jd

        num2start = int((jd_start - self._pl_tranmid)/self._pl_orbper)-1
        num2stop  = int((jd_start - self._pl_tranmid)/self._pl_orbper)+1

        n_transits = np.arange(num2start,num2stop+1)
        if len(n_transits==0):
            return np.nan
        else:
            tr_midp = np.zeros(self.num_transits_to_FINALDATE)
            for i in n_transits:
                tr_midp[i] = self._pl_tranmid + i*self._pl_orbper # in jd
            tr_midp_iso = Time(tr_midp,format="jd").iso
            names = [self._pl_name for i in range(len(tr_midp))]
            df = pd.DataFrame(zip(names,tr_midp_iso,tr_midp),columns=["names","iso","jd"])
            df = df.set_index("iso",drop=False)
            #df = df[startdate:stopdate]
            return df


        #if np.isnan(self._pl_tranmid):
        #    print("No transits")
        #    return np.nan
        #self.location = location
        ## MIDP
        #tr_midp = np.zeros(self.num_transits_to_FINALDATE)
        #for i in range(self.num_transits_to_FINALDATE):
        #    tr_midp[i] = self._pl_tranmid + i*self._pl_orbper # in jd

        print("Name",self._pl_name)

    def get_next_transits(self,startdate=None,stopdate=None,location="Apache Point Observatory",verbose=True,visible=True):
        df = self.next_transits(self.num_transits_to_FINALDATE,startdate=startdate,stopdate=stopdate,location=location,verbose=verbose)
        if visible:
            df = df[df.tr_midp_is_target_observable==True]
        return df


    def next_transits(self,num,startdate=None,stopdate=None,location="Apache Point Observatory",verbose=True):
        """
        Calc the next *num* transits.
        """
        self.location = location
        # MIDP
        tr_midp = np.zeros(num)
        tr_midp_err = np.zeros(num)
        tr_midp_alt = np.zeros(num)
        tr_midp_airm = np.zeros(num)
        tr_midp_sun_alt = np.zeros(num)
        
        # START
        tr_start = np.zeros(num)
        tr_start_noerr = np.zeros(num)
        tr_start_alt = np.zeros(num)
        tr_start_airm = np.zeros(num)
        
        # END
        tr_stop = np.zeros(num)
        tr_stop_noerr = np.zeros(num)
        tr_stop_alt = np.zeros(num)
        tr_stop_airm = np.zeros(num)
        
        # _pl_orbper
        # _pl_orbpererr1
        # _pl_tranmid
        # _pl_tranmiderr1
        # _pl_trandur
        # _pl_trandurerr1

        if np.isnan(self._pl_tranmiderr1):
            print("self._pl_tranmiderr1 is NaN, setting as 0")
            self._pl_tranmiderr1 = 0.

        if np.isnan(self._pl_trandurerr1):
            print("self._pl_trandurerr1 is NaN, setting as 0")
            self._pl_trandurerr1 = 0.

        if np.isnan(self._pl_orbpererr1):
            print("self._pl_orbpererr1 is NaN, setting as 0")
            self._pl_orbpererr1 = 0.

        if np.isnan(self._pl_trandur):
            print("self._pl_trandur is NaN, setting as 0")
            self._pl_trandur = 0.

        if np.isnan(self._pl_trandurerr1):
            print("self._pl_trandurerr1 is NaN, setting as 0")
            self._pl_trandurerr1 = 0.

        if np.isnan(self._pl_trandurerr2):
            print("self._pl_trandurerr2 is NaN, setting as 0")
            self._pl_trandurerr2 = 0.

        if self._pl_trandur==0. or self._pl_trandur is np.nan:
            self._pl_trandur = self.calc_transit_duration()

        if verbose==True:
            print("Calculating transit midpoints, total: ",num)
        # Calculating transit timing
        for i in range(num):
            tr_midp[i] = self._pl_tranmid + i*self._pl_orbper # in jd
            tr_midp_err[i] = self._pl_tranmiderr1 + i*self._pl_orbpererr1
            #tr_start_noerr[i] = tr_midp[i] - (self._t_14/(2*24.*60.)
            #tr_start[i] = tr_midp[i] - (self._t_14/(2*24.*60.) + tr_midp_err[i] + (self._t_14_error_1/(24.*60.)))                                  
            #tr_stop_noerr[i] = tr_midp[i] + (self._t_14/(2*24.*60.)
            #tr_stop[i]  = tr_midp[i] + (self._t_14/(2*24.*60.) + tr_midp_err[i] + (self._t_14_error_1/(24.*60.)))
            #tr_start[i] = tr_midp[i] - (self._t_14/(2*24.*60.) + tr_midp_err[i] + (self._t_14_error_1/(24.*60.)))
            #tr_stop[i]  = tr_midp[i] + (self._t_14/(2*24.*60.) + tr_midp_err[i] + (self._t_14_error_1/(24.*60.)))
            #tr_midp_alt[i], tr_midp_airm[i], tr_midp_sun_alt[i] = gkastro.calc_altitude_airmass(self._RA,self._DEC,tr_midp[i],timeformat="jd",location=location,verbose=False,sun=True)
            #tr_start_alt[i], tr_start_airm[i] = gkastro.calc_altitude_airmass(self._RA,self._DEC,tr_start[i],timeformat="jd",location=location,verbose=False)
            #tr_stop_alt[i],  tr_stop_airm[i]  = gkastro.calc_altitude_airmass(self._RA,self._DEC,tr_stop[i],timeformat="jd",location=location,verbose=False)
            #print(tr_midp[i],tr_start[i],tr_stop[i])
        
        # Calculating start and stop times
        #print(tr_midp)
        tr_start_noerr = tr_midp - self._pl_trandur/2.#24.*60.)
        tr_start = tr_start_noerr - tr_midp_err - self._pl_trandurerr1/2.#*24.*60.)
        #print(tr_start)
        tr_stop_noerr = tr_midp + self._pl_trandur/2.#*24.*60.)
        tr_stop = tr_stop_noerr + tr_midp_err + self._pl_trandurerr1/2.#.*24.*60.)                           

        if verbose==True:
            print("Calculating transit midpoints, total: ",num,"Done!")
            print("Calculating, altitudes")
        tr_midp_alt, tr_midp_airm, tr_midp_sun_alt = gkastro.calc_altitude_airmass(self._ra_str,self._dec_str,tr_midp,timeformat="jd",location=location,verbose=False,sun=True)
        tr_start_alt, tr_start_airm = gkastro.calc_altitude_airmass(self._ra_str,self._dec_str,tr_start,timeformat="jd",location=location,verbose=False)
        #tr_start_noerr_alt, tr_start_noerr_airm = gkastro.calc_altitude_airmass(self._ra_str,self._dec_str,tr_start_noerr,timeformat="jd",location=location,verbose=False)
        tr_stop_alt,  tr_stop_airm  = gkastro.calc_altitude_airmass(self._ra_str,self._dec_str,tr_stop,timeformat="jd",location=location,verbose=False)
        #tr_stop_noerr_alt,  tr_stop_noerr_airm  = gkastro.calc_altitude_airmass(self._ra_str,self._dec_str,tr_stop_noerr,timeformat="jd",location=location,verbose=False)
        if verbose==True:
            print("Calculating, altitudes","Done!")
        tr_midp_is_sun_up = tr_midp_sun_alt > 0.
        tr_midp_is_target_up = tr_midp_alt > 0.
        tr_midp_is_target_observable = (tr_midp_is_sun_up == False ) & (tr_midp_is_target_up == True )
        
        tr_midp_err_min = tr_midp_err*24.*60

        tr_midp = Time(tr_midp,format="jd")
        tr_midp_err = Time(tr_midp_err,format="jd")
        tr_start = Time(tr_start,format="jd")
        tr_stop  = Time(tr_stop,format="jd")
        
        # midpoint errors in minutes
        #tr_midp_err_min = [i*24.*60 for i in tr_midp_err.jd]
        #print(tr_midp.jd)
        #tr_midp_alt = [self.calc_altitude_airmass(i) for i in tr_midp.jd]
        #tr_midp_airm = [airmass(i) for i in tr_midp_alt]

        names = [self._pl_name for i in range(len(tr_stop))]
        
        # Preparing dataframe
        data = zip(range(num),
                   names, 
                   tr_midp.jd,
                   tr_midp.iso,
                   tr_midp_err.jd,
                   tr_midp_err_min,
                   tr_midp_alt,
                   tr_midp_airm,
                   tr_midp_sun_alt,
                   tr_midp_is_sun_up,
                   tr_midp_is_target_up,
                   tr_midp_is_target_observable,
                   tr_start.jd,
                   tr_start.iso,
                   tr_start_noerr,
                   tr_start_alt,
                   tr_start_airm,
                   #tr_start_noerr_alt,
                   tr_stop.jd,
                   tr_stop.iso,
                   tr_stop_noerr,
                   tr_stop_alt,
                   tr_stop_airm)
                   #tr_stop_noerr_alt,)
        columns = columns=["num",
                           "_pl_name",
                           "tr_midp_jd",
                           "tr_midp_cal",
                           "tr_midp_err_jd",
                           "tr_midp_err_min",
                           "tr_midp_alt",
                           "tr_midp_airm",
                           "tr_midp_sun_alt",
                           "tr_midp_is_sun_up",
                           "tr_midp_is_target_up",
                           "tr_midp_is_target_observable",
                           "tr_start_jd",
                           "tr_start_cal",
                           "tr_start_noerr_jd",
                           "tr_start_alt",
                           "tr_start_airm",
                           #"tr_start_noerr_alt",
                           "tr_stop_jd",
                           "tr_stop_cal",
                           "tr_stop_noerr_jd",
                           "tr_stop_alt",
                           "tr_stop_airm"]
                           #"tr_stop_noerr_alt"]
        self.df_tr = pd.DataFrame(data=data,columns=columns)
        self.df_tr = self.df_tr.set_index("tr_midp_cal",drop=False)
        self.df_tr = self.df_tr[startdate:stopdate]
        return self.df_tr

    def calc_transit_duration(self):
        self._pl_trandur = exoplanet_functions.calc_transit_duration(self._pl_orbper,
                                                  self._pl_ratror,
                                                  self._pl_ratdor,
                                                  np.deg2rad(self._pl_orbincl))
        return self._pl_trandur
    
    def plot_expected_light_curve(self,u=[0.3,0.3],limb_dark_model="quadratic",cadence=60.,t0=None,unit=False,error=0.,bin_size_min=30.,start=-0.15,stop=0.15,offset=0.05,ax=None,plot=True,verbose=True):
        """
        # _pl_orbper
        # _pl_orbpererr1
        # _pl_tranmid
        # _pl_tranmiderr1
        # _pl_trandur
        # _pl_trandurerr1

        cadence is in seconds
        error is in ppm
        bin_size is in minutes
        start in days
        stop in days
        """
        params     = batman.TransitParams()
        if t0 == None:
            params.t0  = self._pl_tranmid
        else:
            params.t0 = t0
        params.per = self._pl_orbper
        params.rp  = self._pl_ratror #(1.9*aconst.R_earth)/aconst.R_sun # planet.pl_radj.values*aconst.R_jup/aconst.R_sun                     #planet radius (in units of stellar radii)
        aRs = (self._pl_orbsmax*aconst.au)/(self._st_rad*aconst.R_sun)
        if np.isnan(aRs):
            params.a = self._pl_ratdor # planet.pl_orbsmax.values        #semi-major axis (in units of stellar radii)
        else:
            params.a = aRs
        if np.isnan(self._pl_orbincl):
            print("WARNING, NO INCLINATION, ASSUMING i = 90")
            params.inc = 90.
        else:
            params.inc = self._pl_orbincl # planet.pl_orbincl.values       #orbital inclination (in degrees)
        if np.isnan(self._pl_orbeccen):
            print("WARNING, ECCENTRICITY, ASSUMING e = 0")
            params.ecc = 0. #planet.pl_orbeccen.values       #eccentricity
        else:
            params.ecc = self._pl_orbeccen
        #if np.isnan(self._pl_orbeccen):
        params.w = 0.
        #else:
        #params.w   = self._pl_orblper                     #longitude of periastron (in degrees)
        params.u   = u #limb darkening coefficients # 0.43489202      0.30191999
        params.limb_dark = limb_dark_model        #limb darkening model
        
        if verbose:
            print("T0",params.t0)
            print("Period (days)",params.per)
            print("Rp/Rs)",params.rp)
            print("a (R_star)",params.a)
            print("i",params.inc)
            print("e",params.ecc)
            print("w",params.w)
            print("u",params.u)
            print(params.limb_dark)
        t_start = start+ params.t0
        t_stop = stop + params.t0#0.7
        
        cadence = cadence #s

        self.time_l = np.arange(t_start,t_stop,cadence/(24*3600.))
        m = batman.TransitModel(params,self.time_l)
        self.flux_l = m.light_curve(params)

        title_str = "Expected transit: " + self._pl_name + " cadence: " + gk.num2str(cadence,0) +"s \n Assuming $\sigma_{\mathrm{unbinned}}$="+gk.num2str(error,0)+"ppm"

        if error>0.:
            flux_err = np.random.normal(0,error*1e-6,len(self.flux_l))
            self.flux_l_err = self.flux_l + flux_err
        else:
            self.flux_l_err = self.flux_l 
        
        if plot:
            if ax == None:
                self.fig, self.ax = plt.subplots()
            else:
                self.ax = ax

            if unit=="hours":
                self.ax.plot(self.time_l, self.flux_l,label="Model ",alpha=0.7,lw=1,color="firebrick")
                self.ax.plot(self.time_l*24., self.flux_l_err,label="Model",alpha=0.7,marker="o",ms=4,lw=0.5)
                self.ax.set_xlabel("Time (hours)")
            else:
                self.ax.plot(self.time_l, self.flux_l,label="Model ",alpha=1,lw=1,color="firebrick",zorder=5)
                self.ax.plot(self.time_l, self.flux_l_err,label="Unbinned "+str(cadence)+"s",alpha=0.7,marker="o",ms=4,lw=0.5)
                self.ax.set_xlabel("Time (d)")
            if bin_size_min!=None:
                nbin = int(bin_size_min*60./cadence)
                print("Binning "+str(nbin))
                self.binned = gkastro.bin_data(self.time_l,self.flux_l_err,nbin)
                self.ax.plot(self.binned.x, self.binned.y-offset,alpha=0.7,marker="o",ms=4,lw=0.5,label="Binned "+str(bin_size_min)+"min")
                self.ax.plot(self.time_l, self.flux_l-offset,label="Model ",alpha=1,lw=1,color="firebrick",zorder=5)

            self.ax.legend(loc="lower left",fontsize=10)
            self.ax.minorticks_on()
            self.ax.margins(x=0.05,y=0.2)
            self.ax.set_title(title_str,y=1.03)
            self.ax.set_ylabel("Relative flux")
            for label in self.ax.get_xticklabels():
                label.set_fontsize(12)
            for label in self.ax.get_yticklabels():
                label.set_fontsize(12)

    def get_finder_image(self,survey=["DSS2 Red"],stretch="linear",cmap="gray",ax=None,field_of_view="ARCTIC",overlay_target=True):
        """
        INPUT:
        radius needs astropy unit

        NOTES:
        
        Get list of surveys by:
        SkyView.list_surveys()
        u'Optical:DSS': [u'DSS',
                  u'DSS1 Blue',
                  u'DSS1 Red',
                  u'DSS2 Red',
                  u'DSS2 Blue',
                  u'DSS2 IR'],
         u'Optical:SDSS': [u'SDSSg',
                           u'SDSSi',
                           u'SDSSr',
                           u'SDSSu',
                           u'SDSSz',
                           u'SDSSdr7g',
                           u'SDSSdr7i',
                           u'SDSSdr7r',
                           u'SDSSdr7u',
                           u'SDSSdr7z'],
        """
        self.finder_image = finderimage.FinderImage(self._ra_str,self._dec_str)

        if field_of_view=="ARCTIC":
            FOV    = fov.FOVARCTIC
            radius = 14.*u.arcmin
        elif field_of_view=="CDK24":
            FOV = fov.FOVCDK24
            radius = 40.*u.arcmin
        elif field_of_view=="NOT":
            FOV = fov.FOVNOT
            radius = 20.*u.arcmin
        elif field_of_view=="Kryoneri":
            FOV = fov.FOVKryoneri
            radius = 12.7*u.arcmin
        elif field_of_view=="HDI":
            FOV = fov.FOVHDI
            radius = 25.*u.arcmin
        elif field_of_view=="HCT":
            FOV = fov.FOVHCT
            radius = 10.*u.arcmin
        elif field_of_view=="TMMT":
            FOV = fov.FOVTMMT
            radius = 40.*u.arcmin
        else:
            print("FOV has to be: 'ARCTIC', 'CDK24', 'TMMT', 'HCT', 'Kryoneri'")
            print("Not overlaying any FOV")
            radius = 40.*u.arcmin

        if ax == None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        self.finder_image.get_image(radius=radius)
        self.finder_image.plot(ax=self.ax, cmap=cmap, stretch=stretch, overlay_target=overlay_target)
        self.finder_image.overlay_FOV(FOV,ax=self.ax)
        title_str = "Finder Chart: " + self._pl_name + "\n FOV: " + field_of_view
        self.ax.set_title(title_str,fontsize=12)

    
    def airmassplot(self,tr_midp_cal="",transit_number=None,save=False,startdate=None,stopdate=None,describe=True,ax=None):
        """
        A function to create an airmassplot of a transit midpoint.
        
        EXAMPLE:
            ee = ExoEphem()
            k218b   = ee.get_exoplanet("K2-18b")
            df = k218b.next_transits(40,startdate="2016-10-01 00:00:00")
            df_observable = k218b.df_tr[k218b.df_tr.tr_midp_is_target_observable==True]
            df_observable
            k218b.airmassplot(tr_midp_cal="2017-10-28 10:36:05.990")
            k218b.airmassplot(transit_number=23)
        """
        self.ap = airmassplot.AirmassPlot(self._pl_name,self._ra_str,self._dec_str,location=self.location)
        
        if tr_midp_cal!="":
            dff = self.df_tr[self.df_tr.tr_midp_cal==tr_midp_cal]
        elif transit_number!=None:
            dff = self.df_tr[self.df_tr.num==transit_number]
        
        setattr(self.ap,"_t_midp",Time(dff.ix[0].tr_midp_cal))
        setattr(self.ap,"_t_start",Time(dff.ix[0].tr_start_cal))
        setattr(self.ap,"_t_start_noerr",Time(dff.ix[0].tr_start_noerr_jd,format="jd"))
        setattr(self.ap,"_t_stop",Time(dff.ix[0].tr_stop_cal))
        setattr(self.ap,"_t_stop_noerr",Time(dff.ix[0].tr_stop_noerr_jd,format="jd"))
        setattr(self.ap,"_t_midp_err_min",dff.ix[0].tr_midp_err_min)

        # Plot airmass plot
        self.ap.plot(t_midp=self.ap._t_midp,
                     t_start=self.ap._t_start,
                     t_start_noerr=self.ap._t_start_noerr,
                     t_stop=self.ap._t_stop,
                     t_stop_noerr=self.ap._t_stop_noerr,
                     t_midp_err_min=self.ap._t_midp_err_min,
                     ylim=(0,90),
                     ax=ax)
        #self.ap.plot(t_midp=Time(dff.ix[0].tr_midp_cal),
        #        t_start=Time(dff.ix[0].tr_start_cal),
        #        t_start_noerr=Time(dff.ix[0].tr_start_noerr_jd,format="jd"),
        #        t_stop=Time(dff.ix[0].tr_stop_cal),
        #        t_stop_noerr=Time(dff.ix[0].tr_stop_noerr_jd,format="jd"),
        #        t_midp_err_min=dff.ix[0].tr_midp_err_min,
        #ylim=(0,90))
        if startdate!=None:
            self.ap.ax.plot(startdate,5*np.ones(len(startdate)),marker="d",lw=0)
        if stopdate!=None:
            self.ap.ax.plot(stopdate,5*np.ones(len(stopdate)),marker="d",lw=0)
        if describe==True:
            printstr =  "Vmag: "+str(self._st_vj) + "\n" + \
                    "rmag: "+str(self._st_rc) + "\n" + \
                    "imag: "+str(self._st_ic) + "\n" + \
                    "Jmag: "+str(self._st_j) + "\n" + \
                    "P (d): "+str(self._pl_orbper) + "\n" + \
                    "R_st: "+str(self._st_rad) + "\n" + \
                    "RpRs^2: "+str(self._pl_ratror**2.*1000.) + "mmag\n"
            self.ap.ax.text(self.ap.obstimes[700].datetime,60.,printstr)

        if save==True:
            self.ap.fig.tight_layout()
            self.ap.fig.subplots_adjust(right=0.73,left=0.1)
            self.ap.fig.savefig(self._pl_name+"_"+str(self.ap._t_midp)+".png")
            
    def describe(self):
        """
        Describe the planet, and the main parameters

        print("Planet name: \t %s " % self._pl_name)
        print("Period (d):  \t %f +%f %f" % (self._pl_orbper,self._pl_orbpererr1,self._pl_orbpererr2))
        print("T0: \t \t %f +%f %f" % (self._pl_tranmid,self._pl_tranmiderr1,self._pl_tranmiderr2))
        print("Duration (d): \t %f +%f %f" % (self._pl_trandur,self._pl_trandurerr1,self._pl_trandurerr2))
        print("RA: \t \t %s" % self._ra_str)
        print("DEC: \t \t %s" % self._dec_str)
        print("")
        print("=== Host star Photometry ===")
        print("Spectral Type: \t  %s " % self._st_spstr)
        print("Optmag: \t  %s (Band: %s)" % (self._st_optmag,self._st_optband))
        print("U: \t \t  %s " % self._st_uj)
        print("B: \t \t  %s " % self._st_bj)
        print("V: \t \t  %s " % self._st_vj)
        print("R: \t \t  %s " % self._st_rc)
        print("I: \t \t  %s " % self._st_ic)
        print("J: \t \t  %s " % self._st_j)
        print("H: \t \t  %s " % self._st_h)
        print("K: \t \t  %s " % self._st_k)
        """
        if np.isnan(self._pl_rade):
            print("WARNING, NO planet radius, calculating assuming:")
            print("Rp/R*={}".format(self._pl_ratror))
            print("R*={}".format(self._st_rad))
            self._pl_rade = self._pl_ratror*self._st_rad*aconst.R_sun.value/aconst.R_earth.value
        if np.isnan(self._pl_radeerr1):
            print("WARNING, NO PLANET RADERR1, setting as 10% of radius")
            self._pl_radeerr1 = 0.1*self._pl_rade
        print("Planet name: \t %s " % self._pl_name)
        print("TRANSITFLAG: \t %s " % self._pl_tranflag)
        print("Period (d):  \t %f +%f %f" % (self._pl_orbper,self._pl_orbpererr1,self._pl_orbpererr2))
        print("T0: \t \t %f +%f %f" % (self._pl_tranmid,self._pl_tranmiderr1,self._pl_tranmiderr2))
        print("Duration (d): \t %f +%f %f" % (self._pl_trandur,self._pl_trandurerr1,self._pl_trandurerr2))
        print("Rp/Rs:\t \t %s " % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_pl_ratror")))
        print("Depth (mmag):\t "+ gk.num2str(self._pl_depth))
        print("Inc: \t \t %s" % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_pl_orbincl")))
        print("e: \t \t %s" % self._pl_orbeccen)
        print("a (AU): \t %s " % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_pl_orbsmax")))# self._pl_orbsmax)
        print("a/Rs: \t \t " + gk.num2str(self._pl_a))
        print("a/Rs (ratdor): \t " + gk.num2str(self._pl_ratdor))
        print("")
        print("=== Host star Photometry ===")
        print("RA: \t \t %s" % self._ra_str)
        print("DEC: \t \t %s" % self._dec_str)
        print("Spectral Type: \t  %s " % self._st_spstr)
        print("Rs: \t \t %s" % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_st_rad")))
        print("Ms: \t \t %s" % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_st_mass")))
        print("Teff: \t \t %s" % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_st_teff")))
        print("[Fe/H]: \t %s" % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_st_metfe")))
        print("log(g): \t %s" % mcFunc.latex_mean_low_up(*self.get_attributeAndError("_st_logg")))
        print("Optmag: \t  %s (Band: %s)" % (self._st_optmag,self._st_optband))
        print("U: \t \t  %s " % self._st_uj)
        print("B: \t \t  %s " % self._st_bj)
        print("V: \t \t  %s " % self._st_vj)
        print("R: \t \t  %s " % self._st_rc)
        print("I: \t \t  %s " % self._st_ic)
        print("J: \t \t  %s " % self._st_j)
        print("H: \t \t  %s " % self._st_h)
        print("K: \t \t  %s " % self._st_k)

        #print("Radius (R_star)",params.rp)
        #print("a (R_star)",params.a)
        #print("i",params.inc)
        #print("e",params.ecc)
        #print("w",params.w)
        #print("u",params.u)
        #print(params.limb_dark)

    #def predict_mass_from_radius(self,classify="no"):
    #    """
    #    Use Forecaster from Chen & Kipping to estimate mass from radius

    #    INPUT:
    #    classify = 'Yes' or 'No'

    #    OUTPUT:

    #    NOTES:
    #    can be accessed through the following attributes:
    #        setattr(self,"_pl_pred_masse",Mmedian)
    #        setattr(self,"_pl_pred_masseerr1",Mplus)
    #        setattr(self,"_pl_pred_masseerr2",Mminus)
    #    """
    #    
    #    Mmedian, Mplus, Mminus = mr_forecast.Rstat2M(mean=self._pl_rade, std=self._pl_radeerr1, unit='Earth', sample_size=1000, grid_size=1e3, classify=classify)
    #    setattr(self,"_pl_pred_masse",Mmedian)
    #    setattr(self,"_pl_pred_masseerr1",Mplus)
    #    setattr(self,"_pl_pred_masseerr2",Mminus)

    #    return Mmedian, Mplus, Mminus

    def predict_mass_from_radius_posterior(self,sample_size=10000,classify="No",plot=True):
        """
        Use Forecaster from Chen & Kipping to estimate mass from radius

        OUTPUT:
            Returns the radius and mass posteriors
        """
        rlower = 1e-1; rupper = 1e2;
        mean = self._pl_rade; std = self._pl_radeerr1
        rad_norm = truncnorm.rvs( (rlower-mean)/std, (rupper-mean)/std, loc=mean, scale=std, size=sample_size)
        mass_norm = mr_forecast.Rpost2M(rad_norm, unit='Earth', grid_size=1e3, classify=classify)

        if plot:
            self.fig = corner.corner(np.vstack([rad_norm,mass_norm]).T,
                           labels=[r"$R_\oplus$", r"$M_\oplus$"],
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12},lw=0.5,
                           hist_kwargs={"lw":0.5})

        return rad_norm, mass_norm


    def get_predicted_rv_semiamplitude_posteriors(self,sample_size=10000,classify="Yes",plot=True,xcord=(0.5,-0.2),ycord=(-0.2,0.5),showsuptitle=True,
                                                  ticklabelsize=11,labelsize=12):
        """
        INPUT:
        return_post: True
           if False, then return the a dataframe with the mean and plus minus values
        """
        r_post, m_post = self.predict_mass_from_radius_posterior(sample_size=sample_size,classify=classify,plot=False)

        if np.isnan(self._pl_orbeccen):
            print("WARNING: e not defined. Assuming e=0")
            e = 0.
        else:
            e = self._pl_orbeccen

        if np.isnan(self._pl_orbincl):
            print("WARNING: i not defined. Assuming i=90")
            i = 90.
        else:
            i = self._pl_orbincl

        rv_post = gkastro.rv_semiamplitude(m_1=self._st_mass,
                                          m_2=m_post,
                                          P=self._pl_orbper,
                                          i=i,
                                          e=e)
        
        if plot:
            rcParams["lines.linewidth"] = 1.0
            rcParams["axes.labelpad"] = 20.0
            rcParams["xtick.labelsize"] = ticklabelsize
            rcParams["ytick.labelsize"] = ticklabelsize
            self.fig = corner.corner(np.vstack([r_post,m_post,rv_post]).T,
                           labels=[r"$R_\oplus$", r"$M_\oplus$",r"RV (m/s)"],
                           quantiles=[0.16, 0.5, 0.84],
                           hist_kwargs={"lw":1.5},
                           show_titles=True, title_kwargs={"fontsize": labelsize},lw=1.,
                           label_kwargs={"fontsize":labelsize},
                           xlabcord=(xcord),
                           ylabcord=(ycord));
            if showsuptitle:
                self.fig.suptitle("Expected Mass and RV amplitude: "+str(self._pl_name),fontsize=20,y=1.03)

        return r_post, m_post, rv_post


    def get_attributeAndError(self,attributename):
        """
        mcFunc.latex_mean_low_up(*tres.get_attributeAndError("_pl_trandur"))
        """
        return getattr(self,attributename),getattr(self,attributename+"err1"),getattr(self,attributename+"err1")

    def get_attributeAndErrorAll(self):
        [print(i,self.get_attributeAndError("_"+i)) for i in self.planet_table_parameters + self.planet_stellar_parameters]
        [print(i,getattr(self,"_"+i)) for i in self.planet_stellar_mag]

    def get_latex_table(self):
        """
        Print a nice latex table with parameter values
        """
        mean = np.zeros(len(self.planet_table_parameters))
        low  = np.zeros(len(self.planet_table_parameters))
        up   = np.zeros(len(self.planet_table_parameters))
        latex_mean_low_up = []
        for i,param in enumerate(self.planet_table_parameters):
            mean[i], low[i], up[i] =  self.get_attributeAndError("_"+param)
            latex_mean_low_up.append(mcFunc.latex_mean_low_up(mean[i],low[i],up[i]))

        self.df_planet_param = pd.DataFrame(zip(self.latex_labels_arr[:,0],self.latex_labels_arr[:,1],mean,low,up,latex_mean_low_up),columns=["Labels","Description","medvals","minus","plus","values"])
        return self.df_planet_param

    def get_everest_star(self,epicid,plot=True):
        """
            star.plot_folded(self._pl_tranmid-2454833,self._pl_orbper)
        """
        import k2help
        star = k2help.EverestGK(epicid)

        if plot==True:
            star.plot_folded(self._pl_tranmid-2454833,self._pl_orbper)
        return star
            
    def plot_overview(self,tr_midp_cal,cadence=120.,error=4000.,bin_size_min=10.,offset=0.02,save=False,savesuffix="",savedir="plots/",saveext=".png"):
        """
        A function to plot an overview plot

        TODO:
        - Need to tweak
        """
        if self.location=="Black Moshannon Observatory":
            field_of_view = "CDK24"
        elif self.location=="Las Campanas Observatory":
            field_of_view = "TMMT"
        elif self.location=="Apache Point Observatory":
            field_of_view = "ARCTIC"
        elif self.location=="lapalma":
            field_of_view = "NOT"
        elif self.location=="Kitt Peak National Observatory":
            field_of_view = "HDI"
        elif self.location=="IAO":
            field_of_view = "HCT"
        elif self.location=="Kryoneri":
            field_of_view = "Kryoneri"

        # Get planet parameters
        _ = self.get_latex_table() # stored in self.df_planet_param
            
        ### Define subpanels
        self.fig = plt.figure(1)
        gs1 = gridspec.GridSpec(2,2)
        gs1.update(left=0.0, right=0.40)
        ax1_fov = plt.subplot(gs1[0, :])
        ax2_pl = plt.subplot(gs1[1, 0])
        ax2_st = plt.subplot(gs1[1, 1])
        gs2 = gridspec.GridSpec(20, 1)
        gs2.update(left=0.46, right=0.98, wspace=0.3)
        ax4_air = plt.subplot(gs2[0:9, :])
        ax5_tra = plt.subplot(gs2[12:, :])

        ########################################
        ##### TEXT - PLANET
        ax2_pl.xaxis.set_visible(False)
        ax2_pl.yaxis.set_visible(False)
        ltable, rtable = [], []
        ltable += ["Planet:"]
        rtable += [self._pl_name]
        ltable += list(self.df_planet_param["Labels"].values)
        rtable += list(self.df_planet_param["values"].values)
        yt = 0.875
        for l,r in zip(ltable, rtable):
            ax2_pl.annotate(l, xy=(0.25, yt), xycoords="axes fraction", ha='right', fontsize=8)
            ax2_pl.annotate(r, xy=(0.35, yt), xycoords="axes fraction", fontsize=8)
            yt -= 0.08
        ax2_pl.axis("off")

        ########################################
        ##### TEXT - STAR
        ltable, rtable = [], []
        params = ["_st_spstr","_st_uj","_st_bj","_st_vj","_st_rc","_st_ic","_st_j","_st_h","_st_k"]
        params_with_errors = ["_st_rad","_st_mass","_st_teff","_st_metfe","_st_logg"]
        ltable += ["Star:"]
        rtable += [""]
        ltable += ["SpType:","U","B","V","R","I","J","H","K",r"$R_*$",r"$M_*$",r"$T_{eff}$","[Fe/H]",r"$\log(g)$"]
        rtable += [getattr(self,param) for param in params]
        rtable += [mcFunc.latex_mean_low_up(*self.get_attributeAndError(param)) for param in params_with_errors]
        yt = 0.875
        for l,r in zip(ltable, rtable):
            ax2_st.annotate(l, xy=(0.25, yt), xycoords="axes fraction", ha='right', fontsize=8)
            ax2_st.annotate(r, xy=(0.35, yt), xycoords="axes fraction", fontsize=8)
            yt -= 0.08
        ax2_st.axis("off")

        #########################
        #### FINDER IMAGE
        self.get_finder_image(field_of_view=field_of_view,ax=ax1_fov)
        self.ax.set_xlabel(self.ax.get_xlabel(),fontsize=8)
        self.ax.set_ylabel(self.ax.get_ylabel(),fontsize=8)
        self.ax.set_title(self.ax.get_title(),fontsize=8)
        [label.set_fontsize(8) for label in self.ax.get_xticklabels()]
        [label.set_fontsize(8) for label in self.ax.get_yticklabels()]
        #### 

        #########################
        #### AIRMASS PLOT
        self.airmassplot(tr_midp_cal=tr_midp_cal,ax=ax4_air,describe=False)
        self.ap.ax.set_xlabel(self.ap.ax.get_xlabel(),fontsize=8)
        self.ap.ax.set_ylabel(self.ap.ax.get_ylabel(),fontsize=8)
        self.ap.ax.set_title(self.ap.ax.get_title(),fontsize=8)
        self.ap.ax.legend(self.ap.ax.get_legend_handles_labels()[-1],loc="upper left",fontsize=7)
        [label.set_fontsize(8) for label in self.ap.ax.get_xticklabels()]
        [label.set_fontsize(8) for label in self.ap.ax.get_yticklabels()]
        #### 

        #########################
        #### EXPECTED LIGHT CURVE
        self.plot_expected_light_curve(cadence=cadence,error=error,ax=ax5_tra,offset=offset,bin_size_min=bin_size_min)
        self.ax.set_xlabel(self.ax.get_xlabel(),fontsize=8)
        self.ax.set_ylabel(self.ax.get_ylabel(),fontsize=8)
        self.ax.set_title(self.ax.get_title(),fontsize=8,y=1.0)
        self.ax.legend(self.ax.get_legend_handles_labels()[-1],loc="lower left",fontsize=6)
        [label.set_fontsize(8) for label in self.ax.get_xticklabels()]
        [label.set_fontsize(8) for label in self.ax.get_yticklabels()]
        [label.set_rotation(10) for label in self.ax.get_xticklabels()]
        #### 

        if save:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            if savesuffix!="":
                name = savedir+gkastro.compactString(self._pl_name)+"_"+gkastro.compactString(self.location)+"_"+savesuffix+"_overviewpanel"+saveext
                self.fig.savefig(name)
                print("Saved to: ",name)
            else:
                name = savedir+gkastro.compactString(self._pl_name)+"_"+gkastro.compactString(self.location)+"_overviewpanel"+saveext
                self.fig.savefig(name)
                print("Saved to: ",name)

        #########################
        #### Table
        #import matplotlib
        #def plot_table(df,ax=None):
        #    if ax == None:
        #        fig, ax = plt.subplots()     
        #    ax.table(cellText=np.array(df),colLabels=df.columns,loc='center')
        #    ax.axis('off')
        # dff = df_obs[["tr_midp_cal","tr_midp_jd","tr_midp_airm","tr_start_jd"]]
        # dff = dff["2017-05-10 06:54:41.645":"2017-07-01 06:54:41.645"]
        # plot_table(dff,ax=ax1_tex)
        ####

    def plot_airmassplot_panel(self,df_filter,startdate=None,stopdate=None,verbose=False,ncols=4,save=False,savesuffix="",savedir="plots/",saveext=".png"):
        """
        Plot a 4 column panel of transit plots
        
        INPUT:
        - df_filter is a filtered dataframe from planet.next_transits()
        
        OUTPUT:
        - fig, ax
        
        NOTES:
        """
        num_plotting = len(df_filter)

        ncols = ncols # fixed
        nrows = np.floor(num_plotting/ncols)+1.
        num_plots = nrows*ncols

        self.fig, self.ax = plt.subplots(nrows=int(nrows),ncols=int(ncols), figsize=(ncols*4.,nrows*4.))
        #self.fig.suptitle("Airmass plots: "+str(self._pl_name)+", "+self.location,y=0.95,fontsize=24)
        
        # Loop through and plot individual plots
        for i in range(num_plotting):
            if verbose:
                print("Plotting",i,df_filter.tr_midp_cal[i])
            self.airmassplot(tr_midp_cal=df_filter.tr_midp_cal[i],startdate=startdate,stopdate=stopdate,ax=self.ax.flat[i],describe=False)
            self.ap.ax.set_xlabel(self.ap.ax.get_xlabel(),fontsize=8)
            self.ap.ax.set_ylabel(self.ap.ax.get_ylabel(),fontsize=8)
            #self.ap.ax.set_title("\n".join(self.ap.ax.get_title().split("\n")[1:]),fontsize=10,y=0.99)
            self.ap.ax.set_title(self.ap.ax.get_title(),fontsize=10,y=0.99)
            self.ap.ax.legend(self.ap.ax.get_legend_handles_labels()[-1],loc="upper left",fontsize=6)
            [label.set_fontsize(6) for label in self.ap.ax.get_xticklabels()]
            [label.set_rotation(90) for label in self.ap.ax.get_xticklabels()]
            [label.set_fontsize(6) for label in self.ap.ax.get_yticklabels()]
            self.ap.ax.xaxis.labelpad = 2
            self.ap.ax.yaxis.labelpad = 0
            self.ap.ax.tick_params(axis='x', which='major', pad=1)
            self.ap.ax.tick_params(axis='y', which='major', pad=3)

        # Turn off unused plots
        for i in range(int(num_plotting),int(num_plots)):
            self.ax.flat[i].axis("off")

        self.fig.tight_layout()

        if save:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            if savesuffix!="":
                name = savedir+gkastro.compactString(self._pl_name)+"_"+gkastro.compactString(self.location)+"_"+savesuffix+"_airmasspanel"+saveext
                self.fig.savefig(name)
                print("Saved to: ",name)
            else:
                name = savedir+gkastro.compactString(self._pl_name)+"_"+gkastro.compactString(self.location)+"_airmasspanel"+saveext
                self.fig.savefig(name)
                print("Saved to: ",name)

    def plot_planet_param_table(self,ax=None,fontsize=8,linespacing=0.05):
        """
        Plot a labels and values table on a matplotlib axis
        """
        _ = self.get_latex_table()

        if ax==None:
            fig, ax = plt.subplots()
            
        ########################################
        ##### TEXT - PLANET
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ltable, rtable = [], []
        ltable += ["Planet:"]
        rtable += [self._pl_name]
        ltable += list(self.df_planet_param["Labels"].values)
        rtable += list(self.df_planet_param["values"].values)
        yt = 0.975
        for l,r in zip(ltable, rtable):
            ax.annotate(l, xy=(0.05, yt), xycoords="axes fraction", ha='right', fontsize=fontsize)
            ax.annotate(r, xy=(0.10, yt), xycoords="axes fraction", fontsize=fontsize)
            yt -= linespacing
        ax.axis("off")

        ########################################
        ##### TEXT - STAR
        ltable, rtable = [], []
        params = ["_st_spstr","_st_uj","_st_bj","_st_vj","_st_rc","_st_ic","_st_j","_st_h","_st_k"]
        params_with_errors = ["_st_rad","_st_mass","_st_teff","_st_metfe","_st_logg"]
        ltable += ["Star:"]
        rtable += [""]
        ltable += ["SpType:","U","B","V","R","I","J","H","K",r"$R_*$",r"$M_*$",r"$T_{eff}$","[Fe/H]",r"$\log(g)$"]
        rtable += [getattr(self,param) for param in params]
        rtable += [mcFunc.latex_mean_low_up(*self.get_attributeAndError(param)) for param in params_with_errors]
        yt = 0.975
        for l,r in zip(ltable, rtable):
            ax.annotate(l, xy=(0.65, yt), xycoords="axes fraction", ha='right', fontsize=fontsize)
            ax.annotate(r, xy=(0.70, yt), xycoords="axes fraction", fontsize=fontsize)
            yt -= linespacing   

    @property
    def impact_parameter(self):
        self.a   = (self._pl_orbsmax*aconst.au)/(self._st_rad*aconst.R_sun)
        return self.a*np.cos(np.deg2rad(self._pl_orbincl))

    @property
    def transit_duration(self):
        b = self.impact_parameter
        a = self.a*self._st_rad
        tdur = (self._pl_orbper/np.pi)*np.arcsin(np.sqrt(((self._st_rad)**2. - ((b*self._st_rad)**2.))/a))
        return tdur

    def get_rv_curve(self,times_jd,plot=True, ax=None, verbose=True, plot_tnow=True):
        """
        Calculate the radial velocities at given times *times_jd*

        EXAMPLE:
        nex = astropylib.nexopl.NExSciEphem()
        planet = nex.get_exoplanet("GJ 436b")
        t = astropylib.gkastro.arange_time('2018-02-25T05:00:00','2018-03-25T05:00:00',form="jd",delta=0.1)
        r = planet.get_rv_curve(t,plot=True)
        """
        rvs = obs_help.get_rv_curve(times_jd,self._pl_orbper,
                                                     self._pl_tranmid,
                                                     self._pl_orbeccen,
                                                     self._pl_orblper,
                                                     self._pl_rvamp,plot=plot,verbose=verbose,ax=ax,plot_tnow=plot_tnow)
        return rvs

class NExSciEphem(object):
    """
    A class to interface with the file "exoplanet_transits.csv"
    """
    mainkeys = ["pl_hostname",
                "pl_orbper",
                "pl_orbpererr1",
                "pl_orbpererr2",
                "pl_trandur",
                "pl_trandurerr1",
                "pl_trandurerr2",
                "pl_tranmid",
                "pl_tranmiderr1",
                "pl_tranmiderr2",
                "pl_name",
                "ra_str",
                "dec_str"]

    #def __init__(self,filename="/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/nexscidata_20170315.csv"):
    #def __init__(self,filename=MASTER_DIR+"/data/nexscidatabase/nexscidata_20170708.csv"):
    #def __init__(self,filename=MASTER_DIR+"/data/nexscidatabase/nexscidata_20171027.csv"):
    def __init__(self,filename="../data/nexscidatabase/nexscidata_20180324.csv"):
        self.df_exo = pd.read_csv(filename,sep=",",comment="#")
        # Attributes are all the keys in the .csv file
        self.exo_attributes = self.df_exo.keys()

    def add_df(self,filename):
        """
        Add another df
        "/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/OTHER_CANDIDATES/other_candidates.csv"
        """
        keys = self.df_exo.keys().tolist()
        self.df_exo2 = pd.read_csv(filename)
        self.df_exo = pd.concat([self.df_exo,self.df_exo2])
        self.df_exo = self.df_exo[keys]
        print("Added",len(self.df_exo2),"PLANETS from",filename)

    def add_df_k2cand(self,filename="/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/k2candidates_20170406.csv"):
        """
        """
        keys = self.df_exo.keys().tolist()
        df_exo_k2cand = pd.read_csv(filename,comment="#")
        df_exo_k2cand = df_exo_k2cand[df_exo_k2cand.k2c_disp!="CONFIRMED"]
        # hack pl_names
        df_exo_k2cand["pl_name"] = "CAND:"+df_exo_k2cand["epic_candname"]
        # Deal with duplicates
        df_dupl = df_exo_k2cand[df_exo_k2cand.duplicated("pl_name")]
        df_dupl["pl_name"] = df_dupl["pl_name"]+"dupl"
        df_exo_k2cand = df_exo_k2cand.reset_index()

        # Fix RA / dec sometimes missing
        for i,ra in enumerate(df_exo_k2cand.ra_str.values):
            if type(ra)==float:
                epicnumber = int(df_exo_k2cand.epic_name[i][5:])
                df_exo_k2cand.ra_str[i], df_exo_k2cand.dec_str[i] = k2help.get_k2_star_ra_dec(epicnumber)
        # Hack in pl_name values
        #for i,name in enumerate(self.df_exo_k2cand.pl_name.values):
        #    try:
        #        if np.isnan(name):
        #            self.df_exo_k2cand.pl_name.values[i] = "CAND:"+self.df_exo_k2cand.epic_candname.values[i]
        #    except TypeError:
        #        pass

        self.df_exo_k2cand = df_exo_k2cand
        self.df_exo = pd.concat([self.df_exo,self.df_exo_k2cand])
        print("Added",len(self.df_exo_k2cand),"K2 CANDIDATES from",filename)

    def add_df_cand(self,filename=MASTER_DIR+"/data/nexscidatabase/additional_candidates.csv"):
        """
        This is for auto added ones
        """
        df_exo_cand = pd.read_csv(filename,comment="#")
        #df_exo_cand["pl_name"] = "Candidate: "+df_exo_cand["pl_name"]
        # Deal with duplicates
        self.df_exo_cand = df_exo_cand.reset_index(drop=True)
        self.df_exo = pd.concat([self.df_exo,self.df_exo_cand])
        print("Added",len(self.df_exo_cand),"CANDIDATES from",filename)


    def add_dfs(self):
        """
        Convenience function to add more dfs:
        self.add_df("/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/OTHER_CANDIDATES/other_candidates.csv")
        self.add_df("/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/OTHER_CANDIDATES/dressing_candidates.csv")
        """
        print("Planets in NEXSCIDATABASE:",len(self.df_exo))
        self.add_df("/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/OTHER_CANDIDATES/other_candidates.csv")
        self.add_df("/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/OTHER_CANDIDATES/dressing_candidates.csv")
        self.add_df("/Users/gks/Dropbox/mypylib/notebooks/GIT/astropylib/data/nexscidatabase/yu_c16_candidates_full.csv")
        print("NOT ADDING K2 Candidates")
        #self.add_df_k2cand("/Users/gks/Dropbox/mypylib/data/NEXSCIDATABASE/k2candidates_20170406.csv")
        self.add_df_cand()
    
    def get_exoplanet(self,planet_name):
        # find the data for the exoplanet
        compact_planet_name = compactString(planet_name)
        # Create compact name column
        self.df_exo["compact_name"] = [compactString(i) for i in self.df_exo.pl_name]
        df_for_exoplanet = self.df_exo[self.df_exo["compact_name"]==compact_planet_name]
        
        # Create an exoplanet class using the exoplanet data, populate the attributes
        exoplanet = NExopl(self.exo_attributes,df_for_exoplanet)
        
        # Create an attribute in this instance of this class for this planet
        setattr(self,planet_name,exoplanet)
        return exoplanet

    def get_exoplanet_transits(self,startdate=None,stopdate=None,location="Apache Point Observatory",return_observable=True,return_all_params=True):
        """
        A function to get all exoplanet transits fast between start and stop dates.
        """
        jd_start = astropy.time.Time(startdate,format="iso").jd
        jd_stop  = astropy.time.Time(stopdate, format="iso").jd

        # Find transiting planets
        self.df_transiting = self.df_exo[(np.isfinite(self.df_exo.pl_tranmid)) & (np.isfinite(self.df_exo.pl_orbper))].reset_index()
        n_planets = len(self.df_transiting)

        pl_names      = self.df_transiting.pl_name.values
        pl_ra         = self.df_transiting.ra_str.values
        pl_dec        = self.df_transiting.dec_str.values

        pl_tranmid    = np.nan_to_num(self.df_transiting.pl_tranmid.values)
        pl_tranmiderr = np.nan_to_num(self.df_transiting.pl_tranmiderr1.values)
        pl_orbper     = np.nan_to_num(self.df_transiting.pl_orbper.values)
        pl_orbpererr  = np.nan_to_num(self.df_transiting.pl_orbpererr1.values)
        pl_trandur    = np.nan_to_num(self.df_transiting.pl_trandur.values)
        pl_trandurerr = np.nan_to_num(self.df_transiting.pl_trandurerr1.values)

        # Names
        f_pl_names      = []
        f_pl_ra         = []
        f_pl_dec        = []

        f_pl_tranmid    = []   
        f_pl_tranmiderr = []
        f_pl_orbper     = []
        f_pl_orbpererr  = []
        f_pl_trandur    = []
        f_pl_trandurerr = []

        # START
        f_pl_start =      []
        f_pl_start_noerr =[]
        f_pl_start_alt =  []
        f_pl_start_airm = []

        # END
        f_pl_stop =       []
        f_pl_stop_noerr = []
        f_pl_stop_alt =   []
        f_pl_stop_airm =  []

        # Loop over all of the planets
        for i in range(n_planets):
            num2start = int((jd_start - pl_tranmid[i])/pl_orbper[i])
            num2stop  = int((jd_stop - pl_tranmid[i])/pl_orbper[i])+1
            
            # What number of transits to focus on
            n_transits = np.arange(num2start,num2stop)
            
            # Initialize arrays to contain those
            _pl_names     = [pl_names[i] for kk in range(len(n_transits))]
            _pl_ra        = [pl_ra[i]    for kk in range(len(n_transits))]
            _pl_dec       = [pl_dec[i]   for kk in range(len(n_transits))]

            _pl_tranmid   = np.zeros(len(n_transits))
            _pl_tranmiderr= np.zeros(len(n_transits))
            
            # Calculate timings
            for k,n in enumerate(n_transits):
                _pl_tranmid[k]    = pl_tranmid[i] + n*pl_orbper[i]
                #if np.isnan(pl_tranmiderr[i]):
                #    #print("self._pl_tranmiderr is NaN, setting as 0")
                #    pl_tranmiderr[i] = 0.
                #if np.isnan(pl_orbpererr[i]):
                #    #print("self._pl_orbpererr is NaN, setting as 0")
                #    pl_orbpererr[i] = 0.
                _pl_tranmiderr[k] = pl_tranmiderr[i] + n*pl_orbpererr[i]

            # Calculating start and stop times
            _pl_start_noerr = _pl_tranmid - pl_trandur[i]/2.#24.*60.)
            _pl_start       = _pl_start_noerr - _pl_tranmiderr - pl_trandurerr[i]/2.#*24.*60.)
            _pl_stop_noerr  = _pl_tranmid + pl_trandur[i]/2.#24.*60.)
            _pl_stop        = _pl_stop_noerr + _pl_tranmiderr + pl_trandurerr[i]/2.#*24.*60.)
            
            f_pl_names.append(_pl_names)
            f_pl_ra.append(_pl_ra)
            f_pl_dec.append(_pl_dec)

            f_pl_tranmid.append(_pl_tranmid)
            f_pl_tranmiderr.append(_pl_tranmiderr)

            f_pl_start_noerr.append(_pl_start_noerr)
            f_pl_start.append(_pl_start)
            f_pl_stop_noerr.append(_pl_stop_noerr)
            f_pl_stop.append(_pl_stop)

        def flat(list_of_list):
            return np.array([item for inner_list in list_of_list for item in inner_list])

        f_pl_names          = flat(f_pl_names)
        f_pl_ra             = flat(f_pl_ra)
        f_pl_dec            = flat(f_pl_dec)

        f_pl_tranmid        = flat(f_pl_tranmid)
        f_pl_tranmiderr     = flat(f_pl_tranmiderr)

        f_pl_start_noerr    = flat(f_pl_start_noerr)
        f_pl_start          = flat(f_pl_start)
        f_pl_stop_noerr     = flat(f_pl_stop_noerr)
        f_pl_stop           = flat(f_pl_stop)


        tr_midp_alt, tr_midp_airm, tr_midp_sun_alt = gkastro.calc_altitude_airmass(f_pl_ra,f_pl_dec,f_pl_tranmid,timeformat="jd",location=location,verbose=False,sun=True)
        tr_start_alt, tr_start_airm                = gkastro.calc_altitude_airmass(f_pl_ra,f_pl_dec,f_pl_start,timeformat="jd",location=location,verbose=False)
        tr_stop_alt,  tr_stop_airm                 = gkastro.calc_altitude_airmass(f_pl_ra,f_pl_dec,f_pl_stop,timeformat="jd",location=location,verbose=False)

        tr_midp         = astropy.time.Time(f_pl_tranmid,format      = "jd")
        tr_midp_err_min = f_pl_tranmiderr*24.*60.
        tr_midp_err     = astropy.time.Time(f_pl_tranmiderr,format   = "jd")
        tr_start        = astropy.time.Time(f_pl_start,format        = "jd")
        tr_stop         = astropy.time.Time(f_pl_stop,format         = "jd")

        tr_start_noerr = f_pl_start_noerr
        tr_stop_noerr  = f_pl_stop_noerr
        
        tr_midp_is_sun_up = tr_midp_sun_alt > -10.
        tr_midp_is_target_up = tr_midp_alt > 20.
        tr_midp_is_target_observable = (tr_midp_is_sun_up == False ) & (tr_midp_is_target_up == True )

        data = zip(f_pl_names,
                   tr_midp.jd,
                   tr_midp.iso,
                   tr_midp_err.jd,
                   tr_midp_err_min,
                   tr_midp_alt,
                   tr_midp_airm,
                   tr_midp_sun_alt,
                   tr_midp_is_sun_up,
                   tr_midp_is_target_up,
                   tr_midp_is_target_observable,
                   tr_start.jd,
                   tr_start.iso,
                   tr_start_noerr,
                   tr_start_alt,
                   tr_start_airm,
                   #tr_start_noerr_alt,
                   tr_stop.jd,
                   tr_stop.iso,
                   tr_stop_noerr,
                   tr_stop_alt,
                   tr_stop_airm)

        columns = ["pl_name",
                   "tr_midp_jd",
                   "tr_midp_cal",
                   "tr_midp_err_jd",
                   "tr_midp_err_min",
                   "tr_midp_alt",
                   "tr_midp_airm",
                   "tr_midp_sun_alt",
                   "tr_midp_is_sun_up",
                   "tr_midp_is_target_up",
                   "tr_midp_is_target_observable",
                   "tr_start_jd",
                   "tr_start_cal",
                   "tr_start_noerr_jd",
                   "tr_start_alt",
                   "tr_start_airm",
                   #"tr_start_noerr_alt",
                   "tr_stop_jd",
                   "tr_stop_cal",
                   "tr_stop_noerr_jd",
                   "tr_stop_alt",
                   "tr_stop_airm"]

        dff = pd.DataFrame(data,columns=columns)
        dff = dff.set_index("tr_midp_cal",drop=False)
        dff.index = pd.to_datetime(dff.index)
        dff = dff[startdate:stopdate]
        if return_observable:
            dff = dff[dff.tr_midp_is_target_observable]
        if return_all_params:
            dff = pd.merge(dff,self.df_transiting)
        #dff = dff.sort()
        return dff




    def get_exolist_transits(self,planet_name_list,startdate=None,enddate=None,location="Apache Point Observatory",return_planets=False,plot=False):
        """
        A way to loop over all of the planets in a list

        nex = nexopl.NExSciEphem()
        df_obs, planets = nex.get_exolist_transits(["TRAPPIST 1 b","TRAPPIST 1 c"],
                                          startdate="2017-08-01 00:00:00.0",
                                          enddate="2017-12-01 00:00:00.0",
                                          return_planets=True,
                                          plot=True)

        exoList = [ "K2-45 b", 
                    "Kepler-1258 b", 
                    "Kepler-814 b", 
                    "NEW_GJ 436 b", 
                    "Kepler-1437 b", 
                    "Kepler-273 b",
                    "Kepler-1408 b"]
        nex = nexopl.NExSciEphem()
        df_obs, planets = nex.get_exolist_transits(exoList,
                                          startdate="2017-05-08 00:00:00.0",
                                          enddate="2017-05-09 00:00:00.0",
                                          return_planets=True,
                                          plot=True)
        """
        planets = []
        df_list = []
        if plot:
            self.fig, self.ax = plt.subplots()

        for i,planet_name in enumerate(planet_name_list):
            planet = self.get_exoplanet(planet_name)
            if return_planets==True:
                planets.append(planet)
            df = planet.next_transits(planet.num_transits_to_FINALDATE,location=location)
            df_obs = df[df.tr_midp_is_target_observable==True]
            df_obs = df_obs[startdate:enddate]
            df_list.append(df_obs)
            if plot:
                self.ax.plot(Time(df_obs.tr_midp_jd,format="jd").to_datetime(),i*np.ones(len(df_obs)),lw=0,marker="D",markersize=4,label=planet._pl_name)

        if plot:
                self.ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.),fontsize=12)
                self.ax.set_xlabel("Date (UT)")
                self.ax.margins(x=0.1,y=0.3)
                for label in self.ax.get_xticklabels():
                    label.set_fontsize(8)
                self.ax.tick_params(axis='x', which='major', pad=5)
                self.ax.tick_params(axis='y', which='major', pad=5)
                self.ax.minorticks_on()
                self.fig.tight_layout()

        df_all = pd.concat(df_list)
        if return_planets==False:
            return df_all
        else:
            return df_all, planets

            

    
    def search_host(self,search_name):
        # Create compact name column
        self.df_exo["compact_host_name"] = [compactString(i) for i in self.df_exo.pl_hostname]
        # search
        df_hosts = self.df_exo[self.df_exo["compact_host_name"]==compactString(search_name)]
        return df_hosts



    
# TODO:
# Evolution of midpoint error with time plot
# Calc transits of ALL planets in a given time window
# Calc transit fraction
# Finish expected transit depth plots - add errors
# Add a convenience function to add new Planets to the file and save it
# In search planet, add a re.search
# In search host, add a re.search
# Add timeline plot capability

    def airmassplot2(self,tr_midp_cal="",transit_number=None,save=False):
        """
        A function to create an airmassplot of a transit midpoint.
        
        EXAMPLE:
            ee = ExoEphem()
            k218b   = ee.get_exoplanet("K2-18b")
            df = k218b.next_transits(40,startdate="2016-10-01 00:00:00")
            df_observable = k218b.df_tr[k218b.df_tr.tr_midp_is_target_observable==True]
            df_observable
            k218b.airmassplot(tr_midp_cal="2017-10-28 10:36:05.990")
            k218b.airmassplot(transit_number=23)
        """
        self.ap = airmassplot.AirmassPlot(self._pl_name,self._ra_str,self._dec_str,location=self.location)
        
        if tr_midp_cal!="":
            dff = self.df_tr[self.df_tr.tr_midp_cal==tr_midp_cal]
        elif transit_number!=None:
            dff = self.df_tr[self.df_tr.num==transit_number]
        
        setattr(self.ap,"_t_midp",Time(dff.ix[0].tr_midp_cal))
        setattr(self.ap,"_t_start",Time(dff.ix[0].tr_start_cal))
        setattr(self.ap,"_t_start_noerr",Time(dff.ix[0].tr_start_noerr_jd,format="jd"))
        setattr(self.ap,"_t_stop",Time(dff.ix[0].tr_stop_cal))
        setattr(self.ap,"_t_stop_noerr",Time(dff.ix[0].tr_stop_noerr_jd,format="jd"))
        setattr(self.ap,"_t_midp_err_min",dff.ix[0].tr_midp_err_min)

        # Plot airmass plot
        self.ap.plot(t_midp=self.ap._t_midp,
                     t_start=self.ap._t_start,
                     t_start_noerr=self.ap._t_start_noerr,
                     t_stop=self.ap._t_stop,
                     t_stop_noerr=self.ap._t_stop_noerr,
                     t_midp_err_min=self.ap._t_midp_err_min,
                     ylim=(0,90))
        #self.ap.plot(t_midp=Time(dff.ix[0].tr_midp_cal),
        #        t_start=Time(dff.ix[0].tr_start_cal),
        #        t_start_noerr=Time(dff.ix[0].tr_start_noerr_jd,format="jd"),
        #        t_stop=Time(dff.ix[0].tr_stop_cal),
        #        t_stop_noerr=Time(dff.ix[0].tr_stop_noerr_jd,format="jd"),
        #        t_midp_err_min=dff.ix[0].tr_midp_err_min,
        #ylim=(0,90))

        if save==True:
            self.ap.fig.tight_layout()
            self.ap.fig.subplots_adjust(right=0.73,left=0.1)
            self.ap.fig.savefig(self._pl_name+"_"+str(self.ap._t_midp)+".png")


def print_transit_times(df_obs,index=0):
    """
    Could also do something like this:
    pd.set_eng_float_format(accuracy=10)
df_obs[["tr_start_jd","tr_start_noerr_jd","tr_midp_jd","tr_stop_noerr_jd","tr_stop_jd"]]
    """
    print("Start (err):\t",df_obs.tr_start_jd.values[index])
    print("Start (no_err):\t",df_obs.tr_start_noerr_jd.values[index])
    print("Mid \t \t",df_obs.tr_midp_jd.values[index])
    print("Stop (no_err) \t",df_obs.tr_stop_noerr_jd.values[index])
    print("Stop (err) \t",df_obs.tr_stop_jd.values[index])
