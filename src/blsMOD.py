from __future__ import print_function
import bls
import numpy as np
import matplotlib.pyplot as plt


class blsMOD(object):
    """
    A class to work with Box Least Squares.
    
    See:
    - https://github.com/ridlo/exoplanet_notebook/blob/master/bls_test01.ipynb
    - https://github.com/dfm/python-bls
    
    self.t = t                # time
    self.f = f                # flux
    self.nf = nf              # number of frequency bins to test
    self.fmin = fmin          # minimum frequency to test
    self.df = df              # frequency grid spacing
    self.nb = nb              # number of phase bins
    self.qmi = qmi            # minimum fractional transit duration to test
    self.qma = qma            # maximum fractional transit duration to test
    period_range: 
        Changes fmin and df to match a given period range. nf is unchanged

    RETURNS:
        power is the nf-dimensional power spectrum array at frequencies f = fmin + arange(nf) * df,
        best_period is the best-fit period in the same units as time,
        best_power is the power at best_period,
        depth is the depth of the transit at best_period,
        q is the fractional transit duration,
        in1 is the bin index at the start of transit, and
        in2 is the bin index at the end of transit.

    EXAMPLE:
        qmi = 0.01
        qma = 0.1
        fmin = 0.3 
        df = 0.001 
        nf = 1000
        nb = 200

        bb = blsMOD.blsMOD(t,f, nf, fmin, df, nb, qmi, qma)
        bb.compute()
        bb.plot_power_spectrum()
        bb.plot_work_array()
        bb.plot_folded()

    IF USING EVEREST:
        star.mask_planet(bb._epoch,bb._best_period)
        star.compute()
        star.plot_folded(bb._epoch,bb._best_period)
    """
    
    def __init__(self,t,f,nf,fmin,df,nb,qmi,qma,period_range=None):    
        self.t = t                # time
        self.f = f                # flux
        self.u = np.zeros(len(t)) # zeros: work arrays
        self.v = np.zeros(len(t)) # zeros: work arrays
        self.nf = nf              # number of frequency bins to test
        self.fmin = fmin          # minimum frequency to test
        self.df = df              # frequency grid spacing
        self.nb = nb              # number of bins
        self.qmi = qmi            # minimum transit duration to test
        self.qma = qma            # maximum transit duration to test
        if period_range is not None:
            period_range = np.array(period_range)
            assert period_range[1] > period_range[0]
            freq_range = np.flipud(1./period_range)
            self.fmin = freq_range[0]
            self.df = np.diff(freq_range) / float(nf)
            print("Using period range",period_range)
            print("Overwriting: df=",self.df,"fmin=",fmin,"freq_range",freq_range)
        
    def compute(self,verbose=True):
        """
        Compute using bls.eebls()
        """
        self.results = bls.eebls(self.t,self.f,self.u,self.v,self.nf,self.fmin,self.df,self.nb,self.qmi,self.qma)
        self._power = self.results[0]
        self._freq = self.fmin + np.arange(self.nf)*self.df
        self._best_period = self.results[1]
        self._best_freq = 1.0/self._best_period
        self._best_power = self.results[2]
        self._depth = self.results[3]
        self._q = self.results[4]
        self._fraqdur = self.results[4]
        self._in1 = self.results[5]
        self._in2 = self.results[6]




        # Taking a look at https://github.com/dfm/python-bls/blob/master/ruth_bls2.py
        self._duration = self._best_period * self._q
        self._phase1 = self._in1/float(self.nb)
        self._phase2 = self._in2/float(self.nb)

        self._transit_number = int((max(self.t) - min(self.t)) / self._best_period)
        self._epoch = self.t[0] + self._phase1*self._best_period + self._duration/2.
        self._ingresses = np.zeros(self._transit_number)
        self._egresses = np.zeros(self._transit_number)
        for n in range(0,self._transit_number):
            self._ingresses[n] = (self._epoch + self._best_period*n) #- 0.2
            self._egresses[n] = self._epoch + self._best_period*n + self._duration# + 0.2 # add a margin each side

        self._duration_approx = self._egresses[0] - self._ingresses[0]
        

        if verbose==True:
            print("=====Results====")
            print("Best period:", self._best_period)
            print("Best freq:", self._best_freq)
            print("Depth:",self._depth)
            print("Epoch:",self._epoch)
            print("Number of transits:",self._transit_number)

    def plot_power_spectrum(self,inverse=False,ax=None):
        """
        Plot the power spectrum
        """
        if ax==None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        if inverse==False:
            self.ax.plot(self._freq,self._power,lw=1)
            self.ax.set_xlabel(r"Frequency ($d^{-1}$)")
            self.ax.plot([self._best_freq],[self._best_power],marker="o",lw=0,color="firebrick",alpha=0.7)
        else:
            self.ax.plot(1./self._freq,self._power,lw=1)
            self.ax.set_xlabel(r"Period ($d$)")
            self.ax.plot([1./self._best_freq],[self._best_power],marker="o",lw=0,color="firebrick",alpha=0.7)

        self.ax.set_ylabel(r"SR")
        
    def plot_work_array(self,ax=None):
        """
        # Looks like the light curve, except centered around 0 (flux - 1.0), and time starts at 0 (same unit; days)
        plt.plot(t,f-1,lw=1)
        #plt.plot(bb.u,bb.v)
        #plt.plot(bb.t-bb.t[0],bb.f-np.mean(bb.f),lw=1)
        """
        if ax==None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        # Looks like the light curve, except centered around 0
        self.ax.plot(self.u,self.v,lw=1)
        self.ax.set_xlabel("u")
        self.ax.set_ylabel("v")
        self.ax.minorticks_on()
        
    def plot_folded(self,with_trial=True,ax=None):
        """
        Plot the folded spectrum with the best trial period
        """
        ibi = np.zeros(self.nb)
        y = np.zeros(self.nb)
        self._phase = np.linspace(0.0, 1.0, self.nb)

        # loop over time
        for i in range(len(self.t)):
            ph = self.u[i]*self._best_freq 
            ph = ph - int(ph)
            j = int(self.nb*ph) # data to a bin 
            ibi[j] += 1.0 # number of data in a bin
            y[j] = y[j] + self.v[i] # sum of light in a bin

        self._folded_binned_flux = y/ibi

        if ax==None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        
        self.ax.plot(self._phase, self._folded_binned_flux, marker="o",ms=3,alpha=0.7,lw=0)
        self.ax.set_title("Period: {0} d  bin: {1}".format(1/self._best_freq, self.nb))
        self.ax.set_xlabel(r"Phase ($\phi$)")
        self.ax.set_ylabel(r"Mean value of $x(\phi)$ in a bin")
        self.ax.minorticks_on()
        if with_trial==True:
            self._box = np.zeros(self.nb)   # H
            self._box[self._in1:self._in2] = -self._depth # L
            self.ax.plot(self._phase, self._box,lw=1,color="firebrick")

    def plot_transits(self,ax=None):
        """
        A function to plot when it is in transit
        """
        if ax==None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        
        self.ax.plot(self.t,self.f,lw=1)
        self.ax.plot(self._ingresses,(min(self.f))*np.ones(len(self._ingresses)),marker=">",lw=0,color="red",ms=4)
        self.ax.plot(self._ingresses,(min(self.f))*np.ones(len(self._egresses)),marker="<",lw=0,color="red",ms=4)
        self.ax.minorticks_on()
