import numpy as np
import math
import pandas as pd
import astropy.modeling.functional_models as functional_models
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.major.pad']=2
rcParams['ytick.major.pad']=0
rcParams['xtick.major.width']=1

class PriorSet(object):
    """
    Based on hpparvi's class, see here: https://github.com/hpparvi/PyExoTK

    Extending a bit to give more properties
    """
    def __init__(self, priors):
        self.priors = priors
        self.ndim  = len(priors)

        self.pmins = np.array([p.min() for p in self.priors])
        self.pmaxs = np.array([p.max() for p in self.priors])
        self.bounds= np.array([self.pmins,self.pmaxs]).T

        self.sbounds= np.array([[p.center-p.squeeze*0.5*p.width,
                                 p.center+p.squeeze*0.5*p.width] for p in self.priors])


    def generate_pv_population(self, npop):
        return np.array([[((p.random()[0]-p.center)*p.squeeze)+p.center for p in self.priors] for i in range(npop)])

    def c_log_prior(self, pv):
        return np.sum([p.log(v) for p,v in zip(self.priors, pv)])

    def c_prior(self, pv):
        return math.exp(self.c_log_prior(pv))

    @property
    def names(self):
        return [p.name for p in self.priors]
    
    # EXTENDING
    @property
    def descriptions(self):
        return [p.description for p in self.priors]
    
    @property
    def centers(self):
        return [p.center for p in self.priors]
    
    @property
    def labels(self):
        return [p.name for p in self.priors]
    
    @property
    def random(self):
        return [p.random()[0] for p in self.priors]
    
    @property
    def priortypes(self):
        return [p.priortype for p in self.priors]
    
    @property
    def detrendparams(self):
        priortypes = self.priortypes

    def get_param_type_indices(self,paramtype="detrend"):
        assert (paramtype == "detrend" or paramtype=="model" or paramtype=="fixed" or paramtype=="gp")
        priortypes = self.priortypes
        df = pd.DataFrame(priortypes,columns=["priortypes"])
        return list(df[df["priortypes"] == paramtype].index)
    
    def plot_all(self,pv=[],hspace=0.4,figsize=(6,4)):
        self.fig, self.ax = plt.subplots(nrows=self.ndim,figsize=figsize)
        self.ax.flat[0].set_title("Priors",fontsize=16)
        for i in range(self.ndim):
            self.ax.flat[i].set_ylabel(self.labels[i],fontsize=12)
            if len(pv)==0: mark=None
            else: mark=pv[i]
            self.priors[i].plot(ax=self.ax.flat[i],mark=mark,lw=2)
            self.fig.subplots_adjust(hspace=hspace)
            #ax.flat[i].yaxis.set_visible(False)
            plt.setp(self.ax.flat[i].get_yticklabels(), visible=False)
            self.ax.flat[i].set_ylabel(self.labels[i])
            self.ax.flat[i].minorticks_on()
            xticks = np.linspace(self.priors[i].a,self.priors[i].b,5)
            self.ax.flat[i].set_xticks(xticks)
            for label in self.ax.flat[i].get_xticklabels():
                label.set_fontsize(8)




class Prior(object):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.,priortype=""):
        self.a = float(a)
        self.b = float(b)
        self.center= 0.5*(a+b)
        self.width = b - a
        self.squeeze = squeeze

        self.name = name
        self.description = description
        self.unit = unit
        self.priortype = priortype

    def limits(self): return self.a, self.b 
    def min(self): return self.a
    def max(self): return self.b
   
 
class UniformPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.,priortype="model"):
        super(UniformPrior, self).__init__(a,b,name,description,unit,squeeze,priortype)
        self._f = 1. / self.width
        self._lf = math.log(self._f)

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), self._f, 1e-80)
        else:
            return self._f if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), self._lf, -1e18)
        else:
            return self._lf if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)

    def plot(self,ax=None,mark=None,ymin=-0.1,ymax=1.1,**kwargs):
        if ax==None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        box = functional_models.Box1D()
        self.x = np.linspace(self.a,self.b,100)
        self.y = box.evaluate(self.x,amplitude=1.,x_0=self.center,width=self.width)
        self.ax.plot(self.x,self.y,**kwargs)
        self.ax.set_ylim(ymin,ymax)
        self.ax.vlines(self.a,0,1,color="black",linestyle="--")
        self.ax.vlines(self.b,0,1,color="black",linestyle="--")
        self.ax.set_ylabel(self.name)
        if mark!=None:
            self.ax.vlines(mark,0,1,color="orange",linestyle="--")


class JeffreysPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.,priortype="model"):
        super(JeffreysPrior, self).__init__(a,b,name,description,unit,squeeze,priortype)
        self._f = math.log(b/a)

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), 1. / (x*self._f), 1e-80)
        else:
            return 1. / (x*self._f) if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), np.log(1. / (x*self._f)), -1e18)
        else:
            return math.log(1. / (x*self._f)) if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)


class NormalPrior(Prior):
    def __init__(self, mean, std, name='', description='', unit='', lims=None, limsigma=5, squeeze=1,priortype="model"):
        lims = lims or (mean-limsigma*std, mean+limsigma*std)
        super(NormalPrior, self).__init__(*lims, name=name, description=description, unit=unit,squeeze=squeeze,priortype=priortype)
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1./ math.sqrt(2.*math.pi*std*std)
        self._lf1 = math.log(self._f1)
        self._f2 = 1./ (2.*std*std)

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._f1 * np.exp(-(x-self.mean)**2 * self._f2), 1e-80)
        else:
            return self._f1 * exp(-(x-self.mean)**2 * self._f2) if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._lf1 - (x-self.mean)**2 * self._f2, -1e18)
        else:
            return self._lf1 -(x-self.mean)**2*self._f2 if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size) #normal(self.mean, self.std, size)

    def plot(self,ax=None,mark=None,ymin=-0.1,ymax=1.1,**kwargs):
        if ax==None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        gauss = functional_models.Gaussian1D()
        self.x = np.linspace(self.a,self.b,100)
        self.y = gauss.evaluate(self.x,amplitude=1.,mean=self.mean,stddev=self.std)
        self.ax.plot(self.x,self.y,**kwargs)
        self.ax.set_ylim(ymin,ymax)
        self.ax.vlines(self.a,0,1,color="black",linestyle="--")
        self.ax.vlines(self.b,0,1,color="black",linestyle="--")
        self.ax.set_ylabel(self.name)
        if mark!=None:
            self.ax.vlines(mark,0,1,color="orange",linestyle="--")
    

class DummyPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='',priortype="model"):
        super(DummyPrior, self).__init__(a, b, name=name, description=description, unit=unit,priortype="model")

    def __call__(self, x, pv=None):
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.

    def log(self, x, pv=None):
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)


UP = UniformPrior
JP = JeffreysPrior
GP = NormalPrior
NP = NormalPrior 
DP = DummyPrior
