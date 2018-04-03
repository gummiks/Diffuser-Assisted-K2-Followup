import sys
sys.path.append("/Users/gks/programs/MCcubed2/MCcubed")
sys.path.append("/Users/gks/programs/MCcubed2/MCcubed/examples/models/")
from quadratic import quad
import MCcubed as mc3
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.weight'] = "normal"
rcParams["axes.formatter.useoffset"] = False
rcParams['xtick.major.width']=2
rcParams['xtick.major.size']=7
rcParams['xtick.minor.width']=1
rcParams['xtick.minor.size']=2
rcParams['xtick.direction'] = "in"
rcParams['ytick.direction'] = "in"
rcParams['ytick.major.width']=2
rcParams['ytick.major.size']=7
rcParams['ytick.minor.width']=1
rcParams['ytick.minor.size']=2


class AllanVarianceMC(object):
    """
    A class to make an Allan Variance plot with MCCcubed code.
    
    EXAMPLE:
        avmc = AllanVarianceMC()
        avmc.compute_noise(res,20)
        fig, ax = avmc.plot()
        ax.set_xlabel("bins")
    """
    def __init__(self):
        """
        INPUT:
        """
        
    def compute_noise(self,res,maxbins):
        """
        Compute the noise

        INPUT:
        x - time
        y - residuals
        rms, rmslo, rmshi, stderr, binsz = mc3.rednoise.binrms(res,maxbins)
        """
        self.res = res
        self.maxbins = maxbins
        self.rms, self.rmslo, self.rmshi, self.stderr, self.binsz = mc3.rednoise.binrms(self.res,self.maxbins)
        return self.rms, self.rmslo, self.rmshi, self.stderr, self.binsz
        
    
    def plot(self,label1="",label2="",cadence=False):
        """
        
        """
        self.fig, self.ax = plt.subplots()
        if cadence==False:
            self.ax.errorbar(self.binsz,self.rms,yerr=[self.rmslo,self.rmshi],fmt="k-",lw=1.8, ecolor='0.5', capsize=0,label=label1,elinewidth=1)
            self.ax.plot(self.binsz,self.stderr, color='red', ls='-', lw=2, label=label2)
        else:
            self.mins = self.binsz*cadence/60.
            self.ax.errorbar(self.mins,self.rms,yerr=[self.rmslo,self.rmshi],fmt="k-",lw=1.8, ecolor='0.5', capsize=0,label=label1,elinewidth=1)
            self.ax.plot(self.mins,self.stderr, color='red', ls='-', lw=2, label=label2)
        #ax.hlines(80e-6,mins.min(),mins.max(),color="orange",alpha=1.,lw=1.5,linestyle="--",label="Transit of Earth")
        #self.ax.legend(loc="upper right",fontsize="x-large")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.minorticks_on()
        return self.fig, self.ax
