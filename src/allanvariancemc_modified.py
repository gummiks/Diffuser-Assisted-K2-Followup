import sys
#sys.path.append("/Users/Sophie/git/MCcubed/")
#sys.path.append("/Users/gks/programs/MCcubed2/MCcubed/examples/models/")
import MCcubed as mc3
import numpy as np

# ---------- Graphics ------------
# matplotlib
import seaborn as sns; sns.set()
sns.set_context("poster",font_scale=1.2,rc={"font":"helvetica"});
sns.set_style("white"); #sns.set_style("ticks")
cp = sns.color_palette("colorblind") #sns.palplot(current_palette)
#%matplotlib osx
#%matplotlib nbagg
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["savefig.dpi"] = 100
#rcParams["text.usetex"] = True #uncomment to use tex. Slow, but pretty
#rcParams["font.weight"] = 900

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.weight'] = "normal"
rcParams["axes.formatter.useoffset"] = False
rcParams['xtick.major.width']=1
rcParams['xtick.major.size']=4
rcParams['xtick.minor.width']=0.5
rcParams['xtick.minor.size']=2
rcParams['xtick.direction'] = "in"
rcParams['ytick.direction'] = "in"
rcParams['ytick.major.width']=1
rcParams['ytick.major.size']=4
rcParams['ytick.minor.width']=0.5
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
        
    def compute_noise(self,res):
        """
        Compute the noise

        INPUT:
        x - time
        y - residuals
        rms, rmslo, rmshi, stderr, binsz = mc3.rednoise.binrms(res,maxbins)
        """
        self.red = np.array(res)
        self.maxbins =len(self.red)
        self.white=np.random.normal(0,5,self.maxbins)
        self.maxbins=self.maxbins/5
        self.res=self.white+self.red
        self.rms, self.rmslo, self.rmshi, self.stderr, self.binsz = mc3.rednoise.binrms(self.res,self.maxbins)
        return self.rms, self.rmslo, self.rmshi, self.stderr, self.binsz
     
#gx.errorbar(binsz, rms, yerr=[rmslo, rmshi], fmt="k-", ecolor='0.5', capsize=0, label="Data RMS",zorder=2)
#gx.loglog(binsz, stderr, color='red', ls='-', lw=2, label="Gaussian std.",zorder=1)
#gx.set_xlim(1,100)
#gx.scatter(10,1,label='Cadence=32.5728267431s',color='white',zorder=3) #precision 30mins bin, 1 min
#handles,labels = gx.get_legend_handles_labels()
#handles = [handles[0], handles[2], handles[1]]
#labels = [labels[0], labels[2], labels[1]]
#gx.legend(handles,labels,loc="upper right")
#gx.set_xlabel("Bin size", fontsize=14)
#gx.set_ylabel("RMS", fontsize=14)
        
    
    def plot(self,label1="",label2="",cadence=False):
        """
        
        """
        self.fig, self.ax = plt.subplots(figsize=(6,4))
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