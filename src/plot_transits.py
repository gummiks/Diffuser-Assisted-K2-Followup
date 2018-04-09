import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

sns.set_context("poster",font_scale=1.2,rc={"font":"helvetica"}); 
sns.set_style("white");
cp = sns.color_palette("colorblind")
cp = sns.color_palette()
rcParams["savefig.dpi"] = 200
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.weight'] = "normal"
rcParams["axes.formatter.useoffset"] = False
rcParams['xtick.major.width']=1
rcParams['xtick.major.size']=7
rcParams['xtick.minor.width']=1
rcParams['xtick.minor.size']=4
rcParams['ytick.major.width']=1
rcParams['ytick.major.size']=7
rcParams['ytick.minor.width']=1
rcParams['ytick.minor.size']=4

def plot_transit_with_model(x,y,yerr,yresidual=None,xmodel=None,ymodel=None,offset=0.99,ax=None,
                            label_data="Data",label_model="Model",label_residual="Residual"):
    """
    Plot transit with a transit model, residual and errorbars
    
    INPUT:
        x         - time coordinate, BJD
        y         - normalized flux
        yerr      - 
        yresidual - residual
        xmodel    - x coordinate of model
        ymodel    - y coordinate of model
    
    OUTPUT:
        Plots a nice transit plot plot
        
    NOTES:
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot Data
    ax.errorbar(x,y,yerr,elinewidth=0.3,lw=0,alpha=0.7,marker="o",
                barsabove=True,markersize=4,mew=1.,capsize=4.,capthick=0.5,label=label_data,color=cp[0])
    
    # Plot Residual
    if yresidual is not None:
        ax.errorbar(x,yresidual+offset,yerr,elinewidth=0.3,lw=0,alpha=0.5,marker="o",
                    barsabove=True,markersize=4,mew=1.,capsize=4,capthick=0.5,label=label_residual,color="dimgray")

    # Plot Model
    if xmodel is not None and ymodel is not None:
        ax.plot(xmodel,ymodel,label=label_model,color="crimson",alpha=0.7)
    
    ax.minorticks_on()
    ax.set_xlabel("BJD")
    ax.set_ylabel("Normalized flux")
    ax.grid(lw=0.5,alpha=0.5)