import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.time
import utils
import seaborn as sns; sns.set()
sns.set_context("poster",font_scale=1.2,rc={"font":"helvetica"}); 
sns.set_style("white"); #sns.set_style("ticks")
cp = sns.color_palette("colorblind") #sns.palplot(current_palette)
from matplotlib import rcParams
rcParams["savefig.dpi"] = 100
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

def calculate_transit_midpoints_with_errors(P,TC,n_transits=None,start=None,stop=None,P_err1=0.,P_err2=0.,TC_err1=0.,TC_err2=0.,plot=False,ax=None):
    """
    Calculate transit midpoints along with proper errors on the midpoint. Can plot a simple error-evolution plot
    
    INPUT:
        P - period in days
        TC- transit ephemeris in BJD
        n_transits - number of transits to calculate
        start = start date to start calculating transits e.g., "2014-11-18 18:00:25"
        stop  = date to stop calculating transits, e.g., "2020-06-01 00:00:00"
        P_err1 - abs(upper error)
        P_err2 - abs(lower error)
        TC_err1 - abs(upper error)
        TC_err2 - abs(lower error)
    
    OUTPUT:
        df - a pandas dataframe
    
    NOTES:
        units are in JD days
    
    EXAMPLE:
    T_ref = astropy.time.Time("2021-06-01 00:00:00",format='iso').jd
    n_transits1 = int((T_ref-TC_1)/P_1)
    n_transits2 = int((T_ref-TC_2)/P_2)

    df1 = calculate_transit_midpoints_with_errors(P_1,TC_1,n_transits1,P_1_err1,P_1_err2,TC_1_err1,TC_1_err2)
    df2 = calculate_transit_midpoints_with_errors(P_2,TC_2,n_transits2,P_2_err1,P_2_err2,TC_2_err1,TC_2_err2)

    fig, ax = plt.subplots(figsize=(18,8))
    df1 = calculate_transit_midpoints_with_errors(P_1,TC_1,n_transits1,P_1_err1,P_1_err2,TC_1_err1,TC_1_err2,ax=ax,plot=True)
    df2 = calculate_transit_midpoints_with_errors(P_2,TC_2,n_transits2,P_2_err1,P_2_err2,TC_2_err1,TC_2_err2,ax=ax,plot=True)
    """
    if n_transits is None and start is not None and stop is not None:
        start_jd = astropy.time.Time(start,format='iso').jd
        stop_jd  = astropy.time.Time(stop,format='iso').jd
        n_start  = np.floor( (start_jd-TC)/P ) 
        n_stop   = np.ceil(  (stop_jd-TC)/P ) 
        counters = np.arange(n_start,n_stop,step=1.)
    else:
        counters = np.arange(n_transits)
    midpoints = TC + counters*P
    midp_err1 = TC_err1 + np.abs(counters)*P_err1
    midp_err2 = TC_err2 + np.abs(counters)*P_err2
    df = pd.DataFrame(zip(midpoints,midp_err1,midp_err2,counters),columns=['TC','TC_err1','TC_err2','transit_number'])
    df.index = pd.to_datetime(utils.jd2datetime(df.TC))
    if start is not None and stop is not None:
        df = df[start:stop]
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.fill_between(df.index,y1=df.TC_err1*24.*60.,y2=-1*df.TC_err2*24.*60.,alpha=0.3)
        ax.grid(lw=0.5,alpha=0.5)
        ax.tick_params(axis='both',labelsize=10,pad=5)
        ax.minorticks_on()
        ax.set_xlabel("Date [UT]")
        ax.set_ylabel("Midpoint Error (min)")
    return df


def plot_transit_ephemeris_error_comparison(P_1,
                                            TC_1,
                                            P_1_err1,
                                            P_1_err2,
                                            TC_1_err1,
                                            TC_1_err2,
                                            P_2,
                                            TC_2,
                                            P_2_err1,
                                            P_2_err2,
                                            TC_2_err1,
                                            TC_2_err2,
                                            start="2014-11-18 18:00:00",
                                            stop="2020-06-01 00:00:00",
                                            ax=None,
                                            plot=True):
    """
    A function to plot a comparison transit ephemeris plot. 
    Ephemeris 1 (P1 and TC1) is the older, less constrained ephemeris.
    Ephemeris 2 (P2 and TC2) is the newer, better constrained ephemeris.
    
    INPUT:
        P_1       - period of ephemeris 1
        TC_1      - transit center of first transit
        P_1_err1  - error on P1
        P_1_err2  - error on P1
        TC_1_err1 - error on transit center 1
        TC_1_err2 - error on transit center 1
        P_2       - period of ephemeris 2
        TC_2      - transit center of transit 2
        P_2_err1,   error on P2
        P_2_err2,   error on P
        TC_2_err1,  error on transit center 2
        TC_2_err2,  error on transit center 2
        
    OUTPUT:
        df_com - A pandas dataframe with ephemeris errors for the comparison (df1)
        df_ref - A pandas dataframe with ephemeris errors for the reference (df2)
    
    NOTES:
    Example: Values from Chen et al. 2018
    
    # K2
    P_1      = 2.260449
    P_1_err1 = 0.000023
    P_1_err2 = 0.000023
    TC_1     = 2456980.2503
    TC_1_err1= 0.00037
    TC_1_err2= 0.00039
    print(utils.jd2datetime([TC_1])[0])
    # Joint
    P_2      = 2.2604380
    P_2_err1 = 0.0000015
    P_2_err2 = 0.0000015
    TC_2     = 2457796.26865
    TC_2_err1= 0.00048
    TC_2_err2= 0.00049
    print(utils.jd2datetime([TC_2])[0])
    df_ref, df_com =  plot_transit_ephemeris_error_comparison(P_1,
                                                              TC_1,
                                                              P_1_err1,
                                                              P_1_err2,
                                                              TC_1_err1,
                                                              TC_1_err2,
                                                              P_2,
                                                              TC_2,
                                                              P_2_err1,
                                                              P_2_err2,
                                                              TC_2_err1,
                                                              TC_2_err2,
                                                              start="2014-11-18 18:00:00",stop="2020-06-01 00:00:00",ax=None)
    """ 
    DAY_TO_MIN = 24.*60.
    # The more early ephemeris
    df1 = calculate_transit_midpoints_with_errors(P_1,
                                                  TC_1,
                                                  n_transits=None,
                                                  start=start,
                                                  stop=stop,
                                                  P_err1=P_1_err1,
                                                  P_err2=P_1_err2,
                                                  TC_err1=TC_1_err1,
                                                  TC_err2=TC_1_err2)
    # The later, better constrained ephemeris
    df2 = calculate_transit_midpoints_with_errors(P_2,
                                                  TC_2,
                                                  n_transits=None,
                                                  start=start,
                                                  stop =stop,
                                                  P_err1=P_2_err1,
                                                  P_err2=P_2_err2,
                                                  TC_err1=TC_2_err1,
                                                  TC_err2=TC_2_err2)
    # Plotting
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        
    # df_com is the earlier, less constrained ephemeris
    df_com = df1
    df_com['ephemeris_drift'] = (df_com.TC.values-df2.TC.values)*DAY_TO_MIN
    df_com['ephemeris_drift_err1'] = df_com['ephemeris_drift'] + df_com.TC_err1*DAY_TO_MIN
    df_com['ephemeris_drift_err2'] = df_com['ephemeris_drift'] - df_com.TC_err2*DAY_TO_MIN        
    if plot:
        ax.fill_between(df_com.index,df_com['ephemeris_drift_err1'],df_com['ephemeris_drift_err2'],alpha=0.2,color=cp[0])
        ax.plot(df_com.index,df_com['ephemeris_drift'],color=cp[0],alpha=0.8,ls='--',lw=1)
    
    # df_ref is the better constrained ephemeris
    df_ref = df2[df2.transit_number>=0]
    df_ref['ephemeris_drift'] = df_ref.TC.values-df_ref.TC.values
    df_ref['ephemeris_drift_err1'] = df_ref.TC_err1.values*DAY_TO_MIN
    df_ref['ephemeris_drift_err2'] = -1*df_ref.TC_err2.values*DAY_TO_MIN
    if plot:
        ax.plot(df_ref.index,df_ref['ephemeris_drift'],color=cp[1],ls='--',lw=1)
        ax.fill_between(df_ref.index,df_ref['ephemeris_drift_err1'],df_ref['ephemeris_drift_err2'],
                        alpha=0.5,color=cp[1],zorder=1)
    
        ax.grid(lw=0.5,alpha=0.5)
        ax.minorticks_on()
        ax.set_xlabel('Date [UT]')
        ax.set_ylabel('Midpoint Error [min]')
    return df_ref, df_com
