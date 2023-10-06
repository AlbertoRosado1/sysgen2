
import os
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
import healpy as hp
import fitsio
from glob import glob
from time import time
from scipy.interpolate import splrep, splev
import argparse


def extrapolate_weights(datapix, wt_map, k=1, der=0):
    # get missing weight, note: pixels must be sorted
    pix = datapix
    hpmap = wt_map
    is_unseen = hpmap[pix] == hp.UNSEEN
    t0 = time()
    # #https://datasciencestunt.com/extrapolation-vs-interpolation/#Spline_Extrapolation_with_Python_Code
    x = pix[~is_unseen] 
    y = hpmap[x]#[pix][~is_unseen]
    #print(x,y)
    # Spline extrapolation function
    spl = splrep(x, y, k=k)
    x_new = pix[is_unseen]
    #print(is_unseen,x_new)
    y_new = splev(x_new, spl, der=der)
    print(f"Spline extrapolation finished at {time() - t0:.3f} s")
    
    return x_new, y_new

def sn_wts(data, sn, normalize=False, plot_weights=False):
    dpixs = sysnet_tools.radec2hpix(nside,data['RA'],data['DEC'])
    is_unseen = sn[dpixs] == hp.UNSEEN
    print('unseen values: ', is_unseen.sum())
    if is_unseen.sum() != 0:
        # extrapolate any missing weights
        pix = np.unique(dpixs)
        pixi,wts = extrapolate_weights(np.sort(pix), sn, k=1, der=0)
        print(sn[pixi],sn[pixi].size)
        sn[pixi] = wts
        print(sn[pixi],sn[pixi].size)
        if plot_weights:
            plt.scatter(pix,sn[pix])
            plt.scatter(pixi,sn[pixi],c='r')
            plt.show()
    #print(dpixs,sn[dpixs])
    print(np.mean(sn[dpixs]),np.median(sn[dpixs]))
    if normalize: wts = sn[dpixs]/np.median(sn[dpixs])
    else: wts = sn[dpixs]
    print('sn_wts min/max: ',wts.min(),',',wts.max())
    return wts