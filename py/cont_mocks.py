# script to contaminate second gen mocks with SYSNet weights. The models used in this script are the same used to 
# generate the WEIGHT_SN weights in the catalogs.
# example to run 
# python cont_mocks.py --mockid 10 --type LRG

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

import sys 
sys.path.append(os.path.join(os.getenv("HOME"),'LSS','py'))
from LSS.imaging import sysnet_tools
import LSS.common_tools as common

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

def use_sn_wts(data, sn, normalize=False, plot_weights=False, clip=False):
    # sn is hpmap with nn-weights
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
    if normalize: 
        wts = sn[dpixs]/np.median(sn[dpixs])
    else: 
        wts = sn[dpixs]
    if clip: 
        print(f"clipping SYSNet predictions")
        wts = wts.clip(0.5,2.0)
    print('sn_wts min/max: ',wts.min(),',',wts.max())
    return wts

parser = argparse.ArgumentParser()
parser.add_argument("--mockid",help="mock id",type=int)
parser.add_argument("--type", help="tracer type to be selected")
parser.add_argument("--mockdir", help="base mock directory for input",default="/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/")
parser.add_argument("--imsys_nside",help="healpix nside used for imaging systematic regressions",default=256,type=int)
parser.add_argument("--outdir", help="directory for output",default="/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/")
parser.add_argument("--clip_snwts", help="clip SYSNet predictions",choices=['y','n'],default="y")
parser.add_argument("--do_randoms", help="add column of NN weights as ones",choices=['y','n'],default="y")
opt = parser.parse_args()
print(opt)

# parameters (assign argparse to each)
tp = opt.type # tracer type ['ELG_LPOnotqso','LRG','QSO]
mockid = opt.mockid
nside = opt.imsys_nside
clip_snwts = opt.clip_snwts
if clip_snwts=='y':
    clip = True
else:
    clip = False
do_randoms = opt.do_randoms
if do_randoms=='y':
    do_randoms=True
else:
    do_randoms=False

# directories
mockdir =  opt.mockdir 
outdir  = opt.outdir
prep_mockdir = outdir+f"/mock{mockid}/sysnet_cont"
    
# redshift bins tracer was trained on
if tp[:3] == 'ELG':
    zrl = [(0.8,1.1),(1.1,1.6)]
if tp[:3] == 'QSO':
    zrl = [(0.8,1.3),(1.3,2.1)]# mocks do not have ,(2.1,3.5)] 
if tp[:3] == 'LRG':
    zrl = [(0.4,0.6),(0.6,0.8),(0.8,1.1)]  
if tp[:3] == 'BGS':
    zrl = [(0.1,0.4)]
print(tp,zrl)

# load data
for reg in ['NGC','SGC']:
    dfn   = os.path.join(mockdir,f"mock{mockid}",f"{tp}_{reg}_clustering.dat.fits")  
    outfn = os.path.join(outdir,f"mock{mockid}",f"{tp}_{reg}_clustering.dat.fits")
    print(dfn)
    dat = Table.read(dfn)
    dat['WEIGHT_SNCONT'] = np.ones(len(dat))
    for zl in zrl:
        zw = str(zl[0])+'_'+str(zl[1])
        zmin,zmax = zl[0],zl[1]
        print('\t',zmin,zmax)
        zgood = (zmin < dat['Z']) & (dat['Z'] < zmax)
        sn_fn = os.path.join(prep_mockdir,f"nn-weights_{tp}{zw}_NS.fits")
        sn_map = hp.read_map(sn_fn) 
        sn_wts = use_sn_wts(dat[zgood],sn_map,normalize=True,plot_weights=False,clip=clip)
        dat['WEIGHT_SNCONT'][zgood] = sn_wts
    print(f"saving {outfn}")
    dat.write(outfn,overwrite=True)
    
# load randoms, at the moment the values in WEIGHT_SNCONT are only 1
if do_randoms:
    for reg in ['NGC','SGC']:
        rfn_l = glob(os.path.join(mockdir,f"mock{mockid}",f"{tp}_{reg}_*_clustering.ran.fits"))
        #print(rfn_l)
        for rfn in rfn_l:
            ran = Table.read(rfn) 
            ran['WEIGHT_SNCONT'] = np.ones(len(ran))

            outfn = os.path.join(outdir,f"mock{mockid}",rfn.split('/')[-1])
            print(f"saving {outfn}")
            ran.write(outfn,overwrite=True)
