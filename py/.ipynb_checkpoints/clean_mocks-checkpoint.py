# script that generates NN weights hpmap and and adds NN weights to mocks as WEIGHT_SN
# example to run 
# python clean_mocks.py --type ELG_LOP_complete_gtlimaging --mockid 5

import os
import numpy as np
import healpy as hp
import fitsio
import argparse
from time import time
from scipy.interpolate import splrep, splev
from matplotlib import pyplot as plt
from astropy.table import Table
from glob import glob

import sys 
sys.path.append(os.path.join(os.getenv("HOME"),'LSS','py'))
from LSS.imaging import sysnet_tools

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
parser.add_argument("--imsys_nside",help="healpix nside used for imaging systematic regressions",default=256,type=int)
parser.add_argument("--mockdir", help="base mock directory for input",default="/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/")
parser.add_argument("--outdir", help="directory for output",default="/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/")
parser.add_argument("--clip_snwts", help="clip SYSNet predictions",choices=['y','n'],default="y")
parser.add_argument("--do_randoms", help="add column of NN weights as ones",choices=['y','n'],default="y")
opt = parser.parse_args()
print(opt)

# parameters (assign argparse to each)
tp = opt.type # tracer type ['ELG_LPOnotqso','LRG','QSO]
#sv = opt.survey_version # survey version
nside = opt.imsys_nside
#nran = opt.nran
nest=True # survey hpmaps ordering
mockid = opt.mockid
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
mockdir =  opt.mockdir #'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/'
sysdir = os.path.join(mockdir,f'mock{mockid}','sysnet_clean')
outdir  = opt.outdir #os.path.join(os.getenv('HOME'),'sysgen2')
prep_mockdir = os.path.join(outdir,f'mock{mockid}','sysnet_clean')
print(prep_mockdir)
if not os.path.isdir(prep_mockdir):
    os.makedirs(prep_mockdir)
    print(f"created dir: {prep_mockdir}")

# redshift bins tracer was trained on
if tp[:3] == 'ELG':
    zrl = [(0.8,1.1),(1.1,1.6)]
    fit_maps = ['STARDENS','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','EBV_DIFF_GR','EBV_DIFF_RZ','HI']
if tp[:3] == 'QSO':
    zrl = [(0.8,1.3),(1.3,2.1)]# mocks do not have ,(2.1,3.5)] 
    fit_maps = ['STARDENS','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','EBV_DIFF_GR','EBV_DIFF_RZ','HI']
    fit_maps.append('PSFDEPTH_W1')
    fit_maps.append('PSFDEPTH_W2')
if tp[:3] == 'LRG':
    zrl = [(0.4,0.6),(0.6,0.8),(0.8,1.1)]  
    fit_maps = ['STARDENS','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','HI','PSFDEPTH_W1']
if tp[:3] == 'BGS':
    zrl = [(0.1,0.4)] 
    fit_maps = ['STARDENS','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','HI']
print(len(fit_maps),fit_maps)
       
# combine N and S weight maps
for zl in zrl:
    zw = str(zl[0])+'_'+str(zl[1])
    count_i = np.zeros(12*nside*nside)
    wind_i = np.zeros(12*nside*nside)
    for reg in ['N','S']:
        print(reg)
        nnfn  = os.path.join(sysdir,f"{tp}{zw}_{reg}","nn-weights.fits")
        nn = fitsio.read(nnfn)
        wt = np.mean(nn['weight'],axis=1)
        wind_i[nn['hpix']] += wt
        count_i[nn['hpix']] += 1.0
        
    output_path = os.path.join(prep_mockdir,f"nn-weights_{tp}{zw}_NS.fits")
    #print(output_path)
    is_good = count_i > 0.0
    wind_i[is_good] = wind_i[is_good] / count_i[is_good]
    wind_i[~is_good] = hp.UNSEEN
    hp.write_map(output_path, wind_i, dtype=np.float64, fits_IDL=False, overwrite=True) 
    print(f"saved {output_path}")

# load data
for reg in ['NGC','SGC']:
    dfn   = os.path.join(mockdir,f"mock{mockid}",f"{tp}_{reg}_clustering.dat.fits")  
    outfn = os.path.join(outdir,f"mock{mockid}",f"{tp}_{reg}_clustering.dat.fits")
    print(dfn)
    dat = Table.read(dfn)
    dat['WEIGHT_SN'] = np.ones(len(dat))
    for zl in zrl:
        zw = str(zl[0])+'_'+str(zl[1])
        zmin,zmax = zl[0],zl[1]
        print('\t',zmin,zmax)
        zgood = (zmin < dat['Z']) & (dat['Z'] < zmax)
        sn_fn = os.path.join(prep_mockdir,f"nn-weights_{tp}{zw}_NS.fits")
        sn_map = hp.read_map(sn_fn) 
        sn_wts = use_sn_wts(dat[zgood],sn_map,normalize=True,plot_weights=False,clip=clip)
        dat['WEIGHT_SN'][zgood] = 1./sn_wts
    print(f"saving {outfn}")
    dat.write(outfn,overwrite=True)
    
# load randoms, at the moment the values in WEIGHT_SN are only 1
if do_randoms:
    for reg in ['NGC','SGC']:
        rfn_l = glob(os.path.join(mockdir,f"mock{mockid}",f"{tp}_{reg}_*_clustering.ran.fits"))
        #print(rfn_l)
        for rfn in rfn_l:
            ran = Table.read(rfn) 
            ran['WEIGHT_SN'] = np.ones(len(ran))

            outfn = os.path.join(outdir,f"mock{mockid}",rfn.split('/')[-1])
            print(f"saving {outfn}")
            ran.write(outfn,overwrite=True)