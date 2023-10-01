# script to prepare contaminated 2dgen mocks for SYSNet training.
# example to run 
# python prepare_contaminated_mocks.py --type LRG --nran 15 --mockid 5

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

parser = argparse.ArgumentParser()
parser.add_argument("--mockid",help="mock id",type=int)
parser.add_argument("--type", help="tracer type to be selected")
parser.add_argument("--survey_version", help="survey version",default='v0.6')
parser.add_argument("--nran",help="number of randoms to use",default=4,type=int)
parser.add_argument("--imsys_nside",help="healpix nside used for imaging systematic regressions",default=256,type=int)
parser.add_argument("--mockdir", help="base mock directory for input",default="/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/")
parser.add_argument("--outdir", help="directory for output",default="/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/")
opt = parser.parse_args()
print(opt)

# parameters (assign argparse to each)
tp = opt.type # tracer type ['ELG_LPOnotqso','LRG','QSO]
sv = opt.survey_version # survey version for getting imaging maps
nside = opt.imsys_nside
nran = opt.nran
nest=True
mockid = opt.mockid

# directories
survey_dir = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{sv}/"
mockdir =  opt.mockdir #'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/'
outdir  = opt.outdir 

prep_mockdir = outdir+f"/mock{mockid}/sysnet_clean"
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
    
# load mock
datl = []
datn = []
dats = []
for reg in ['NGC','SGC']:
    dfn = os.path.join(mockdir,f"mock{mockid}",f"{tp}_{reg}_clustering.dat.fits")  
    print(dfn)
    dat = fitsio.read(dfn)

    if reg == 'NGC':
        is_north = dat['DEC'] >= 32.375
        datn.append(dat[is_north])
        dats.append(dat[~is_north])
    else:
        dats.append(dat)
    datl.append(dat)
dat = np.concatenate(datl)
datn = np.concatenate(datn)
dats = np.concatenate(dats)

# load mock randoms
ranl = []
rands_n = []
rands_s = []
for reg in ['NGC','SGC']:
    rfn_l = [os.path.join(mockdir,f"mock{mockid}",f"{tp}_{reg}_{i}_clustering.ran.fits") for i in range(0,nran)]
    #print(rfn_l)
    for rfn in rfn_l:
        ran = fitsio.read(rfn, columns=['RA', 'DEC']) 
        if reg == 'NGC':
            is_north = ran['DEC'] >= 32.375
            rands_n.append(ran[is_north])
            rands_s.append(ran[~is_north])
        else:
            rands_s.append(ran)
        ranl.append(ran)
drands = np.concatenate(ranl)
rands_n = np.concatenate(rands_n)
rands_s = np.concatenate(rands_s)

# load Ronpu EBV maps
dirmap = '/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/v0/kp3_maps/'
eclrs = ['gr','rz']
debv = Table()
for ec in eclrs:
    ebvn = fitsio.read(dirmap+'v0_desi_ebv_'+ec+'_'+str(nside)+'.fits')
    debv_a = ebvn['EBV_DESI_'+ec.upper()]-ebvn['EBV_SFD']
    debv_a = hp.reorder(debv_a,r2n=True)
    debv['EBV_DIFF_'+ec.upper()] = debv_a

# tabulate mock into format needed to use SYSNet
for zl in zrl:
    zw = str(zl[0])+'_'+str(zl[1])
    zmin,zmax = zl[0],zl[1]
    for reg in ['N','S']:
        if tp[:3] == 'ELG':
            pwf = os.path.join(survey_dir,'hpmaps',"QSO"+'_mapprops_healpix_nested_nside'+str(nside)+'_'+reg+'.fits')
        else:
            pwf = os.path.join(survey_dir,'hpmaps',tp[:3]+'_mapprops_healpix_nested_nside'+str(nside)+'_'+reg+'.fits')
        sys_tab = Table.read(pwf)
        cols = list(sys_tab.dtype.names)
        for col in cols:
            if 'DEPTH' in col:
                bnd = col.split('_')[-1]
                sys_tab[col] *= 10**(-0.4*common.ext_coeff[bnd]*sys_tab['EBV'])
        for ec in ['GR','RZ']:
            if 'EBV_DIFF_'+ec in fit_maps: 
                sys_tab['EBV_DIFF_'+ec] = debv['EBV_DIFF_'+ec]
        
        if reg == 'N':
            d,r = datn,rands_n
        else:
            d,r = dats, rands_s
        zgood = (d['Z'] > zmin) & (d['Z'] < zmax)
        
        wsys = d['WEIGHT_SNCONT'][zgood]
        data_hpmap, rands_hpmap = sysnet_tools.hpixelize(nside, d[zgood], r, weights=wsys,
                                                         nest=False, return_mask=False, nest2ring=False) 
        hpmaps = sysnet_tools.create_sysmaps(sys_tab, nest=nest, columns=fit_maps)
        prep_table = sysnet_tools.hpdataset(data_hpmap, rands_hpmap, hpmaps, fit_maps)
        fnout = os.path.join(prep_mockdir,'prep_'+tp+zw+'_'+reg+'.fits')
        print(fnout)
        fitsio.write(fnout,prep_table)
