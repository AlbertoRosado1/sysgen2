#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu 
#SBATCH --gpus=1
#SBATCH --account=desi_g
#SBATCH -q shared
#SBATCH -J cont_Y1mocks
#SBATCH -t 15
#SBATCH -L SCRATCH
#SBATCH --mem=60GB
#SBATCH -o slurm_outputs/%A_%a.out
#SBATCH -e slurm_outputs/%A_%a.err
#SBATCH --array=0-24

echo $SLURM_ARRAY_TASK_ID

source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

tracer=ELG_LOP_complete_gtlimaging 
version=v0.6 # survey version to get SYSNet weights and hpmaps from
nran=15 # number of randoms to use 
mockid=$SLURM_ARRAY_TASK_ID
echo $tracer

pydir=$HOME/sysgen2/py

python $pydir/forward_mocks_all.py --type $tracer --survey_version $version --nran $nran --mockid $mockid
python $pydir/cont_mocks.py --mockid $mockid --type $tracer