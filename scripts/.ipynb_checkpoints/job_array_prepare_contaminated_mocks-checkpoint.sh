#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -J prep_Y1mocks
#SBATCH -t 5
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

python $pydir/prepare_contaminated_mocks.py --type $tracer --survey_version $version --nran $nran --mockid $mockid