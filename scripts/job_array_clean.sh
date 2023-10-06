#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -J clean_Y1mocks
#SBATCH -t 10
#SBATCH -L SCRATCH
#SBATCH --mem=60GB
#SBATCH -o slurm_outputs/%A_%a.out
#SBATCH -e slurm_outputs/%A_%a.err
#SBATCH --array=0-24

echo $SLURM_ARRAY_TASK_ID

source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

tracer=ELG_LOP_complete_gtlimaging 
mockid=$SLURM_ARRAY_TASK_ID
do_randoms=n # whether or not to add NN weight to randoms, at the moment array of ones
echo $tracer

pydir=$HOME/sysgen2/py

python $pydir/clean_mocks.py --mockid $mockid --type $tracer --do_randoms $do_randoms
# python $HOME/sysgen2/py/clean_mocks.py --mockid 0 --type ELG_LOP_complete_gtlimaging