#!/bin/bash
#SBATCH -N 3
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J xi_Y1mocks
#SBATCH -t 00:20:00
#SBATCH -L SCRATCH
#SBATCH -o slurm_outputs/%A_%a.out
#SBATCH -e slurm_outputs/%A_%a.err
#SBATCH --array=0-24

echo $SLURM_ARRAY_TASK_ID

source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 
PYTHONPATH=$PYTHONPATH:$HOME/LSS/py

xirun=/global/homes/a/arosado/LSS/scripts/xirunpc.py

tracer=ELG_LOP_complete_gtlimaging
nran=4
mockid=$SLURM_ARRAY_TASK_ID
basedir=/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/mock${mockid}/

# null xi
srun -N 1 -n 1 python $xirun --nthreads 128 --corr_type smu --njack 0 --tracer $tracer --weight_type default --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir/xi &

# cont xi
srun -N 1 -n 1 python $xirun --nthreads 128 --corr_type smu --njack 0 --tracer $tracer --weight_type default_cont --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir/xi &

# clean xi
srun -N 1 -n 1 python $xirun --nthreads 128 --corr_type smu --njack 0 --tracer $tracer --weight_type default_clean --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir/xi &
wait 