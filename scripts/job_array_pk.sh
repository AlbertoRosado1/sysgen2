#!/bin/bash
#SBATCH -N 3
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J pk_Y1mocks
#SBATCH -t 00:10:00
#SBATCH -L SCRATCH
#SBATCH -o slurm_outputs/%A_%a.out
#SBATCH -e slurm_outputs/%A_%a.err
#SBATCH --array=0-24

echo $SLURM_ARRAY_TASK_ID

source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 
PYTHONPATH=$PYTHONPATH:$HOME/LSS/py

pkrun=/global/homes/a/arosado/LSS/scripts/pkrun.py

tracer=ELG_LOP_complete_gtlimaging
nran=4
mockid=$SLURM_ARRAY_TASK_ID
basedir=/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/mock${mockid}/

# null pk
srun -N 1 -n 128 python $pkrun --tracer $tracer --weight_type default --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir & 

# cont pk
srun -N 1 -n 128 python $pkrun --tracer $tracer --weight_type default_cont --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir &

# clean pk
srun -N 1 -n 128 python $pkrun --tracer $tracer --weight_type default_clean --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir &
wait

