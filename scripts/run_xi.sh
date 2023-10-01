#!/bin/bash
#from https://desi.lbl.gov/trac/wiki/keyprojects/y1kp3/Y1details
# bash run_pk.sh
source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 
PYTHONPATH=$PYTHONPATH:$HOME/LSS/py

xirun=/global/homes/a/arosado/LSS/scripts/xirunpc.py

tracer=$1 #ELG_LOP_complete_gtlimaging
nran=4

start_idx=0
end_idx=24

for (( VARIABLE=$start_idx; VARIABLE<=$end_idx; VARIABLE++ ))  
do
    mockid=$VARIABLE
    basedir=/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/mock${mockid}/
    # clean pk
    srun -n 1 python $xirun --nthreads 128 --corr_type smu --njack 0 --tracer $tracer --weight_type default --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir/xi

    # cont pk
    srun -n 1 python $xirun --nthreads 128 --corr_type smu --njack 0 --tracer $tracer --weight_type default_cont --nran $nran --region NGC SGC --basedir $basedir --outdir $basedir/xi
done
