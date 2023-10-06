#!/bin/bash
# bash run_clean.sh ELG_LOP_complete_gtlimaging 0

source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

tracer=$1 #ELG_LOP_complete_gtlimaging
mockid=$2
pydir=$HOME/sysgen2/py

start_idx=0
end_idx=0

for (( VARIABLE=$start_idx; VARIABLE<=$end_idx; VARIABLE++ ))
do
    mockid=$VARIABLE
    python $pydir/clean_mocks.py --mockid $mockid --type $tracer
done