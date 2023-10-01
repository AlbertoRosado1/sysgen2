#!/bin/bash
# bash run_contaminate.sh ELG_LOP_complete_gtlimaging v0.6 15

source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

tracer=$1 #ELG_LOP_complete_gtlimaging
version=$2 # survey version to get SYSNet weights and hpmaps from
nran=$3 # number of randoms to use 
pydir=$HOME/sysgen2/py

start_idx=0
end_idx=0

for (( VARIABLE=$start_idx; VARIABLE<=$end_idx; VARIABLE++ ))
do
    mockid=$VARIABLE
    python $pydir/forward_mocks_all.py --type $tracer --survey_version $version --nran $nran --mockid $mockid
    python $pydir/cont_mocks.py --mockid $mockid --type $tracer
done