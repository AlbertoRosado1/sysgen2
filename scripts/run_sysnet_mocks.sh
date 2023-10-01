#!/bin/bash

# command line args 
run=$1 # north or south
tracer=$2

do_LRfinder=$3 #false #for running the learning rate finder
do_nnrun=$4 # true #false

batchsize=$5
learnrate=$6
model=$7
loss=$8
mockid=${9} # mockid

# below is code for reading data and where output is
output_nn=/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/mock$mockid/sysnet_clean/${tracer}_${run}
input_data=/pscratch/sd/a/arosado/SecondGenMocks/AbacusSummit/mock$mockid/sysnet_clean/prep_${tracer}_${run}.fits
echo using $input_data and saving to $output_nn

# nn parameters
axes="all"
nchain=5
nepoch=200
nns=(4 20)
bsize=$batchsize #5000
lr=$learnrate
model=$model #dnnp
loss=$loss #pnll
etamin=0.00001
    
if [ "${do_LRfinder}" = true ]
then
   du -h $input_data
   sysnet-app -i ${input_data} -o ${output_nn} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -fl
fi

if [ "${do_nnrun}" = true ]
then
   du -h $input_data
   echo using lr ${lr[${ph}]}
   sysnet-app -i ${input_data} -o ${output_nn} -ax ${axes[@]} -bs ${bsize} --model $model --loss $loss --nn_structure ${nns[@]} -lr ${lr[${ph}]} --eta_min $etamin -ne $nepoch -nc $nchain -k 
fi