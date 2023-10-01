#!/bin/bash
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J Y1mocks_sysnet
#SBATCH -t 3:00:00
#SBATCH -L SCRATCH
#SBATCH -o slurm_outputs/%A_%a.out
#SBATCH -e slurm_outputs/%A_%a.err
#SBATCH --array=0-24

echo $SLURM_ARRAY_TASK_ID

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main 

tracer=ELG_LOP_complete_gtlimaging # IMPORTANT! if doing LRG change -N 2 -> -N 3
mockid=$SLURM_ARRAY_TASK_ID
SUBSTRING=${tracer:0:3}
echo $SUBSTRING

if [ $SUBSTRING == 'ELG' ]
then
    zbin1=0.8_1.1
    zbin2=1.1_1.6
    srun -N 1 -n 1 ./Y1mocks_sysnet_tracer_zbin.sh $tracer $zbin1 $mockid &
    srun -N 1 -n 1 ./Y1mocks_sysnet_tracer_zbin.sh $tracer $zbin2 $mockid &
    wait
fi

if [ $SUBSTRING == 'LRG' ]
then
    zbin1=0.4_0.6
    zbin2=0.6_0.8
    zbin3=0.8_1.1
    srun -N 1 -n 1 ./Y1mocks_sysnet_tracer_zbin.sh $tracer $zbin1 $mockid &
    srun -N 1 -n 1 ./Y1mocks_sysnet_tracer_zbin.sh $tracer $zbin2 $mockid &
    srun -N 1 -n 1 ./Y1mocks_sysnet_tracer_zbin.sh $tracer $zbin3 $mockid &
    wait
fi

if [ $SUBSTRING == 'QSO' ]
then
    zbin1=0.8_1.3
    zbin2=1.3_2.1
    srun -N 1 -n 1 ./Y1mocks_sysnet_tracer_zbin.sh $tracer $zbin1 $mockid &
    srun -N 1 -n 1 ./Y1mocks_sysnet_tracer_zbin.sh $tracer $zbin2 $mockid &
    wait
fi