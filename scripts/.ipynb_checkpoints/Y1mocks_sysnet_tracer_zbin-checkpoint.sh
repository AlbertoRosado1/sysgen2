#!/bin/bash
# Y1mocks_sysnet_tracer_zbin.sh $tracer $zw $mockid
# Y1mocks_sysnet_tracer_zbin.sh ELG_LOP_complete_gtlimaging 0.8_1.1 0

$HOME/sysgen2/scripts/run_sysnet_mocks.sh N $1$2 true false 1024 0.003 dnnp pnll $3
$HOME/sysgen2/scripts/run_sysnet_mocks.sh S $1$2 true false 1024 0.003 dnnp pnll $3 

$HOME/sysgen2/scripts/run_sysnet_mocks.sh N $1$2 false true 1024 0.004 dnnp pnll $3
$HOME/sysgen2/scripts/run_sysnet_mocks.sh S $1$2 false true 1024 0.004 dnnp pnll $3 