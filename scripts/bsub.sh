#!/bin/bash

rm output_logs/*

#####################################
#### MSE distance ######
#####################################

bsub -J 'norm_mse[1-50]' -W 1:00 -n 3 -e output_logs/norm_mse%I.err -o output_logs/norm_mse%I.out sh scripts/mse/norm.sh