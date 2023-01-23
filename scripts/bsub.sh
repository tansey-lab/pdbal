#!/bin/bash

rm output_logs/*

#####################################
#### MSE distance ######
#####################################

bsub -J 'norm_mse[1-50]' -W 1:00 -n 3 -e output_logs/norm_mse%I.err -o output_logs/norm_mse%I.out sh scripts/mse/norm.sh

bsub -J 'norm_row[1-50]' -W 1:00 -n 3 -e output_logs/norm_row%I.err -o output_logs/norm_row%I.out sh scripts/row/norm.sh

bsub -J 'norm_max[1-50]' -W 1:00 -n 3 -e output_logs/norm_max%I.err -o output_logs/norm_max%I.out sh scripts/max/norm.sh