#!/bin/bash

rm output_logs/*



bsub -J 'brenton_mse[1-50]' -W 1:00 -n 3 -e output_logs/brenton_mse%I.err -o output_logs/brenton_mse%I.out sh scripts/mse/brenton.sh

bsub -J 'brenton_row[1-50]' -W 1:00 -n 3 -e output_logs/brenton_row%I.err -o output_logs/brenton_row%I.out sh scripts/row/brenton.sh

bsub -J 'brenton_max[1-50]' -W 1:00 -n 3 -e output_logs/brenton_max%I.err -o output_logs/brenton_max%I.out sh scripts/max/brenton.sh