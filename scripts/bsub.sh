#!/bin/bash

rm output_logs/*


## brenton
bsub -J 'brenton_mse[1-50]' -W 2:00 -n 3 -e output_logs/brenton_mse%I.err -o output_logs/brenton_mse%I.out sh scripts/mse/brenton.sh

bsub -J 'brenton_row[1-50]' -W 2:00 -n 3 -e output_logs/brenton_row%I.err -o output_logs/brenton_row%I.out sh scripts/row/brenton.sh

bsub -J 'brenton_max[1-50]' -W 2:00 -n 3 -e output_logs/brenton_max%I.err -o output_logs/brenton_max%I.out sh scripts/max/brenton.sh

bsub -J 'brenton_kendall[1-50]' -W 2:00 -n 3 -e output_logs/brenton_kendall%I.err -o output_logs/brenton_kendall%I.out sh scripts/kendall/brenton.sh

bsub -J 'brenton_regret[1-50]' -W 2:00 -n 3 -e output_logs/brenton_regret%I.err -o output_logs/brenton_regret%I.out sh scripts/regret/brenton.sh


## marshall
bsub -J 'marshall_mse[1-50]' -W 3:00 -n 3 -e output_logs/marshall_mse%I.err -o output_logs/marshall_mse%I.out sh scripts/mse/marshall.sh

bsub -J 'marshall_row[1-50]' -W 3:00 -n 3 -e output_logs/marshall_row%I.err -o output_logs/marshall_row%I.out sh scripts/row/marshall.sh

bsub -J 'marshall_max[1-50]' -W 3:00 -n 3 -e output_logs/marshall_max%I.err -o output_logs/marshall_max%I.out sh scripts/max/marshall.sh

bsub -J 'marshall_kendall[1-50]' -W 3:00 -n 3 -e output_logs/marshall_kendall%I.err -o output_logs/marshall_kendall%I.out sh scripts/kendall/marshall.sh

bsub -J 'marshall_regret[1-50]' -W 3:00 -n 3 -e output_logs/marshall_regret%I.err -o output_logs/marshall_regret%I.out sh scripts/regret/marshall.sh


## momf

bsub -J 'momf_cluster[1-50]' -W 3:00 -n 4 -e output_logs/momf_cluster%I.err -o output_logs/momf_cluster%I.out sh scripts/momf/cluster.sh
