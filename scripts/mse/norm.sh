#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate py39
module load gcc/11.2.0

seed=$(($LSB_JOBINDEX + 100))

python src/simul_mf.py --seed $seed --obj "mse" --mode "norm"
