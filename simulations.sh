#!/bin/bash

## To run all the experiments in the paper, set N=250 (this may take some time)
N=2

models='linreg logreg poisson beta'
objectives='first max kendall euclidean influence'

for (( i=1; i<=$N; i++ ))
do
    seed=$(($i + 100))
    for model in $models
    do
        for objective in $objectives
        do
            python src/simul.py --seed $seed --obj $objective --model $model
        done
    done
done

python src/plot.py