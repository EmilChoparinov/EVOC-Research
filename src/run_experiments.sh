#!/bin/bash
rm -f most-fit-xy-run-*.csv

# Bash script to automate multiple runs with different alpha and fitness function combinations
# chmod +x run_experiments.sh in terminal
# ./run_experiments.sh in terminal
# Define the combinations of alpha values and fitness functions
for alpha in 1 0.5 0
do
  for fit in "distance" "similarity" "blended"
  do
    echo "Running with alpha=$alpha and fitness function=$fit"
    # python3 run.py --alpha $alpha --with-fit $fit # too much time
    python3 run.py --alpha $alpha --with-fit $fit --gens 1 --runs 5
  done
done