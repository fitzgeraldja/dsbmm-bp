#!/bin/bash

# allow parallel execution by terminating line w &
# using "\j", @P for num current processes requires
# bash >= 4.4
# If wanted multiple arguments passed in file, then can
# expand with e.g. ${args[@]}
params='toy_proc.txt'
num_procs=10 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running
while read -r proc
do
  while ((${num_jobs@P}>=$num_procs)); do
    wait -n
  done
  python toy_apply.py --name_ext ${proc} --params_path "/scratch/fitzgeraldj/data/toy_sims/toy_param_grid${proc}.pkl" &
done < $params
exit 0
