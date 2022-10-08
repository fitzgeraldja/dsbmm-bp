#!/bin/bash

# allow parallel execution by terminating line w &
# using "\j", @P for num current processes requires
# bash >= 4.4
# If wanted multiple arguments passed in file, then can
# expand with e.g. ${args[@]}
params='toy_proc.txt'
num_procs=10 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running
for proc in 0 1 2 3 4 5 6 7 8 9
do
  while ((${num_jobs@P}>=$num_procs)); do
    wait -n
  done
  python toy_apply.py --name_ext "_unif${proc}" --params_path "/scratch/fitzgeraldj/data/toy_sims/toy_param_grid_${proc}.pkl" --msg_init_mode uniform &
done
exit 0
