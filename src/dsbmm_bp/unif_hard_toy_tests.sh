#!/bin/bash

# allow parallel execution by terminating line w &
# using "\j", @P for num current processes requires
# bash >= 4.4
# If wanted multiple arguments passed in file, then can
# expand with e.g. ${args[@]}
num_procs=10 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running
for proc_num in 0 1 2 3 4
do
  while ((${num_jobs@P}>=$num_procs)); do
    wait -n
  done
  python toy_apply.py --name_ext "_hardunif${proc_num}" --params_path "/scratch/fitzgeraldj/data/toy_sims/toy_param_grid_hard${proc_num}.pkl" --msg_init_mode uniform &
done 
exit 0
