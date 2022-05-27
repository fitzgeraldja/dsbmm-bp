#!/bin/bash

# allow parallel execution by terminating line w &
# using "\j", @P for num current processes requires
# bash >= 4.4
# If wanted multiple arguments passed in file, then can
# expand with e.g. ${args[@]}
params='alignment_params.txt'
num_procs=20 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running
while read -r tun_par
do
  while ((${num_jobs@P}>=$num_procs)); do
    wait -n
  done
  python apply.py --test align --tuning_param ${tun_par} &
done < $params
exit 0
