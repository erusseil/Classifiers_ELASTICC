#!/bin/bash

nb_cores=5
job_per_core=1


for i in $(seq 0 $((nb_cores-1)))
do
   nohup python start_AL.py $i $job_per_core  > nohup/nohup_AL_core$i.out 2>&1 &
   echo "Core $((i+1)) / $nb_cores started"
done
