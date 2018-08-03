#!/bin/bash

s=0
fmin=0.6
fmax=1.4

temp_result=./fric_${fmin}_${fmax}_seed${s}/
python3 main_train.py --env MyHopper-v2 --num-timesteps 0 --num_iters 100 --batch_episodes 100 --fric_min $fmin --fric_max $fmax --outdir $temp_result
for((i=0;i<6;i++));do
    python3 main_train.py --env MyHopper-v2 --num-timesteps 0 --num_iters 120 --batch_episodes 100 --fric_min $fmin --fric_max $fmax --indir $temp_result --outdir ${temp_result}seed$i/ --trj_percentile 20 --seed $i
    python3 main_test.py --env MyHopper-v2 --fric_min $fmin --fric_max $fmax --indir ${temp_result}seed$i/ --outdir ${temp_result}seed$i/test/
done
