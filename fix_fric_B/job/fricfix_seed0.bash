#!/bin/bash

array=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0)

s=0

for i in ${array[@]}; do
        python3 run_mujoco.py --env MyHopper-v1 --outdir fricfix${i}_seed${s} --num-timesteps 2000000 --fric_fix $i --seed $s
done






#python3 train_trpo_gym.py --env Hopper-v2 --gpu -1
