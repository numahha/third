#!/bin/bash

python3 run_mujoco.py --env Hopper-v2 --outdir ./result_batchepi/ --num-timesteps 0 --num_iters 100 --batch_episodes 32
python3 plot.py --input result_batchepi/progress.csv --output curve_batchepi.png
python3 run_mujoco.py --env Hopper-v2 --outdir ./result_batchstep/ --num-timesteps 1000000 --batch_steps 5000
python3 plot.py --input result_batchstep/progress.csv --output curve_batchstep.png
