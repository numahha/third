#!/bin/bash

python3 run_mujoco.py --env Hopper-v2 --outdir ./result/ --num-timesteps 200000
python3 plot.py --input result/progress.csv --output curve.png

python3 run_mujoco.py --env Hopper-v2 --outdir ./result1/ --num-timesteps 100000
python3 plot.py --input result1/progress.csv --output curve1.png

python3 run_mujoco.py --env Hopper-v2 --outdir ./result2/ --indir ./result1/ --num-timesteps 100000
python3 plot.py --input result2/progress.csv --output curve2.png
