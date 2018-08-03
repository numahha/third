#!/bin/bash

find ./ -name '*progress.csv' -exec python3 plot.py --input {} --output {}.png \;
