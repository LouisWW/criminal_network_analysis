#!/bin/bash
# This script contains all the bash commands to run different results
# using the different flags defined in config/config.py
#
#
# __author__ = Louis Weyland
# __date__   = 6/05/22


# Run normal simulation
python3 main.py -sim-mart-vaq -read-data montagna_calls -save

# Run the sensitivity analysis
nohup python3 main.py -read-data montagna_meetings -sa sim-mart-vaq -n-samples 1024  -r 3000 -output-value ratio_criminal -save &

# Run the phase diagram
python3 main.py -phase-diag -read-data montagna_calls -save
