#!/bin/bash
# This script contains all the bash commands to run different results
# using the different flags defined in config/config.py
#
#
# __author__ = Louis Weyland
# __date__   = 6/05/22


# Run normal simulation
python3 main.py -sim-mart-vaq -read-data montagna_calls -attach-meth preferential -save

# Run normal simulation, each repetition on a new network
python3 main.py -entirely-sim-mart-vaq -read-data montagna_calls -attach-meth preferential -save

# Run the sensitivity analysis
nohup python3 main.py -read-data montagna_meetings -sa sim-mart-vaq -n-samples 1024  -r 3000 -output-value ratio_criminal -save &

# Run the phase diagram
python3 main.py -phase-diag -read-data montagna_calls -save

# Run a comparison analysis on the topology using different network configuration
# Thereby, for each indv simulation, the same network is used
python3 main.py -read-data -compare-simulations -r 800

# Run a comparison analysis on the topology using different network configruation
# Thereby, for each indv simulation a new network is created
python3 main.py -read-data -entirely-compare-simulations -r 800
