#!/bin/bash
# This script contains all the bash commands to run different results
# using the different flags defined in config/config.py
#
#
# __author__ = Louis Weyland
# __date__   = 6/05/22


# Run normal simulation
python3 main.py -sim-mart-vaq -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -attach-meth random -r 10 -n-samples 20 -save

# Run normal simulation, each repetition on a new network
python3 main.py -sim-mart-vaq-w-net -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -attach-meth random -r 100 -n-samples 20 -save

# Run an anlysis on the criminal likelihood
nohup python3 main.py -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -criminal-likelihood-corr -r 400 -n-samples 10 -save &

# Run the sensitivity analysis
nohup python3 main.py -read-data montagna_meetings -sa sim-mart-vaq -n-samples 1024  -r 3000 -output-value ratio_criminal -save &

# Run the phase diagram
python3 main.py -phase-diag -read-data montagna_calls -save

# Run a comparison analysis on the topology using different network configuration
# Thereby, for each indv simulation, the same network is used
python3 main.py -read-data -topo-meas -r 800

# Run a comparison analysis on the topology using different network configruation
# Thereby, for each indv simulation a new network is created
python3 main.py -read-data montagna_calls -topo-meas-w-net -r 800


# Run a comparsion analysis on the characterisitcs of a network
python3 new_main.py -read-data montagna_calls -get-network-stats -n-samples 50


# Run the whole pipeline
nohup python3 new_main.py -read-data montagna_calls -whole-pipeline -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save -exec sequential &

# Run the simulation chunck vise
nohup python3 new_main.py -read-data montagna_calls -sim-mart-vaq -case cnst -attach-meth preferential -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save &
