#!/bin/bash
# This script contains all the bash commands to run different results
# using the different flags defined in config/config.py
#
#
# __author__ = Louis Weyland
# __date__   = 6/05/22




# Run the sensitivity analysis
echo Hello
python3 main.py -read-data montagna_meetings -sa sim-mart-vaq -n-samples 10 -output-value ratio_criminal -save-results
