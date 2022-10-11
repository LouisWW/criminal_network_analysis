#!/bin/bash
# This script contains all the bash commands to run different results
# using the different flags defined in config/config.py
#
# It is possible that some commands here need to be updated!!
#
# __author__ = Louis Weyland
# __date__   = 6/05/22
set -e

declare -a arr_structure=("preferential" "random" "small-world")
declare -a arr_k=(74 74 74)
declare -a arr_n_links=(20 35 50 65 80)

###################################################################################################
## Run a comparsion analysis on the characterisitcs of a network
#python3 new_main.py -read-data montagna_calls -get-network-stats -n-samples 50
#
#
## Run the whole pipeline
#nohup python3 new_main.py -read-data montagna_calls -whole-pipeline -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save -exec sequential &
#
#


###################################################################################################
# Create the populations
#for i in "${!arr_structure[@]}"
#do
#   echo "Doing ${arr_structure[i]}"
#   for k in {0..5}
#   do
#      python3 new_main.py -read-data montagna_calls --create-population -ratio-honest 0.99 -ratio-wolf 0.001 -n-sample 10 -attach-meth ${arr_structure[i]} -k ${arr_k[i]} -exec parallel
#   done
#done
#

###################################################################################################
## Create the populations for the different links
#for i in "${!arr_structure[@]}"
#do
#   echo "Doing ${arr_n_links[i]}"
#   for k in {0..5}
#   do
#      python3 new_main.py -read-data montagna_calls --create-population -ratio-honest 0.99 -ratio-wolf 0.001 -n-sample 10 -attach-meth ${arr_structure[i]} -k ${arr_n_links[i]} -exec parallel
#   done
#done


###################################################################################################
## Run the simulation chunck vise
#for i in "${!arr_structure[@]}"
#do
#   echo "Doing ${arr_structure[i]}"
#   for k in {0..50}
#   do
#      python3 new_main.py -read-data montagna_calls -sim-mart-vaq -case growth -attach-meth ${arr_structure[i]} -k ${arr_k[i]}\
#      -ratio-honest 0.99 -ratio-wolf 0.001 -n-groups 1 -r 1000000 -n-samples 1 -topo-meas -criminal-likelihood-corr -exec sequential -save
#   done
#done

###################################################################################################

#echo "Doing link sensitivity analysis"
#python3 new_main.py --sensitivity-analysis-links -case growth -read-data montagna_calls -ratio-honest 0.99 -ratio-wolf 0.001 -n-groups 1 -n-sample 30 -r 1000000  -exec parallel -output-value ratio_criminal
#
###################################################################################################

## Running the phase diag
#echo "Running phase diag analysis"
#python3 new_main.py --phase-diagram -read-data montagna_calls -ratio-honest 0.99 -ratio-wolf 0.001 -n-groups 1  -r 1000000 -n-sample 30  -exec parallel -save

###################################################################################################

#echo "Run sensitivity analysis"
#python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000 -n-samples 512  -exec parallel -output-value ratio_criminal -attach-meth random -k 2 -save
#
#echo "Run sensitvitiy analysis in chuncks"
#for i in "${!arr_structure[@]}"
#do
#   echo "Doing ${arr_structure[i]}"
#   for k in {0..1500}
#   do
#      python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.99 -ratio-wolf 0.001 -n-groups 1\
#      -r 1000000 -n-samples 4096  -exec parallel -output-value ratio_criminal -attach-meth ${arr_structure[i]} -k ${arr_k[i]}\
#      -running-chunk -save
#done
