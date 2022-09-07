#!/bin/bash
# This script contains all the bash commands to run different results
# using the different flags defined in config/config.py
#
#
# __author__ = Louis Weyland
# __date__   = 6/05/22
set -e

declare -a arr=("preferential" "random" "small-world")
## Run a comparsion analysis on the characterisitcs of a network
#python3 new_main.py -read-data montagna_calls -get-network-stats -n-samples 50
#
#
## Run the whole pipeline
#nohup python3 new_main.py -read-data montagna_calls -whole-pipeline -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save -exec sequential &
#
#
## Run the simulation chunck vise
#for i in "${arr[@]}"
#do
#   echo "Doing $i"
#   for k in {0..5}
#   do
#      python3 new_main.py -read-data montagna_calls -sim-mart-vaq -case const -attach-meth $i -k 17 -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 10 -topo-meas -criminal-likelihood-corr -save
#   done
#done
#
#
#
#echo "Run sensitivity analysis"
#python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000 -n-samples 512  -exec parallel -output-value ratio_criminal -attach-meth random -k 2 -save
#
#cho "Run sensitvitiy analysis in chuncks"
for i in "${arr[@]}"
do
   echo "Doing $i"
   for k in {0..20}
   do
      python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000 -n-samples 512  -exec parallel -output-value ratio_criminal -attach-meth $1 -k 17 -running-chunk -save
   done
done
