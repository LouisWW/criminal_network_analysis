#!/bin/bash
# This script contains all the bash commands to run different results
# using the different flags defined in config/config.py
#
#
# __author__ = Louis Weyland
# __date__   = 6/05/22
set -e

declare -a arr_structure=("preferential" "random" "small-world")
declare -a arr_k=(17 17 35)
## Run a comparsion analysis on the characterisitcs of a network
#python3 new_main.py -read-data montagna_calls -get-network-stats -n-samples 50
#
#
## Run the whole pipeline
#nohup python3 new_main.py -read-data montagna_calls -whole-pipeline -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save -exec sequential &
#
#
## Run the simulation chunck vise
for i in "${!arr_structure[@]}"
do
   echo "Doing ${arr_structure[i]}"
   for k in {0..5}
   do
      python3 new_main.py -read-data montagna_calls -sim-mart-vaq -case growth -attach-meth ${arr_structure[i]} -k ${arr_k[i]}\
      -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 350000 -n-samples 10 -topo-meas -criminal-likelihood-corr -save
   done
done
#
#
#
#echo "Run sensitivity analysis"
#python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000 -n-samples 512  -exec parallel -output-value ratio_criminal -attach-meth random -k 2 -save
#
echo "Run sensitvitiy analysis in chuncks"
for i in "${!arr_structure[@]}"
do
   echo "Doing ${arr_structure[i]}"
   for k in {0..20}
   do
      python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1\
      -r 250000 -n-samples 512  -exec parallel -output-value ratio_criminal -attach-meth ${arr_structure[i]} -k ${arr_k[i]}\
      -running-chunk -save
   done
done
