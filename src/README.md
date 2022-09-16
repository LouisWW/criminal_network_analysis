Explain src folder



---
# Running commands
### Run a comparsion analysis on the characterisitcs of a network
    python3 new_main.py -get-network-stats -read-data montagna_calls -ratio-honest 0.99 -ratio-wolf 0.001 -n-samples 10 -save


### Run the whole pipeline
    nohup python3 new_main.py -read-data montagna_calls -whole-pipeline -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save -exec sequential &


### Run the simulation chunck vise
    nohup python3 new_main.py -read-data montagna_calls -sim-mart-vaq -case const -attach-meth preferential -k 17 -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save  -exec parallel &

### Run sensitivity analysis
    python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 2500 -n-samples 32  -exec parallel -output-value ratio_criminal -attach-meth random -k 2 -save

### Run sensitvitiy analysis in chuncks
    python3 new_main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000 -n-samples 512  -exec parallel -output-value ratio_criminal -attach-meth preferential -k 2 -running-chunk -save

## Run sensitivity analysis on the number of links
    python3 new_main.py --sensitivity-analysis-links -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000  -exec parallel -output-value ratio_criminal

## Run the phase-diagramms
    python3 new_main.py --phase-diagram -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -k17 -n-groups 1  -r 250000 -attach-meth random  -exec parallel
    
## Profiling the code

To profile the code, the cprofile package was used together with the gprof package to visualise the critical part. To conduct such an analysis, please use the following commands

    $ python3 -m cProfile -o output.pstats main.py -read-data montagna_calls -sim-mart-vaq
    $ gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
    $ python3 -c"import pstats;p = pstats.Stats('output.pstats');p.sort_stats('cumulative').print_stats('simulators')"
