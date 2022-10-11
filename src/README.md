## Code

---
### First steps

Through the *make* command, various processes can be initialised. Firstly, the following command needs to be run in order to compile the Cython code

    make compile_cython

Next to this command, with *make* the following commands are possible

* make tests               (to run the tests)
* make test_notebooks     (to test the notebooks (not necessary))
* make docs_view          (generate the documentation)
* make profiling_sim_mart (profiles the simulations and returns .pstasts file)
* make clean              (deleted cache and other garbage files)

In this folder, all the code is included for reproducing the simulation. The following tree shows the structure of the code.

```
./
├── config/                            (contains the flag/args.parser)
├── data/                              (contains data about criminal networks)
├── network_utils/                     (contains all the code to analyse a social network)
├── research/                          (contains notebook for quick researches (garbage folder))
├── results/
│   ├── data/                          (contains the generated time-demanding results)
│   │   ├── network_stats/             (contains the results about the populations' characteristics)
│   │   ├── sensitivity_analysis/      (contains the sensitivity analysis results)
│   │   └── sim_mart_vaq/              (contains the results of the simulation including topological measurements)
│   ├── figures/                       (contains all the generated & saved figures)
│   ├── notebooks/                     (contains the results shown in a notebook)
│   └── video/                         (contains the visualisation of the populations as gifs)
├── simulators/                        (contains the code for the simulation as well as creating the populations)
├── test/                              (contains the testing framework using pytest)
└── utils/                             (contains all the utility functions)
.main.py                               (contains all the helper functions to run the different analysis/simulations)
```

---
# Running commands
The different results/simulations are run through [main.py](main.py) using various commands as shown below. Thereby, it is important to run all the commands. In case a programme takes too long, it is possible to run it in the background using the [bash.script](bash_script.sh)  with the following command

    nohup ./bash_script.sh &


### Run a comparison analysis on the characteristics of a network
To get an overview of the different characteristics of the populations, the following command computes the mean of various social network analysis concepts and saves them in the folder [network_stats](results/data/network_stats/). The network changes depending on the ratio of honest/wolf but also depending on k, the number of links the added nodes have.

    python3 main.py -get-network-stats -read-data montagna_calls -ratio-honest 0.99 -ratio-wolf 0.001 -k 80 -n-samples 10 -save


### Run the whole pipeline
This function will run the whole pipeline, create the different populations and collect the data along the simulation before plotting the results (Takes a lot of time!!!)

    python3 main.py -whole-pipeline -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save -exec sequential

### Run the simulation in chunks
To overcome the aforementioned constraints, the simulation can be run in chunks. Thereby, one needs to specify which population structure needs to be simulated. The results are appended in the folder [sim_mart_vaq](results/data/sim_mart_vaq/).

    python3 main.py -sim-mart-vaq -read-data montagna_calls  -case const -attach-meth preferential -k 17 -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1 -r 250000 -n-samples 30 -topo-meas -criminal-likelihood-corr -save -exec parallel &

### Run sensitivity analysis
To run the sensitivity analysis, the following command can be used.
The resulting table is saved in [sensitivity_analysis](results/data/sensitivity_analysis/)

    python3 main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 2500 -k 2 -n-samples 32  -exec parallel -output-value ratio_criminal -attach-meth random -save

### Run sensitivity analysis in chunks
Since the sensitivity analysis takes up some time, it is possible to run it chunkwise. The results are saved in [sensitivity_analysis](results/data/sensitivity_analysis/).

    python3 main.py --sensitivity-analysis -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000 -n-samples 512  -exec parallel -output-value ratio_criminal -attach-meth preferential -k 2 -running-chunk -save

### Run sensitivity analysis on the number of links
To analyse the impact of paramters *k* on the simulation, an analysis is conducted computing on various *k*. The results are saved in [sensitivity_links](results/data/sensitivity_links)

    python3 main.py --sensitivity-analysis-links -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -n-groups 1  -r 250000  -exec parallel -output-value ratio_criminal

### Run the phase-diagramms
The following command runs the phase diagram of different parameters defined in main.py. The results are saved in [phase_diag](results/data/phase_diag).

    python3 main.py --phase-diagram -read-data montagna_calls -ratio-honest 0.96 -ratio-wolf 0.01 -k 17 -n-groups 1  -r 250000 -attach-meth random  -exec parallel

### Profiling the code

To profile the code, the cprofile package was used together with the gprof package to visualise the critical part. To conduct such an analysis, please use the following commands

    $ python3 -m cProfile -o output.pstats main.py -read-data montagna_calls -sim-mart-vaq
    $ gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
    $ python3 -c"import pstats;p = pstats.Stats('output.pstats');p.sort_stats('cumulative').print_stats('simulators')"

---
# Tips and Tricks

The above commands are specifically tailored to the analysis reported in the Master Thesis. However, if one wishes to use the simulation only for one's own purposes, the following commands can be used:

When dealing with a huge criminal network, building a population may take some time. As a tip, it is possible to build n populations and store them. These networks can be quickly loaded later for simulation. The following command can be run in a bash script and create 10 different populations of each seizure method:

```bash
    #!/bin/bash
    declare -a arr_structure=("preferential" "random" "small-world")
    declare -a arr_k=(74 74 74)
    for i in "${!arr_structure[@]}"
    do
       echo "Doing ${arr_n_links[i]}"
       for k in {0..5}
       do
          python3 new_main.py -read-data montagna_calls --create-population -ratio-honest 0.99 -ratio-wolf 0.001 -n-sample 10 -attach-meth ${arr_structure[i]} -k ${arr_n_links[i]} -exec parallel
      done
    done
```

If one wished to read own network, create a population and simulate it, the following python script will do so:

```python
    # Init meta_simulator with preferential attachment
    meta_sim = MetaSimulator(
    network_name=network_name,
    ratio_honest=0.3,
    ratio_wolf=0.3,
    k=6,
    attachment_method="preferential")

    # Get overview/characterisitics of the population
    # Get overview of the new network
    complete_network_stats = NetworkStats(
    NetworkConverter.gt_to_nk(simulators_pref.network))
    complete_network_stats_pref.get_overview()


    # The simulator is called within the meta_simulator
    data_collector = meta_sim.avg_play(
    network=meta_sim.network,
    rounds=20000,
    n_groups=1,
    repetition=5,
    ith_collect=20,
    collect_fitness=True)

    # Plot the evolution ratio
    plotter.plot_lines(
    dict_data={"preferential": data_collector},
    y_data_to_plot=["mean_ratio_honest", "mean_ratio_wolf", "mean_ratio_criminal"],
    x_data_to_plot="mean_iteration",
    title="Testing the simulation",
    xlabel="rounds",
    ylabel="ratio",
    plot_std="True")
```

### ***To study the actual code of the different functions and classes, please consult the documentation in [docs](docs/) by opening the html files in your browser. In google chrome this can be done by giving the abolute path of the .html file as a search link.***
