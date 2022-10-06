The [config.py](config.py) file captures all the flags given to main.py. Below, you can find the list of all possible flags can be used in different combinations.

````
usage: CRIMINAL NETWORK ANALYSIS [-h] [-draw-network [{c,n}]] [-execute {parallel,sequential}] [-create-population] [-get-network-stats] [-compare-w-rnd-init]
                                 [-read-data [{cunha,montagna_calls,montagna_meetings}]] [-attach-meth [{preferential,random,small-world}]] [-k K]
                                 [-animate-attachment-process] [-sim-mart-vaq] [-case {const,growth,decline}] [-whole-pipeline] [-criminal-likelihood-corr]
                                 [-sim-mart-vaq-w-net] [-animate-simulation [{unfiltered,filtered}]] [-sa [{sim-mart-vaq}]] [--sensitivity-analysis-links] [-phase-diag]
                                 [-topo-meas] [-topo-meas-w-net] [-n-samples N_SAMPLES] [-output-value OUTPUT_VALUE] [-save] [-plot] [-r ROUNDS] [-n-groups N_GROUPS]
                                 [-ratio-honest RATIO_HONEST] [-ratio-wolf RATIO_WOLF] [-delta DELTA] [-tau TAU] [-gamma GAMMA] [-beta-s BETA_S] [-beta-h BETA_H]
                                 [-beta-c BETA_C] [-c-w C_W] [-c-c C_C] [-r-w R_W] [-r-c R_C] [-r-h R_H] [-temp TEMPERATURE] [-mutation-prob MUTATION_PROB] [-verbose]
                                 [-running-chunk]

Flags needed to run a given Pipeline and generate the desired results

optional arguments:
  -h, --help            show this help message and exit
  -draw-network [{c,n}]
                        Define if the network should be visualized. c = circular network, n = normal/random
  -execute {parallel,sequential}
                        Run the simulation in parallel or sequential.
  -create-population    Create and save the population (-save flag not needed)
  -get-network-stats    Returns the mean characteristics of a population (preferential/random/small-world)
  -compare-w-rnd-init   Compare the outcome of the simulations if a random init was used or not.
  -read-data [{cunha,montagna_calls,montagna_meetings}]
                        Define which network to read; cunha, montagna_meetings, montagna_calls.
  -attach-meth [{preferential,random,small-world}]
                        Define the attachment methods around the criminal network
  -k K                  Define how many new connection a new node is making while generating a population.
  -animate-attachment-process
                        Create an animation of the attachment process.
  -sim-mart-vaq         Define if the simulation based on Martinez-Vaquero is run. Thereby, for each repetition the another network is used.
  -case {const,growth,decline}
                        Define which case is simulated
  -whole-pipeline       Run the whole simulation for the different structures in one go. Caution: Might be unstable!!!
  -criminal-likelihood-corr
                        Define if a correlation between criminal and node centrality exists.
  -sim-mart-vaq-w-net   Define if the simulation based on Martinez-Vaquero is run. Thereby, for each repetition an new network is created.
  -animate-simulation [{unfiltered,filtered}]
                        Create an animation of the simulation.
  -sa [{sim-mart-vaq}], --sensitivity-analysis [{sim-mart-vaq}]
                        Define to run a sensitivity analysis on one of the choices.
  --sensitivity-analysis-links
                        Define to run a sensitivity analysis on one of the choices.
  -phase-diag, --phase-diagram
                        Create a phase diagram with the defined parameters.
  -topo-meas            Define to run a comparative analysis of the different simulations. Thereby, for each repetition the same network is used.
  -topo-meas-w-net      Define to run a comparative analysis of the different simulations. Thereby, for each repetition an new network is created.
  -n-samples N_SAMPLES  Define the sampling number for the saltelli method.(default: 15)
  -output-value OUTPUT_VALUE
                        Define on which output value to focus for the sensitivity analysis.
  -save                 Define if the results should be saved.
  -plot                 Define if the results should be plotted.ss
  -r ROUNDS, --rounds ROUNDS
                        Define the numbers of rounds played. Can be applied to SimMartVaq.play and SensitivityAnalyser.sim_mart_vaq_sa
  -n-groups N_GROUPS    Define the number of groups for each round.
  -ratio-honest RATIO_HONEST
                        Define the initial ratio of honest in a population.
  -ratio-wolf RATIO_WOLF
                        Define the initial ratio of wolves in a population.
  -delta DELTA          Define the influence of criminals on the acting of the wolf (SimMartVaq)
  -tau TAU              Influence of wolf's action on criminals (SimMartVaq)
  -gamma GAMMA          Punishment ratio for the members of a criminal organization (SimMartVaq)
  -beta-s BETA_S        State punishment value (SimMartVaq)
  -beta-h BETA_H        Civil punishment value (SimMartVaq)
  -beta-c BETA_C        Criminal punishment value (SimMartVaq)
  -c-w C_W              Damage caused by wolf (SimMartVaq)
  -c-c C_C              Damage caused by criminal (SimMartVaq)
  -r-w R_W              Reward ratio for wolf (SimMartVaq)
  -r-c R_C              Reward ratio for criminal (SimMartVaq)
  -r-h R_H              Reward ratio for honest (SimMartVaq)
  -temp TEMPERATURE, --temperature TEMPERATURE
                        Temperature for the fermi function (SimMartVaq)
  -mutation-prob MUTATION_PROB
                        Mutation probability (SimMartVaq)
  -verbose              Print extra info
  -running-chunk        Running the analysis in chunk by saving in between
