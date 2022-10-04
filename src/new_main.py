"""
This script contains regroups all the functions and pipelines needed to recreate the results.

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
import json
import logging
import os
import re
from copy import deepcopy

import numpy as np
import pandas as pd
from config.config import ConfigParser
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from network_utils.network_stats import NetworkStats
from scipy import stats
from simulators.meta_simulator import MetaSimulator
from simulators.sim_mart_vaq import SimMartVaq
from utils.animation import Animateur
from utils.numpy_arrayy_encoder import NumpyArrayEncoder
from utils.plotter import Plotter
from utils.sensitivity_analysis import SensitivityAnalyser
from utils.stats import compare_time_series
from utils.stats import concat_df
from utils.stats import dict_mean
from utils.stats import get_mean_std_over_list
from utils.tools import DirectoryFinder
from utils.tools import timestamp

# Catch the flags
args = ConfigParser().args
plotter = Plotter()


# Define logger output
logger = logging.getLogger("logger")
logger_handler = logging.StreamHandler()  # Handler for the logger
logger_handler.setFormatter(
    logging.Formatter(
        "[%(levelname)s]\t %(message)-100s ---- (%(asctime)s.%(msecs)03d) %(filename)s",
        datefmt="%H:%M:%S",
    )
)
logger.addHandler(logger_handler)
logger.propagate = False

if args.verbose:
    logger.setLevel(logging.INFO)


if args.create_population:
    """Creating population and save them."""
    meta_sim = MetaSimulator(
        network_name=args.read_data,
        attachment_method=args.attach_meth,
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        k=args.k,
        random_fit_init=False,
    )

    meta_sim.create_network_and_save(args.n_samples)

if args.sim_mart_vaq:
    """Simulate the simulation form
    Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
    Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.

    Runs only one structure at the time
    """
    # Add nodes to network
    # First convert to gt
    meta_sim = MetaSimulator(
        network_name=args.read_data,
        attachment_method=args.attach_meth,
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        k=args.k,
        random_fit_init=False,
    )

    # Get overview of the new network
    complete_network_stats = NetworkStats(NetworkConverter.gt_to_nk(meta_sim.network))
    complete_network_stats.get_overview()
    logger.info("Doing preferential simulation")

    data_collector = meta_sim.avg_play(
        rounds=args.rounds,
        n_groups=args.n_groups,
        repetition=args.n_samples,
        ith_collect=int(args.rounds / 15),
        measure_topology=args.topo_meas,
        measure_likelihood_corr=args.criminal_likelihood_corr,
        execute=args.execute,
        show_no_bar=False,
        collect_fitness=False,
    )
    logger.info("Done")

    if args.save:

        file_name = (
            DirectoryFinder().result_dir_data_sim_mart_vaq
            + args.case
            + "_"
            + args.attach_meth
            + ".json"
        )

        # check if df is in dict and convert it to json
        for run in data_collector.keys():
            if "df" in data_collector[run]:
                data_collector[run]["df"] = data_collector[run]["df"].to_json()

        if os.path.isfile(file_name) and os.access(file_name, os.R_OK):
            with open(file_name) as fp:
                previous_results = json.load(fp)

            # get the number of runs already done
            n_prev_runs = max(int(key) for key in previous_results.keys())
            data_collector
            simple_runs = {
                str(int(k) + 1 + n_prev_runs): v
                for k, v in data_collector.items()
                if k.isdigit()
            }
            new_results = {**previous_results, **simple_runs}
            with open(
                file_name,
                "w",
            ) as fp:
                json.dump(new_results, fp, indent=4)

        elif not os.path.isfile(file_name):
            # Get the simple runs
            simple_runs = {k: v for k, v in data_collector.items() if k.isdigit()}
            with open(
                file_name,
                "w",
            ) as fp:
                json.dump(simple_runs, fp, indent=4)

    if args.plot:
        ax_0 = plotter.plot_lines(
            dict_data={args.attach_meth: data_collector},
            y_data_to_plot=[
                "mean_ratio_honest",
                "mean_ratio_wolf",
                "mean_ratio_criminal",
            ],
            x_data_to_plot="mean_iteration",
            xlabel="Rounds",
            ylabel="Ratio (%)",
            plot_deviation="std",
        )

elif args.plot:
    """Plot the results collected over the different runs.

    Important to specify the cases
    """
    results_dir = DirectoryFinder().result_dir_data_sim_mart_vaq + args.case
    preferential_file = results_dir + "_preferential.json"
    random_file = results_dir + "_random.json"
    sw_file = results_dir + "_small-world.json"

    whole_data = {}
    for structure, file_name in [
        ("preferential", preferential_file),
        ("random", random_file),
        ("small-world", sw_file),
    ]:
        with open(file_name) as fp:
            whole_data[structure] = json.load(fp)

    # since df is saved as json, need to convert it back
    for structure in whole_data.keys():
        for key in whole_data[structure].keys():
            whole_data[structure][key]["df"] = pd.read_json(
                whole_data[structure][key]["df"]
            )

    # Get the mean of the data
    for structure, data in whole_data.items():
        whole_data[structure] = get_mean_std_over_list(data)
        whole_data[structure] = concat_df(whole_data[structure], 100000)

    compare_time_series(whole_data)

    # Ready to be plotted
    ax_0 = plotter.plot_lines(
        dict_data=whole_data,
        y_data_to_plot=["mean_ratio_honest", "mean_ratio_wolf", "mean_ratio_criminal"],
        x_data_to_plot="mean_iteration",
        title=True,
        xlabel="Rounds",
        ylabel="Ratio (%)",
        plot_deviation="sem",
        square_plot=True,
    )

    ax_1 = plotter.plot_lines(
        dict_data={
            " ": {
                "mean_preferential": whole_data["preferential"]["mean_ratio_criminal"],
                "sem_preferential": whole_data["preferential"]["sem_ratio_criminal"],
                "std_preferential": whole_data["preferential"]["std_ratio_criminal"],
                "mean_random": whole_data["random"]["mean_ratio_criminal"],
                "sem_random": whole_data["random"]["sem_ratio_criminal"],
                "std_random": whole_data["random"]["std_ratio_criminal"],
                "mean_small-world": whole_data["small-world"]["mean_ratio_criminal"],
                "sem_small-world": whole_data["small-world"]["sem_ratio_criminal"],
                "std_small-world": whole_data["small-world"]["std_ratio_criminal"],
                "mean_iteration": whole_data["preferential"]["mean_iteration"],
            }
        },
        y_data_to_plot=["mean_preferential", "mean_random", "mean_small-world"],
        x_data_to_plot="mean_iteration",
        title=False,
        xlabel="Rounds",
        ylabel="Ratio (%)",
        plot_deviation="sem",
        ylim=[0, 0.5],
        legend_size=20,
        axes_size=30,
        tick_size=30,
    )

    # compare_time_series(whole_data)
    ax_2 = plotter.plot_lines_comparative(
        dict_data=whole_data,
        y_data_to_plot=[
            "mean_secrecy",
            "mean_flow_information",
            "mean_size_of_largest_component",
        ],
        x_data_to_plot="mean_iteration",
        xlabel="Rounds",
        plot_deviation="sem",
    )

    ax_3 = plotter.plot_hist(
        dict_data=whole_data,
        y_data_to_plot=[
            "mean_secrecy",
            "mean_flow_information",
            "mean_size_of_largest_component",
        ],
        xlabel=True,
        ylabel=True,
    )

    ax_4 = plotter.plot_lines_correlation_grid(
        dict_data=whole_data,
        y_data_to_plot=[
            "degree",
            "betweenness",
            "closeness",
            "eigen vector",
        ],
        x_data_to_plot="criminal_likelihood",
    )

elif args.whole_pipeline:
    """Simulate the simulation form
    Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
    Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.
    """

    # Add nodes to network
    # First convert to gt
    meta_sim_pref = MetaSimulator(
        network_name=args.read_data,
        attachment_method="preferential",
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        k=74,
        random_fit_init=False,
    )

    meta_sim_rnd = MetaSimulator(
        network_name=args.read_data,
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        k=74,
        attachment_method="random",
    )
    meta_sim_sw = MetaSimulator(
        network_name=args.read_data,
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        prob=0.2,
        k=74,
        attachment_method="small-world",
    )

    # Get overview of the new network
    complete_network_stats_pref = NetworkStats(
        NetworkConverter.gt_to_nk(meta_sim_pref.network)
    )
    complete_network_stats_rand = NetworkStats(
        NetworkConverter.gt_to_nk(meta_sim_rnd.network)
    )
    complete_network_stats_sw = NetworkStats(
        NetworkConverter.gt_to_nk(meta_sim_sw.network)
    )
    complete_network_stats_pref.get_overview()
    complete_network_stats_rand.get_overview()
    complete_network_stats_sw.get_overview()

    logger.info("Doing preferential simulation")
    data_collector_pref = meta_sim_pref.avg_play(
        rounds=args.rounds,
        n_groups=args.n_groups,
        repetition=args.n_samples,
        ith_collect=int(args.rounds / 15),
        measure_topology=args.topo_meas,
        measure_likelihood_corr=args.criminal_likelihood_corr,
        execute=args.execute,
    )
    logger.info("Done")
    logger.info("Doing random simulation")
    data_collector_rand = meta_sim_rnd.avg_play(
        rounds=args.rounds,
        n_groups=args.n_groups,
        repetition=args.n_samples,
        ith_collect=int(args.rounds / 15),
        measure_topology=args.topo_meas,
        measure_likelihood_corr=args.criminal_likelihood_corr,
        execute=args.execute,
    )
    logger.info("Done")
    logger.info("Doing small-world simulation")
    data_collector_sw = meta_sim_sw.avg_play(
        rounds=args.rounds,
        n_groups=args.n_groups,
        repetition=args.n_samples,
        ith_collect=int(args.rounds / 15),
        measure_topology=args.topo_meas,
        measure_likelihood_corr=args.criminal_likelihood_corr,
        execute=args.execute,
    )
    logger.info("Done")

    whole_data = {
        "preferential": data_collector_pref,
        "random": data_collector_rand,
        "small-world": data_collector_sw,
    }

    ax_0 = plotter.plot_lines(
        dict_data=whole_data,
        y_data_to_plot=["mean_ratio_honest", "mean_ratio_wolf", "mean_ratio_criminal"],
        x_data_to_plot="mean_iteration",
        title=True,
        xlabel="Rounds",
        ylabel="Ratio (%)",
        plot_deviation="sem",
    )

    ax_1 = plotter.plot_lines(
        dict_data={
            "Compare mean criminal ratio": {
                "mean_preferential": whole_data["preferential"]["mean_ratio_criminal"],
                "sem_preferential": whole_data["preferential"]["sem_ratio_criminal"],
                "std_preferential": whole_data["preferential"]["std_ratio_criminal"],
                "mean_random": whole_data["random"]["mean_ratio_criminal"],
                "sem_random": whole_data["random"]["sem_ratio_criminal"],
                "std_random": whole_data["random"]["std_ratio_criminal"],
                "mean_small-world": whole_data["small-world"]["mean_ratio_criminal"],
                "sem_small-world": whole_data["small-world"]["sem_ratio_criminal"],
                "std_small-world": whole_data["small-world"]["std_ratio_criminal"],
                "mean_iteration": whole_data["preferential"]["mean_iteration"],
            }
        },
        y_data_to_plot=["mean_preferential", "mean_random", "mean_small-world"],
        x_data_to_plot="mean_iteration",
        title=True,
        xlabel="Rounds",
        ylabel="Ratio (%)",
        plot_deviation="sem",
    )

    if args.topo_meas:
        compare_time_series(whole_data)
        print(f"{whole_data['preferential']['mean_secrecy']=}")
        print(f"{whole_data['preferential']['sem_secrecy']=}")
        print(f"{whole_data['random']['mean_secrecy']=}")
        print(f"{whole_data['small-world']['mean_secrecy']=}")

        ax_2 = plotter.plot_lines_comparative(
            dict_data=whole_data,
            y_data_to_plot=[
                "mean_secrecy",
                "mean_flow_information",
                "mean_size_of_largest_component",
            ],
            x_data_to_plot="mean_iteration",
            xlabel="Rounds",
            plot_deviation="sem",
        )

        ax_3 = plotter.plot_hist(
            dict_data=whole_data,
            y_data_to_plot=[
                "mean_secrecy",
                "mean_flow_information",
                "mean_size_of_largest_component",
            ],
            xlabel=True,
            ylabel=True,
        )

    if args.criminal_likelihood_corr:
        ax_4 = plotter.plot_lines_correlation(
            dict_data=whole_data,
            y_data_to_plot=[
                "degree",
                "betweenness",
                "katz",
                "closeness",
                "eigen vector",
            ],
            x_data_to_plot="criminal_likelihood",
        )
        ax_5 = plotter.plot_lines_correlation_grid(
            dict_data=whole_data,
            y_data_to_plot=[
                "degree",
                "betweenness",
                "katz",
                "closeness",
                "eigen vector",
            ],
            x_data_to_plot="criminal_likelihood",
        )

elif args.sensitivity_analysis:
    """Runs a sensitivity analysis on the given choice."""
    if args.sensitivity_analysis == "sim-mart-vaq":

        sa = SensitivityAnalyser()
        sobol_indices = sa.sim_mart_vaq_sa(
            output_value=args.output_value,
            n_samples=args.n_samples,
            rounds=args.rounds,
        )

elif args.sensitivity_analysis_links:
    """Performs an kinda sensitivity analysis on the impact of links on the network."""
    n_links = np.linspace(20, 80, 5, dtype=int)
    file_name = DirectoryFinder().result_dir_data_sa_link + args.case + ".json"
    if os.path.isfile(file_name) and os.access(file_name, os.R_OK):
        with open(file_name) as fp:
            whole_data = json.load(fp)

    elif not os.path.isfile(file_name):
        whole_data = {"preferential": {}, "random": {}, "small-world": {}}

    for structure in whole_data.keys():
        for link in n_links:
            if str(link) in whole_data[structure]:
                pass
            else:
                print(f"Doing {structure=}, {link=}")
                # create meta_simulator
                meta_sim = MetaSimulator(
                    network_name=args.read_data,
                    attachment_method=structure,
                    ratio_honest=args.ratio_honest,
                    ratio_wolf=args.ratio_wolf,
                    k=link,
                    random_fit_init=False,
                )

                # Play the games parallel
                logger.info(f"Doing {structure} simulation")
                data_collector = meta_sim.avg_play(
                    rounds=args.rounds,
                    n_groups=args.n_groups,
                    repetition=args.n_samples,
                    ith_collect=args.rounds,
                    execute=args.execute,
                    show_no_bar=False,
                )
                whole_data[structure][str(link)] = list(
                    data_collector["m_ratio_criminal"][:, 1]
                )

                # saving each step to a dictionary
                with open(
                    file_name,
                    "w",
                ) as fp:
                    json.dump(whole_data, fp, indent=4)

    plotter.plot_violin(
        whole_data,
        ylabel="Ratio (%)",
        xlabel="Density",
        density_conv=[0.0074, 0.013, 0.018, 0.023, 0.029],
    )

elif args.phase_diagram:

    file_name = DirectoryFinder().result_dir_data_phase_diag + "data.json"

    resolution = 6  # of the grid

    if os.path.isfile(file_name) and os.access(file_name, os.R_OK):
        with open(file_name) as fp:
            meta_phase_diag = json.load(fp)

    elif not os.path.isfile(file_name):
        # Get the simple runs
        phase_diag = {
            "case_1": {
                "param_y": "gamma",
                "y_range": np.linspace(0, 1, resolution),
                "x_range": np.linspace(0, 1000, resolution),
                "param_x": "r_c",
            },
            "case_2": {
                "param_y": "beta_s",
                "y_range": np.linspace(0, 1000, resolution),
                "x_range": np.linspace(0, 1000, resolution),
                "param_x": "r_c",
            },
            "case_3": {
                "param_y": "beta_h",
                "y_range": np.linspace(0, 1000, resolution),
                "x_range": np.linspace(0, 1000, resolution),
                "param_x": "r_c",
            },
            "case_4": {
                "param_y": "delta",
                "y_range": np.linspace(0, 1, resolution),
                "x_range": np.linspace(0, 1, resolution),
                "param_x": "tau",
            },
        }

        meta_phase_diag = {}
        meta_phase_diag["preferential"] = deepcopy(phase_diag)
        meta_phase_diag["random"] = deepcopy(phase_diag)
        meta_phase_diag["small-world"] = deepcopy(phase_diag)

    for structure in meta_phase_diag.keys():
        # init simulation
        # Add nodes to network
        # First convert to gt
        # Get actual criminal network
        nx_network = NetworkReader().get_data(args.read_data)

        meta_sim = MetaSimulator(
            attachment_method=structure,
            network_name=args.read_data,
            ratio_honest=args.ratio_honest,
            ratio_wolf=args.ratio_wolf,
            prob=0.2,
            k=74,
        )
        for case in meta_phase_diag[structure].keys():
            # check if grid is already been calculated
            if "grid_status" in meta_phase_diag[structure][case].keys():
                pass
            else:
                logger.info(f"Doing {case=} of {structure=}")
                nx = resolution
                ny = resolution
                meta_phase_diag[structure][case]["grid_status"] = np.empty(
                    (ny, nx), dtype=object
                )
                meta_phase_diag[structure][case]["grid_value"] = np.empty(
                    (ny, nx), dtype=float
                )
                meta_phase_diag[structure][case]["ratio_criminal"] = np.empty(
                    (ny, nx), dtype=float
                )

                for x_i in range(0, nx):
                    for y_i in range(0, ny):

                        if case != "case_4":
                            variable_dict = dict(
                                zip(
                                    [
                                        meta_phase_diag[structure][case]["param_x"],
                                        meta_phase_diag[structure][case]["param_y"],
                                        "c_c",
                                    ],
                                    [
                                        meta_phase_diag[structure][case]["x_range"][
                                            x_i
                                        ],
                                        meta_phase_diag[structure][case]["y_range"][
                                            y_i
                                        ],
                                        meta_phase_diag[structure][case]["x_range"][
                                            x_i
                                        ],
                                    ],
                                )
                            )
                        else:
                            variable_dict = dict(
                                zip(
                                    [
                                        meta_phase_diag[structure][case]["param_x"],
                                        meta_phase_diag[structure][case]["param_y"],
                                    ],
                                    [
                                        meta_phase_diag[structure][case]["x_range"][
                                            x_i
                                        ],
                                        meta_phase_diag[structure][case]["y_range"][
                                            y_i
                                        ],
                                    ],
                                )
                            )

                        all_files = [
                            file
                            for file in os.listdir(
                                DirectoryFinder().population_data_dir
                            )
                            if re.search(structure, file)
                        ]
                        matches = [
                            f"h_{meta_sim.ratio_honest}",
                            f"w_{meta_sim.ratio_wolf}",
                            f"k_{meta_sim.k}",
                        ]
                        matching_files = [
                            file
                            for file in all_files
                            if all(word in file for word in matches)
                        ]
                        if len(matching_files) != 0:
                            list_of_population = meta_sim.load_list_of_populations(
                                args.n_samples, matching_files
                            )

                        simulators = SimMartVaq(
                            network=meta_sim.network, **variable_dict
                        )

                        data_collector = simulators.avg_play(
                            network=list_of_population,
                            rounds=args.rounds,
                            n_groups=args.n_groups,
                            ith_collect=args.rounds,  # only need the last measurement
                            repetition=args.n_samples,
                            show_no_bar=True,
                        )

                        print(f"{variable_dict=}")
                        print(f"{simulators.r_c=},{simulators.c_c=}")
                        print(
                            "criminal_ratio = "
                            + str(data_collector["mean_ratio_criminal"][-1])
                        )

                        # Only look at the ratio and get the status with the highest ratio
                        # at the end
                        filtered_dict = {
                            k: v
                            for k, v in data_collector.items()
                            if k
                            in [
                                "mean_ratio_criminal",
                                "mean_ratio_wolf",
                                "mean_ratio_honest",
                            ]
                        }

                        meta_phase_diag[structure][case]["grid_status"][y_i, x_i] = max(
                            filtered_dict, key=lambda x: filtered_dict[x][-1]
                        )
                        meta_phase_diag[structure][case]["grid_value"][
                            y_i, x_i
                        ] = filtered_dict[
                            meta_phase_diag[structure][case]["grid_status"][y_i, x_i]
                        ][
                            -1
                        ]
                        meta_phase_diag[structure][case]["ratio_criminal"][
                            y_i, x_i
                        ] = data_collector["mean_ratio_criminal"][-1]

                with open(
                    file_name,
                    "w",
                ) as fp:
                    json.dump(meta_phase_diag, fp, indent=4, cls=NumpyArrayEncoder)

    ax = plotter.plot_phase_diag(meta_phase_diag)

elif args.animate_simulation:
    """Create an animation of the simulation."""
    animateur = Animateur()
    animateur.create_animation()

elif args.animate_attachment_process:
    """Create an animation of the attachment process."""
    animateur = Animateur()
    animateur.create_animation()

elif args.get_network_stats:
    """Return the mean/standard deviation of a population structure.

    The strucutre is either preferential,random or small-world
    """

    nx_network = NetworkReader().get_data(args.read_data)

    # Preferential
    meta_sim_pref = MetaSimulator(
        network_name=nx_network.name,
        attachment_method="preferential",
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        k=80,
        random_fit_init=False,
    )

    # Random
    meta_sim_rnd = MetaSimulator(
        network_name=nx_network.name,
        attachment_method="random",
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        k=80,  # 0.0034 for random
        random_fit_init=False,
    )

    # Small-world
    meta_sim_sw = MetaSimulator(
        network_name=nx_network.name,
        attachment_method="small-world",
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        prob=0.2,
        k=80,
        random_fit_init=False,
    )

    # Create populations
    list_of_populations_pref = meta_sim_pref.create_list_of_populations(
        repetition=args.n_samples
    )
    list_of_populations_rnd = meta_sim_rnd.create_list_of_populations(
        repetition=args.n_samples
    )
    list_of_populations_sw = meta_sim_sw.create_list_of_populations(
        repetition=args.n_samples
    )

    list_of_network_stats_pref = [
        NetworkStats(NetworkConverter.gt_to_nk(network)).get_overview()
        for network in list_of_populations_pref
    ]
    list_of_network_stats_rnd = [
        NetworkStats(NetworkConverter.gt_to_nk(network)).get_overview()
        for network in list_of_populations_rnd
    ]
    list_of_network_stats_sw = [
        NetworkStats(NetworkConverter.gt_to_nk(network)).get_overview()
        for network in list_of_populations_sw
    ]

    mean_dict_of_network_stats_pref = dict_mean(list_of_network_stats_pref)
    mean_dict_of_network_stats_rnd = dict_mean(list_of_network_stats_rnd)
    mean_dict_of_network_stats_sw = dict_mean(list_of_network_stats_sw)

    with open(
        DirectoryFinder().result_dir_data_network_stats + f"result_{timestamp()}.json",
        "w",
    ) as fp:
        json.dump(
            {
                "preferential": mean_dict_of_network_stats_pref,
                "random": mean_dict_of_network_stats_rnd,
                "small-world": mean_dict_of_network_stats_sw,
            },
            fp,
            indent=4,
        )

elif args.compare_w_rnd_init:
    """Compare simulations with random fitness init or not."""

    structures = [
        "preferential",
        "random",
        "small-world",
        "preferential_rnd",
        "random_rnd",
        "small-world_rnd",
    ]
    whole_data = {}
    for structure in structures:
        # Add nodes to network
        # First convert to gt
        meta_sim = MetaSimulator(
            network_name=args.read_data,
            attachment_method=structure.replace("_rnd", ""),
            ratio_honest=args.ratio_honest,
            ratio_wolf=args.ratio_wolf,
            k=args.k,
        )

        logger.info(f"Doing {structure} simulation")
        data_collector = meta_sim.avg_play(
            rounds=args.rounds,
            n_groups=args.n_groups,
            repetition=args.n_samples,
            ith_collect=int(args.rounds / 15),
            measure_topology=args.topo_meas,
            measure_likelihood_corr=args.criminal_likelihood_corr,
            execute=args.execute,
            rnd_fit_init=False if "rnd" not in structure else True,
        )
        logger.info("Done")

        whole_data[structure] = data_collector
    print(whole_data.keys())
    ax_1 = plotter.plot_lines(
        dict_data={
            " ": {
                "mean_preferential": whole_data["preferential"]["mean_ratio_criminal"],
                "sem_preferential": whole_data["preferential"]["sem_ratio_criminal"],
                "std_preferential": whole_data["preferential"]["std_ratio_criminal"],
                "mean_random": whole_data["random"]["mean_ratio_criminal"],
                "sem_random": whole_data["random"]["sem_ratio_criminal"],
                "std_random": whole_data["random"]["std_ratio_criminal"],
                "mean_small-world": whole_data["small-world"]["mean_ratio_criminal"],
                "sem_small-world": whole_data["small-world"]["sem_ratio_criminal"],
                "std_small-world": whole_data["small-world"]["std_ratio_criminal"],
                "mean_iteration": whole_data["preferential"]["mean_iteration"],
                "mean_preferential_rnd": whole_data["preferential_rnd"][
                    "mean_ratio_criminal"
                ],
                "sem_preferential_rnd": whole_data["preferential_rnd"][
                    "sem_ratio_criminal"
                ],
                "std_preferential_rnd": whole_data["preferential_rnd"][
                    "std_ratio_criminal"
                ],
                "mean_random_rnd": whole_data["random_rnd"]["mean_ratio_criminal"],
                "sem_random_rnd": whole_data["random_rnd"]["sem_ratio_criminal"],
                "std_random_rnd": whole_data["random_rnd"]["std_ratio_criminal"],
                "mean_small-world_rnd": whole_data["small-world_rnd"][
                    "mean_ratio_criminal"
                ],
                "sem_small-world_rnd": whole_data["small-world_rnd"][
                    "sem_ratio_criminal"
                ],
                "std_small-world_rnd": whole_data["small-world_rnd"][
                    "std_ratio_criminal"
                ],
                "mean_iteration_rnd": whole_data["preferential_rnd"]["mean_iterations"],
            }
        },
        y_data_to_plot=[
            "mean_preferential",
            "mean_random",
            "mean_small-world",
            "mean_preferential_rnd",
            "mean_random_rnd",
            "mean_small-world_rnd",
        ],
        x_data_to_plot="mean_iteration",
        title=False,
        xlabel="Rounds",
        ylabel="Ratio (%)",
        plot_deviation="sem",
        # ylim=[0, 0.1],
        legend_size=20,
        axes_size=30,
        tick_size=30,
    )
