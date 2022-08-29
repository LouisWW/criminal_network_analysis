"""
This script contains regroups all the functions and pipelines needed to recreate the results.

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
import json
import logging
import os

import numpy as np
from config.config import ConfigParser
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from network_utils.network_stats import NetworkStats
from simulators.meta_simulator import MetaSimulator
from simulators.sim_mart_vaq import SimMartVaq
from utils.animation import Animateur
from utils.plotter import Plotter
from utils.sensitivity_analysis import SensitivityAnalyser
from utils.stats import compare_time_series
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
        print(file_name)
        if os.path.isfile(file_name) and os.access(file_name, os.R_OK):
            with open(file_name) as fp:
                previous_results = json.load(fp)

            # get the number of runs already done
            n_prev_runs = max(int(key) for key in previous_results.keys())
            simple_runs = {
                str(int(k) + n_prev_runs): v
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

        ax_1 = plotter.plot_lines(
            dict_data={args.attach_meth: data_collector},
            y_data_to_plot=[
                "mean_fitness_honest",
                "mean_fitness_wolf",
                "mean_fitness_criminal",
            ],
            x_data_to_plot="mean_iteration",
            xlabel="Rounds",
            ylabel="Average fitness",
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
            whole_data[structure] = json.load(file_name)

    # Get the mean of the data
    for structure, data in whole_data.items():
        whole_data[structure] = get_mean_std_over_list(data)

    # Ready to be plotted
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

    compare_time_series(whole_data)
    ax_2 = plotter.plot_lines_comparative(
        dict_data=whole_data,
        y_data_to_plot=[
            "mean_density",
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
            "mean_density",
            "mean_flow_information",
            "mean_size_of_largest_component",
        ],
        title=True,
    )

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
        k=2,
        random_fit_init=False,
    )

    meta_sim_rnd = MetaSimulator(
        network_name=args.read_data,
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        k=2,
        attachment_method="random",
    )
    meta_sim_sw = MetaSimulator(
        network_name=args.read_data,
        ratio_honest=args.ratio_honest,
        ratio_wolf=args.ratio_wolf,
        prob=0.4,
        k=6,
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
        print(f"{whole_data['preferential']['mean_density']=}")
        print(f"{whole_data['preferential']['sem_density']=}")
        print(f"{whole_data['random']['mean_density']=}")
        print(f"{whole_data['small-world']['mean_density']=}")

        ax_2 = plotter.plot_lines_comparative(
            dict_data=whole_data,
            y_data_to_plot=[
                "mean_density",
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
                "mean_density",
                "mean_flow_information",
                "mean_size_of_largest_component",
            ],
            title=True,
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
            problem=None,
            n_samples=args.n_samples,
            rounds=args.rounds,
        )

elif args.phase_diagram:
    parameter_dict = {
        "1": "beta_s",
        "2": "beta_h",
        "3": "beta_c",
        "4": "delta",
        "5": "gamma",
        "6": "tau",
        "7": "r_w",
        "8": "r_c",
    }

    print("Which parameters to test? Please give a number:")
    for k, v in parameter_dict.items():
        print(f"{k:<2}|{v:<15}")
    param_x = input("Parameter 1:\n")
    param_y = input("Parameter 2:\n")
    x_input = input("Range of parameter 1:  example: 0,5\n")
    x_input_tpl = tuple(int(x) for x in x_input.split(","))
    y_input = input("Range of parameter 2:  example: 0,5\n")
    y_input_tpl = tuple(int(x) for x in y_input.split(","))

    assert param_x != param_y, " Parameter can't be the same!"
    # create a mesh grid
    nx, ny = (10, 10)
    x_range = np.linspace(x_input_tpl[0], x_input_tpl[1], nx)
    y_range = np.linspace(y_input_tpl[0], y_input_tpl[1], ny)
    grid = np.empty((nx, ny), dtype=object)

    # init simulation
    # Add nodes to network
    # First convert to gt
    # Get actual criminal network
    nx_network = NetworkReader().get_data(args.read_data)

    meta_sim = MetaSimulator(
        network_name=nx_network.name,
        ratio_honest=0.33,
        ratio_wolf=0.33,
        random_fit_init=False,
    )
    for x_i in range(0, nx):
        for y_i in range(0, ny):
            variable_dict = dict(
                zip(
                    [parameter_dict[param_x], parameter_dict[param_y]],
                    [x_range[x_i], y_range[y_i]],
                )
            )
            simulators = SimMartVaq(
                network=meta_sim.network,
                **variable_dict,
                mutation_prob=0.0001,  # only fermi function
            )
            data_collector = simulators.avg_play(
                network=simulators.network,
                rounds=args.rounds,
                n_groups=1,
                ith_collect=args.rounds,  # only need the last measurement
                repetition=args.n_samples,
            )

            # Only look at the ratio and get the status with the highest ratio at the end
            filtered_dict = {
                k: v
                for k, v in data_collector.items()
                if k in ["mean_ratio_criminal", "mean_ratio_wolf", "mean_ratio_honest"]
            }

            grid[x_i, y_i] = max(filtered_dict, key=lambda x: filtered_dict[x][-1])

    ax = plotter.plot_phase_diag(
        grid,
        x_range,
        y_range,
        parameter_dict[param_x],
        parameter_dict[param_y],
        simulator=simulators,
    )

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
    ratio_honest = 0.95
    ratio_wolf = 0.01
    logger.info(f"Ration : {ratio_honest=}, {ratio_wolf=}")

    # Preferential
    meta_sim_pref = MetaSimulator(
        network_name=nx_network.name,
        attachment_method="preferential",
        ratio_honest=ratio_honest,
        ratio_wolf=ratio_wolf,
        k=2,
        random_fit_init=False,
    )

    # Random
    meta_sim_rnd = MetaSimulator(
        network_name=nx_network.name,
        attachment_method="random",
        ratio_honest=ratio_honest,
        ratio_wolf=ratio_wolf,
        k=2,  # 0.0034 for random
        random_fit_init=False,
    )

    # Small-world
    meta_sim_sw = MetaSimulator(
        network_name=nx_network.name,
        attachment_method="small-world",
        ratio_honest=ratio_honest,
        ratio_wolf=ratio_wolf,
        k=6,
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
