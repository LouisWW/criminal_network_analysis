"""This file contains the SensitivityAnalyser class.

The SensitivityAnalyser is based on the SALib library.

__author__ = Louis Weyland
__date__   = 5/05/2022
"""
import datetime
import functools
import logging
import multiprocessing
import os
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli
from src.config.config import ConfigParser
from src.network_utils.network_converter import NetworkConverter
from src.network_utils.network_reader import NetworkReader
from src.simulators.sim_mart_vaq import SimMartVaq
from tqdm import tqdm


logger = logging.getLogger("logger")


def sim_mart_vaq_sa_helper(tuple_of_variable: Any) -> float:
    """Run the simulation Mart-Vaq given the parameter."""
    # Unpack input variables
    gt_network, problem, params, output_value, rounds = tuple_of_variable

    variable_dict = OrderedDict().fromkeys(problem["names"], 0)
    variable_dict = dict(zip(variable_dict.keys(), params))

    np.random.seed(0)
    simulator = SimMartVaq(network=gt_network, **variable_dict)
    _, data_collector = simulator.play(network=simulator.network, rounds=rounds)
    return data_collector[output_value][-1]


class SensitivityAnalyser(ConfigParser):
    """Performs sensitivity analysis on different models."""

    def __init__(self) -> None:
        """Inherit from Configparser."""
        super().__init__()

    def save_results(func: Callable) -> Any:
        """Save the sensitivity analysis results.

        Acts as a wrapper and must be at the top of the class
        """

        @functools.wraps(func)
        def wrapper_decorator(
            self: SensitivityAnalyser, *args: Tuple, **kwargs: Dict[str, Any]
        ) -> Any:
            value = func(self, *args, **kwargs)
            # Save results
            if self.args.save:
                # Get timestamp
                e = datetime.datetime.now()
                timestamp = e.strftime("%d-%m-%Y-%H-%M")
                # Get the saving directory
                # Get directory first
                path = os.getcwd()
                par_dir = os.path.abspath(os.path.join(path, "../"))
                # par_dir = ../src/
                savig_dir = par_dir + "/results/data/sensitivity_analysis/"
                file_name = (
                    savig_dir
                    + func.__name__
                    + "_"
                    + str(kwargs["output_value"])
                    + "_"
                    + timestamp
                )

                df_list = value.to_df()
                df = pd.concat([df_list[0], df_list[1], df_list[2]], axis=1)
                with open(file_name, "w") as fo:
                    fo.write(df.__repr__())

            return value

        return wrapper_decorator

    @save_results
    def sim_mart_vaq_sa(
        self, output_value: str, n_samples: int, rounds: int, problem: dict = None
    ) -> Dict[str, List[float]]:
        """Perform a sensitivity analysis on the Martinez-Vaquero model.

        Args:
            network  (gt.graph): Criminal network
            output_value  (str): Name of the interested output value such as
                                    ['ratio_honest','ratio_wolf','ratio_criminal'
                                    'fitness_honest','fitness_criminal','fitness_wolf]
            problem      (dict): Define which variables to conduct sensitivity analysis
                                    with defining the bounds
            n_samples     (int): Define the size of the search space
            rounds        (int); Define the number of rounds played in the Simulation
        """
        # Get the network of criminal first
        nx_network = NetworkReader().get_data(self.args.read_data)
        gt_network = NetworkConverter.nx_to_gt(nx_network)
        if problem is None:
            problem = {
                "num_vars": 2,
                "names": [
                    "delta",
                    "tau",
                    "gamma",
                    "beta_s",
                    "beta_h",
                    "beta_c",
                    "c_w",
                    "c_c",
                    "r_w",
                    "r_c",
                    "temperature",
                    "mutation_prob",
                ],
                "bounds": [[0, 1], [0.1, 0.28]],
            }
        # sample
        param_values = saltelli.sample(problem, n_samples)

        # Running multiprocessing
        num_core = multiprocessing.cpu_count() - 1
        num_core = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(num_core)
        Y = []
        for res in tqdm(
            pool.map(
                sim_mart_vaq_sa_helper,
                (
                    (gt_network, problem, params, output_value, rounds)
                    for params in param_values
                ),
            ),
            desc="Running sensitivity analysis...",
            total=len(param_values),
        ):
            Y.append(res)

        Y_array = np.asarray(Y)
        pool.close()
        pool.join()
        # analyse
        sobol_indices = sobol.analyze(problem, Y_array)
        return sobol_indices
