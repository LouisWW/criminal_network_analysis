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
from typing import TypeVar

import numpy as np
import pandas as pd
from p_tqdm import p_map
from SALib.analyze import sobol
from SALib.sample import saltelli
from src.config.config import ConfigParser
from src.network_utils.network_converter import NetworkConverter
from src.network_utils.network_reader import NetworkReader
from src.simulators.sim_mart_vaq import SimMartVaq

logger = logging.getLogger("logger")

SA = TypeVar("SA", bound="SensitivityAnalyser")


def sim_mart_vaq_sa_helper(tuple_of_variable: Any) -> float:
    """Run the simulation Mart-Vaq given the parameter."""
    # Unpack input variables
    gt_network, problem, params, output_value, rounds = tuple_of_variable

    variable_dict = OrderedDict().fromkeys(problem["names"], 0)
    variable_dict = dict(zip(variable_dict.keys(), params))

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
        def wrapper_decorator(self: SA, *args: Tuple, **kwargs: Dict[str, Any]) -> Any:
            value = func(self, *args, **kwargs)
            # Save results
            if self.args.save:
                # Get timestamp
                e = datetime.datetime.now()
                timestamp = e.strftime("%d-%m-%Y-%H-%M")
                # Get the saving directory
                # Get directory first
                path = os.getcwd()
                par_dir = os.path.abspath(os.path.join(path))
                # par_dir = ../src/
                savig_dir = par_dir + "/results/data/sensitivity_analysis/"
                file_name = (
                    savig_dir
                    + func.__name__
                    + "_"
                    + str(kwargs["output_value"])
                    + "_r_"
                    + str(kwargs["rounds"])
                    + "_n_s_"
                    + str(kwargs["n_samples"])
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

        The analysis is limited on "delta","tau","gamma","beta_s","beta_h" and "beta_c".
        For the sake of time, no sensitivity analysis is conducted on the ration of h/w/c.
        Actually, h/w/c are correlated and would therefore return falsified Sobol results.
        Furthermore, the importance of the two parameters, temperature or the probability
        of random mutation is known. If the temperature is too high, the system will have
        random chaotic switched between status. If the probability of random mutation is high,
        the system with converge to a ration of 0.33 for h/w/c. What is interesting is to see the
        importance of the investigation stage as well as the influence of criminals on lone wolfs.
        Also it is interesting to see, what influence the penalty on the whole criminal organisation
        can have.

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
                "num_vars": 6,
                "names": [
                    "delta",
                    "tau",
                    "gamma",
                    "beta_s",
                    "beta_h",
                    "beta_c",
                ],
                "bounds": [[0, 1], [0, 1], [0, 1], [0, 200], [0, 200], [0, 200]],
            }
        # sample
        param_values = saltelli.sample(problem, n_samples)

        # ((number of loops*rounds/average_time_per_round)/n_threads)/ convert_to_hours
        approx_running_time = ((len(param_values) * rounds * (45 / 1000)) / 23) / 3600
        logger.warning(
            f"The sensitivity analysis will take approx {approx_running_time:.2f}h on\
            24 cpus (~ 3.8 GHz))"
        )
        # Running multiprocessing
        num_cpus = multiprocessing.cpu_count() - 1
        Y = p_map(
            sim_mart_vaq_sa_helper,
            (
                [
                    (gt_network, problem, params, output_value, rounds)
                    for params in param_values
                ]
            ),
            **{"num_cpus": num_cpus, "desc": "Running sensitivity analysis"},
        )

        Y_array = np.asarray(Y)

        # analyse
        sobol_indices = sobol.analyze(problem, Y_array)
        return sobol_indices
