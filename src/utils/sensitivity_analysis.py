"""This file contains the SensitivityAnalyser class.

The SensitivityAnalyser is based on the SALib library.

__author__ = Louis Weyland
__date__   = 5/05/2022
"""
import multiprocessing
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from src.config.config import ConfigParser
from src.network_utils.network_converter import NetworkConverter
from src.network_utils.network_reader import NetworkReader
from src.simulators.sim_mart_vaq import SimMartVaq


def sim_mart_vaq_sa_helper(tuple_of_variable: Any) -> float:
    """Run the simulation Mart-Vaq given the parameter."""
    # Unpack input variables
    gt_network, problem, params, output_value = tuple_of_variable

    variable_dict = OrderedDict().fromkeys(problem["names"], 0)
    variable_dict = dict(zip(variable_dict.keys(), params))

    simulator = SimMartVaq(network=gt_network, **variable_dict)
    _, data_collector = simulator.play(network=simulator.network, rounds=10)
    return data_collector[output_value][-1]


class SensitivityAnalyser(ConfigParser):
    """Performs sensitivity analysis on different models."""

    def __init__(self) -> None:
        """Inherit from Configparser."""
        super().__init__()

    def sim_mart_vaq_sa(
        self, output_value: str, search_space: dict = None, n_samples: int = 20
    ) -> Dict[str, List[float]]:
        """Perform a sensitivity analysis on the Martinez-Vaquero model.

        Args:
            network  (gt.graph): Criminal network
            output_value  (str): Name of the interested output value such as
                                    ['ratio_honest','ratio_wolf','ratio_criminal'
                                    'fitness_honest','fitness_criminal','fitness_wolf]
            search_space (dict): Define which variables to conduct sensitivity analysis
                                    with defining the bounds
            n_samples     (int): Define the size of the search space
        """
        # Get the network of criminal first
        nx_network = NetworkReader().get_data(self.args.read_data)
        gt_network = NetworkConverter.nx_to_gt(nx_network)

        if search_space is not None:
            problem = {
                "num_vars": 2,
                "names": ["tau", "ratio_wolf"],
                "bounds": [[0, 1], [0.1, 0.28]],
            }

        # sample
        param_values = saltelli.sample(problem, n_samples)

        num_core = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(num_core)
        Y =np.array(pool.map(
            sim_mart_vaq_sa_helper,
            ((gt_network, problem, params, output_value) for params in param_values),
        ))
        
        pool.close()
        pool.join()

        # analyse
        sobol_indices = sobol.analyze(problem, Y)
        return sobol_indices
