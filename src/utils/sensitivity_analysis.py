"""This file contains the SensitivityAnalyser class.

The SensitivityAnalyser is based on the SALib library.

__author__ = Louis Weyland
__date__   = 5/05/2022
"""
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
from typing import Union

import graph_tool.all as gt
import numpy as np
import pandas as pd
import tqdm
from config.config import ConfigParser
from SALib.analyze import sobol
from SALib.sample import saltelli
from simulators.meta_simulator import MetaSimulator
from simulators.sim_mart_vaq import SimMartVaq
from utils.tools import DirectoryFinder
from utils.tools import timestamp

logger = logging.getLogger("logger")

SA = TypeVar("SA", bound="SensitivityAnalyser")


class SensitivityAnalyser(ConfigParser):
    """Performs sensitivity analysis on different models."""

    def __init__(self, problem: dict = None) -> None:
        """Inherit from Configparser."""
        super().__init__()
        if problem is None:
            self.problem = {
                "num_vars": 10,
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
                ],
                "bounds": [
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 20],
                    [0, 20],
                    [0, 20],
                    [0, 100],
                    [0, 100],
                    [0, 100],
                    [0, 100],
                ],
            }

    def save_results(func: Callable) -> Any:
        """Save the sensitivity analysis results.

        Acts as a wrapper and must be at the top of the class
        """

        @functools.wraps(func)
        def wrapper_decorator(self: SA, *args: Tuple, **kwargs: Dict[str, Any]) -> Any:
            value = func(self, *args, **kwargs)
            # Save results
            if self.args.save:
                file_name = (
                    DirectoryFinder().result_dir_data_sa
                    + func.__name__
                    + "_"
                    + str(kwargs["output_value"])
                    + "_r_"
                    + str(kwargs["rounds"])
                    + "_n_s_"
                    + str(kwargs["n_samples"])
                    + "_"
                    + self.args.attach_meth
                    + "_"
                    + timestamp()
                )

                df_list = value.to_df()
                df = pd.concat([df_list[0], df_list[1], df_list[2]], axis=1)
                with open(file_name, "w") as fo:
                    fo.write(df.to_string())

            return value

        return wrapper_decorator

    @save_results
    def sim_mart_vaq_sa(
        self,
        output_value: str,
        n_samples: int,
        rounds: int,
    ) -> Union[int, Dict[str, List[float]]]:
        """Runnning the sensitivity analysis."""
        if self.args.running_chunk:
            # chuck the sensitivity analysis and save the dataframe inbetween
            self.check_file_and_graph_exist(n_samples, rounds)
            param_values, gt_network = self.load_file_and_graph(n_samples, rounds)
            # Getting all param_values that haven't run yet
            latest_param_values = param_values[param_values.isnull().any(axis=1)]

            for sub_pd in self.chunker(latest_param_values, 200):
                latest_param_values_to_list = sub_pd.values.tolist()
                list_of_param_comb = [
                    (gt_network, self.problem, params, output_value, rounds)
                    for params in latest_param_values_to_list
                ]

                y = self.sensitivity_analysis_parallel(list_of_param_comb)
                param_values.loc[sub_pd.index, "y"] = y
                self.overwrite_file(param_values, n_samples, rounds)

            # analyse
            sobol_indices = sobol.analyze(self.problem, np.asarray(param_values["y"]))
            return sobol_indices

        elif not self.args.running_chunk:
            # Get the network of criminal first
            meta_sim = MetaSimulator(
                network_name=self.args.read_data,
                attachment_method=self.args.attach_meth,
                ratio_honest=self.args.ratio_honest,
                ratio_wolf=self.args.ratio_wolf,
                k=self.args.k,
            )
            gt_network = meta_sim.network
            param_values = self.create_saltelli_samples(self.problem, n_samples)
            param_values = [
                (gt_network, self.problem, params, output_value, rounds)
                for params in param_values
            ]

            results = self.sensitivity_analysis_parallel(param_values)

            # analyse
            sobol_indices = sobol.analyze(self.problem, results)
            return sobol_indices
        return -1

    def sensitivity_analysis_parallel(self, list_of_param_comb: list) -> np.ndarray:
        """Run the simulation parallel."""
        # ((number of loops*rounds/average_time_per_round)/n_threads)/ convert_to_hours
        # Running multiprocessing
        num_cpus = multiprocessing.cpu_count() - 1
        Y = []
        with multiprocessing.Pool(num_cpus) as p:
            for result in tqdm.tqdm(
                p.imap(self.sim_mart_vaq_sa_helper, list_of_param_comb),
                total=len(list_of_param_comb),
            ):
                Y.append(result)
            p.close()
            p.join()

        Y_array = np.asarray(Y)
        return Y_array

    def sim_mart_vaq_sa_helper(self, tuple_of_variable: Any) -> float:
        """Run the simulation Mart-Vaq given the parameter."""
        # Set the seed each time, otherwise the simulation will be exactly the same
        np.random.seed(0)
        (gt_network, problem, params, output_value, rounds) = tuple_of_variable

        # Unpack input variables
        variable_dict = OrderedDict().fromkeys(problem["names"], 0)
        variable_dict = dict(zip(variable_dict.keys(), params))

        simulator = SimMartVaq(network=gt_network, **variable_dict)
        _, data_collector = simulator.play(
            # ith_collect == rounds to collect only at the end
            network=simulator.network,
            rounds=rounds,
            ith_collect=rounds,
        )
        return data_collector[output_value][-1]

    def create_saltelli_samples(self, problem: dict, n_samples: int) -> np.ndarray:
        """Return saltelli samples."""
        return saltelli.sample(problem, n_samples)

    def check_file_and_graph_exist(self, n_samples: int, rounds: int) -> None:
        """Check if file and graph doesn't exist otherwise create."""
        file = (
            DirectoryFinder().result_dir_data_sa
            + self.args.attach_meth
            + "_"
            + str(rounds)
            + "_"
            + str(n_samples)
            + ".csv"
        )
        graph_file = (
            DirectoryFinder().result_dir_data_sa
            + self.args.attach_meth
            + "_graph.xml.gz"
        )
        if not os.path.isfile(file):
            param_values = self.create_saltelli_samples(self.problem, n_samples)
            param_values = pd.DataFrame.from_records(
                param_values, columns=self.problem["names"]
            )
            param_values["y"] = np.nan
            param_values.to_csv(file, index=True)
        if not os.path.isfile(graph_file):
            # create graph
            meta_sim = MetaSimulator(
                network_name=self.args.read_data,
                attachment_method=self.args.attach_meth,
                ratio_honest=self.args.ratio_honest,
                ratio_wolf=self.args.ratio_wolf,
                k=self.args.k,
            )

            meta_sim.network.save(graph_file)

    def overwrite_file(
        self, param_values: pd.DataFrame, n_samples: int, rounds: int
    ) -> None:
        """Overwite the file with new results."""
        file = (
            DirectoryFinder().result_dir_data_sa
            + self.args.attach_meth
            + "_"
            + str(rounds)
            + "_"
            + str(n_samples)
            + ".csv"
        )
        param_values.to_csv(file, index=True)

    def load_file_and_graph(
        self, n_samples: int, rounds: int
    ) -> Tuple[pd.DataFrame, gt.Graph]:
        """Load the data and the graph."""
        file = (
            DirectoryFinder().result_dir_data_sa
            + self.args.attach_meth
            + "_"
            + str(rounds)
            + "_"
            + str(n_samples)
            + ".csv"
        )
        graph_file = (
            DirectoryFinder().result_dir_data_sa
            + self.args.attach_meth
            + "_graph.xml.gz"
        )
        param_values = pd.read_csv(file)
        graph = gt.load_graph(graph_file)
        return param_values, graph

    def chunker(self, seq: pd.DataFrame, size: int) -> pd.DataFrame:
        """Chucks the panda DataFrame."""
        return (seq[pos: pos + size] for pos in range(0, len(seq), size))
