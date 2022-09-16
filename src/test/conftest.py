"""
This script's contains all the pytest fixtures.

The fixtures are a piece of code which can be used
for multiple tests as setup.
By doing so we can avoid to setup time-consuming lines of code
for multiple tests


__author__ = Louis Weyland
__date__   = 23/02/2022
"""
from collections import defaultdict
from copy import deepcopy
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List

import graph_tool.all as gt
import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from network_utils.network_converter import NetworkConverter
from network_utils.network_generator import NetworkGenerator
from network_utils.network_reader import NetworkReader
from network_utils.network_stats import NetworkStats
from simulators.meta_simulator import MetaSimulator


@pytest.fixture(scope="session")
def create_networkx() -> nx.Graph:
    """
    Create a well-known network with attributes.

    Since it is well-known we know how the outcome of different
    functions need to be!
    """
    network = nx.Graph()
    network.add_edge(0, 1)
    network.add_edge(0, 2)
    network.add_edge(0, 4)
    network.add_edge(1, 2)
    network.add_edge(2, 3)
    network.add_edge(3, 4)

    # set nodes attributes fitness
    network.nodes[0]["fitness"] = 10
    network.nodes[1]["fitness"] = 8
    network.nodes[2]["fitness"] = 6
    network.nodes[3]["fitness"] = 3
    network.nodes[4]["fitness"] = 11

    # set nodes attributes age
    network.nodes[0]["age"] = 30
    network.nodes[1]["age"] = 35
    network.nodes[2]["age"] = 20
    network.nodes[3]["age"] = 60
    network.nodes[4]["age"] = 4

    # set nodes attributes state
    network.nodes[0]["state"] = "c"
    network.nodes[1]["state"] = "h"
    network.nodes[2]["state"] = "h"
    network.nodes[3]["state"] = "w"
    network.nodes[4]["state"] = "w"

    return network


@pytest.fixture(scope="session")
def network_stats_obj(create_networkx: nx.Graph) -> NetworkStats:
    """Create a NetworkStats object.

    Thereby the create_networkx is converted to nk
    """
    conv_graph = NetworkConverter.nx_to_nk(create_networkx)
    network_stats = NetworkStats(conv_graph)
    return network_stats


@pytest.fixture(scope="session")
def scale_free_network() -> nk.Graph:
    """Return a nk scale-free network."""
    return NetworkGenerator.generate_barabasi_albert(n_nodes=5000, n_edges=200)


@pytest.fixture(scope="session")
def random_network() -> nk.Graph:
    """Return a nk random network."""
    return NetworkGenerator.generate_random(n_nodes=300)


@pytest.fixture(scope="function")
def gt_network() -> gt.Graph:
    """Return the montagna_calls network as gt."""
    nx_network = NetworkReader().get_data("montagna_calls")
    gt_network = NetworkConverter.nx_to_gt(nx_network)
    return gt_network


@pytest.fixture(scope="function")
def meta_simulator_network() -> gt.Graph:
    """Return the graph from the MetaSimulator obj.

    In comparison to the gt_network_function, it will return
    the complete network with honest and wolf nodes
    """
    np.random.seed(0)
    meta_sim = MetaSimulator(
        "montagna_calls", ratio_honest=0.8, attachment_method="preferential"
    )
    return meta_sim.network


@pytest.fixture(scope="function")
def meta_simulator() -> gt.Graph:
    """Return the graph from the MetaSimulator obj.

    In comparison to the gt_network_function, it will return
    the complete network with honest and wolf nodes
    """
    np.random.seed(0)
    meta_sim = MetaSimulator(
        "montagna_calls", ratio_honest=0.3, attachment_method="preferential"
    )
    return deepcopy(meta_sim)


@pytest.fixture(scope="function")
def bigger_gt_network(random_network: nk.Graph) -> gt.Graph:
    """Return a bigger gt network."""
    gt_network = NetworkConverter.nk_to_gt(random_network)
    return gt_network


@pytest.fixture(scope="function")
def create_gt_network(create_networkx: nx.Graph) -> gt.Graph:
    """Return the known networkx graph."""
    return NetworkConverter.nx_to_gt(create_networkx)


@pytest.fixture(scope="session")
def create_gt_network_session(create_networkx: nx.Graph) -> gt.Graph:
    """Return the known networkx graph.

    This function is run once. Thus the returned object might
    be manipulated when used in other functions since it refers
    to the same address.
    """
    return NetworkConverter.nx_to_gt(create_networkx)


@pytest.fixture(scope="session")
def fake_topological_data() -> Dict[str, DefaultDict[str, List[Any]]]:
    """Create fake topological data."""
    # create fake data
    fake_data = {}
    for key in ["preferential", "small-world", "random"]:
        data_collector = defaultdict(list)  # type: DefaultDict[str, Any]

        x = np.linspace(0, 30)
        data_collector["mean_security_efficiency"] = list(
            np.exp(0.34) * np.exp(0.09 * x) * np.random.normal(0, 10)
        )
        data_collector["m_security_efficiency"] = np.random.random((10, 10))

        data_collector["mean_information"] = list(
            np.exp(0.34) * np.exp(0.09 * x) * np.random.normal(0, 10)
        )
        data_collector["m_information"] = np.random.random((10, 10))

        data_collector["mean_gcs"] = list(
            np.exp(0.34) * np.exp(0.09 * x) * np.random.normal(0, 10)
        )
        data_collector["m_gcs"] = np.random.random((10, 10))

        data_collector["mean_iteration"] = x
        data_collector["std_security_efficiency"] = list(
            np.random.normal(1, 50, size=len(x))
        )
        data_collector["sem_security_efficiency"] = list(
            np.random.normal(1, 50, size=len(x))
        )
        data_collector["std_information"] = list(np.random.normal(1, 50, size=len(x)))
        data_collector["sem_information"] = list(np.random.normal(1, 50, size=len(x)))

        data_collector["std_gcs"] = list(np.random.normal(1, 50, size=len(x)))
        data_collector["sem_gcs"] = list(np.random.normal(1, 50, size=len(x)))

        fake_data[key] = data_collector

    return fake_data


@pytest.fixture(scope="session")
def fake_correlation_data() -> Dict[str, DefaultDict[str, List[Any]]]:
    """Create fake correlation data."""
    fake_data = {}

    for key in ["preferential", "small-world", "random"]:
        # create fake data
        # The desired covariance matrix.
        num_samples = 20000
        # The desired mean values of the sample.
        mu = np.array([5.0, 0.0, 10.0])
        r = np.array([[3.40, -2.75, -2.00], [-2.75, 5.50, 1.50], [-2.00, 1.50, 1.25]])
        # Generate the random samples.
        rng = np.random.default_rng()
        y = rng.multivariate_normal(mu * np.random.uniform(), r, size=num_samples)

        data_collector = defaultdict(list)  # type: DefaultDict[str, Any]

        data_collector["df_total"] = pd.DataFrame(
            {
                "criminal_likelihood": y[:, 0],
                "degree": y[:, 1],
                "betweenness": y[:, 2],
                "katz": y[:, 2],
                "closeness": y[:, 2],
                "eigen vector": y[:, 2],
            }
        )

        fake_data[key] = data_collector

    return fake_data


@pytest.fixture(scope="session")
def fake_link_sensitivity_analysis() -> Dict[str, Dict[str, List[Any]]]:
    """Create fake correlation data."""
    fake_data = {
        "preferential": {},
        "random": {},
        "small-world": {},
    }  # type: Dict[str, Dict[Any, Any]]
    n_links = np.linspace(6, 40, 10, dtype=int)

    for structure in fake_data.keys():
        for link in n_links:
            fake_data[structure][link] = np.random.normal(3, 2.5, 30)
    return fake_data


@pytest.fixture(scope="session")
def fake_phase_diag_data() -> Dict[str, Dict[str, List[Any]]]:
    """Create fake phase_diag data."""
    fake_phase_diag = {
        "case_1": {
            "param_y": "gamma",
            "y_range": np.linspace(0, 1, 10),
            "x_range": np.linspace(0, 40, 10),
            "param_x": "r_c",
            "grid_status": np.random.choice(
                ["mean_ratio_criminal", "mean_ratio_wolf", "mean_ratio_honest"], 100
            ).reshape(10, 10),
            "grid_value": np.random.random((10, 10)),
        },
        "case_2": {
            "param_y": "gamma",
            "y_range": np.linspace(0, 1, 10),
            "x_range": np.linspace(0, 40, 10),
            "param_x": "r_c",
            "grid_status": np.random.choice(
                ["mean_ratio_criminal", "mean_ratio_wolf", "mean_ratio_honest"], 100
            ).reshape(10, 10),
            "grid_value": np.random.random((10, 10)),
        },
    }
    fake_meta_phase_diag = {
        "preferential": fake_phase_diag,
        "small-world": fake_phase_diag,
    }

    return fake_meta_phase_diag


def pytest_deselected(items: List) -> None:
    """Print deselected tests."""
    if not items:
        return
    config = items[0].session.config
    reporter = config.pluginmanager.getplugin("terminalreporter")
    reporter.ensure_newline()
    for item in items:
        reporter.line(f"deselected: {item.nodeid}", yellow=True, bold=True)
