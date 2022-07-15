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
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List

import graph_tool.all as gt
import networkit as nk
import networkx as nx
import numpy as np
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
    for key in ["preferential attachment", "small world", "random attachment"]:
        data_collector = defaultdict(list)  # type: DefaultDict[str, Any]
        loc, scale = np.random.randint(0, 10), np.random.randint(0, 10)
        data_collector["mean_security_efficiency"] = list(
            np.random.logistic(loc, scale, 10) * np.random.randint(0, 10)
        )
        data_collector["m_security_efficiency"] = np.random.random((10, 10))

        data_collector["mean_information"] = list(np.ones(10))
        data_collector["m_information"] = np.random.random((10, 10))

        data_collector["mean_gcs"] = list(np.ones(10) * 5)
        data_collector["m_gcs"] = np.random.random((10, 10))

        data_collector["mean_iteration"] = list(range(0, 10))
        data_collector["std_security_efficiency"] = list(
            np.random.normal(1, 50, size=10)
        )
        data_collector["std_information"] = list(np.random.normal(1, 50, size=10))
        data_collector["std_gcs"] = list(np.random.normal(1, 50, size=10))

        fake_data[key] = data_collector

    return fake_data


def pytest_deselected(items: List) -> None:
    """Print deselected tests."""
    if not items:
        return
    config = items[0].session.config
    reporter = config.pluginmanager.getplugin("terminalreporter")
    reporter.ensure_newline()
    for item in items:
        reporter.line(f"deselected: {item.nodeid}", yellow=True, bold=True)
