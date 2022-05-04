"""
This script's contains all the pytest fixtures.

The fixtures are a piece of code which can be used
for multiple tests as setup.
By doing so we can avoid to setup time-consuming lines of code
for multiple tests


__author__ = Louis Weyland
__date__   = 23/02/2022
"""
import graph_tool.all as gt
import networkit as nk
import networkx as nx
import pytest
from src.network_utils.network_converter import NetworkConverter
from src.network_utils.network_generator import NetworkGenerator
from src.network_utils.network_reader import NetworkReader
from src.network_utils.network_stats import NetworkStats


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
    return NetworkGenerator.generate_random(n_nodes=5000)


@pytest.fixture(scope="function")
def gt_network() -> gt.Graph:
    """Return the montagna_calls network as gt."""
    nx_network = NetworkReader().get_data("montagna_calls")
    gt_network = NetworkConverter.nx_to_gt(nx_network)
    return gt_network


@pytest.fixture(scope="function")
def bigger_gt_network(random_network: gt.Graph) -> gt.Graph:
    """Return a bigger gt network."""
    gt_network = NetworkConverter.nx_to_gt(random_network)
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
