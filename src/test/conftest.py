"""
This script's contains all the pytest fixtures.

The fixtures are a piece of code which can be used
for multiple tests as setup.
By doing so we can avoid to setup time-consuming lines of code
for multiple tests


__author__ = Louis Weyland
__date__   = 23/02/2022
"""
import networkx as nx
import pytest


@pytest.fixture(scope="session")
def create_networkx():
    """
    Create a well-known network with attributes.

    Since it is well-known we know how the outcome of different
    functions need to be!
    """
    network = nx.Graph()
    network.add_edge(1, 2)
    network.add_edge(1, 3)
    network.add_edge(1, 5)
    network.add_edge(2, 3)
    network.add_edge(3, 4)
    network.add_edge(4, 5)

    # set nodes attributes fitness
    network.nodes[1]["fitness"] = 10
    network.nodes[2]["fitness"] = 8
    network.nodes[3]["fitness"] = 6
    network.nodes[4]["fitness"] = 3
    network.nodes[5]["fitness"] = 11

    # set nodes attributes age
    network.nodes[1]["age"] = 30
    network.nodes[2]["age"] = 35
    network.nodes[3]["age"] = 20
    network.nodes[4]["age"] = 60
    network.nodes[5]["age"] = 4

    return network
