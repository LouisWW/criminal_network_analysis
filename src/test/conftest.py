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

    return network
