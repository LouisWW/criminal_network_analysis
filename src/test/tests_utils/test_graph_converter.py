"""Test if the conversions between graph types are right."""
import graph_tool as gt
import networkit as nk
import networkx as nx
import pytest
from utils.graph_converter import NetworkConverter


class TestNetworkConverter:
    """Class for unit tests for  NetworkConverter."""

    @pytest.mark.essential
    def test_nx_to_gt(self, create_networkx):
        """Test if conversion nx to gt works well.

        Test if all the attributes and connection
        as imported correctly to graph tool
        """
        converted_network = NetworkConverter().nx_to_gt(create_networkx)
        assert isinstance(
            converted_network, gt.Graph
        ), "Graph object is not created properly"

        # test if all attributes are there
        assert converted_network.vp.id, "Id attribute doesn't exist"
        assert converted_network.vp.fitness, "Fitness attribute doesn't exist"
        assert converted_network.vp.age, "Age attribute doesn't exist"

        # get the rid id since networkx starts from 1 and
        # graph-tool stats from 0 and the positions can be
        # switched around
        id_list = list(converted_network.vp.id)

        id_1 = id_list.index("1")
        edge_0 = converted_network.vertex(id_1)
        assert converted_network.vp.id[edge_0] == "1", "Id is not converted correctly"
        assert (
            converted_network.vp.age[edge_0] == 30
        ), "Attribute was not converted correctly"
        assert (
            converted_network.vp.fitness[edge_0] == 10
        ), "Attribute was not converted correctly"

        id_4 = id_list.index("4")
        edge_4 = converted_network.vertex(id_4)
        assert converted_network.vp.id[edge_4] == "4", "Id is not converted correctly"
        assert (
            converted_network.vp.age[edge_4] == 60
        ), "Attribute was not converted correctly"
        assert (
            converted_network.vp.fitness[edge_4] == 3
        ), "Attribute was not converted correctly"

        id_5 = id_list.index("5")
        edge_5 = converted_network.vertex(id_5)
        assert converted_network.vp.id[edge_5] == "5", "Id is not converted correctly"
        assert (
            converted_network.vp.age[edge_5] == 4
        ), "Attribute was not converted correctly"
        assert (
            converted_network.vp.fitness[edge_5] == 11
        ), "Attribute was not converted correctly"

        # get neighbors of node 1 from networkx
        neighbors = converted_network.get_all_neighbors(id_1)
        assert len(neighbors) == 3, "Number of neighbors is not correct"
        for neighbor in neighbors:
            assert converted_network.vp.id[converted_network.vertex(neighbor)] in [
                "2",
                "3",
                "5",
            ], "Id is not converted correctly"

        # get neighbors of node 5 from networkx
        neighbors = converted_network.get_all_neighbors(id_5)
        assert len(neighbors) == 2, "Number of neighbors is not correct"
        for neighbor in neighbors:
            assert converted_network.vp.id[converted_network.vertex(neighbor)] in [
                "1",
                "4",
            ], "Id is not converted correctly"

    @pytest.mark.essential
    def test_nx_to_nk(self, create_networkx):
        """Test if conversion nx to nk works well."""
        converted_network = NetworkConverter().nx_to_nk(create_networkx)

        nx_adj_matrix = nx.adjacency_matrix(create_networkx)
        nk_adj_matrix = nk.algebraic.adjacencyMatrix(
            converted_network, matrixType="sparse"
        )

        assert (
            nk_adj_matrix != nx_adj_matrix
        ).nnz == 0, "Nx to Nk conversion didn't work properly"
