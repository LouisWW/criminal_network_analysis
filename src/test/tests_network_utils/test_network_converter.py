"""Test if the conversions between graph types are right."""
from unittest import main

import graph_tool.all as gt
import networkit as nk
import networkx as nx
import pytest
from network_utils.network_converter import NetworkConverter


class TestNetworkConverter:
    """Class for unit tests for  NetworkConverter."""

    @pytest.mark.essential
    def test_nx_to_gt(self, create_networkx: nk.Graph) -> None:
        """Test if conversion nx to gt works well.

        Test if all the attributes and connection
        as imported correctly to graph tool
        """
        gt_converted_network = NetworkConverter.nx_to_gt(create_networkx)
        assert isinstance(
            gt_converted_network, gt.Graph
        ), "Graph object is not created properly"

        # test if all attributes are there
        assert gt_converted_network.vp.fitness, "Fitness attribute doesn't exist"
        assert gt_converted_network.vp.age, "Age attribute doesn't exist"

        # get the rid id since networkx starts from 1 and
        # graph-tool stats from 0 and the positions can be
        # switched around
        id_list = list(gt_converted_network.vp.id)

        id_0 = id_list.index("0")
        edge_0 = gt_converted_network.vertex(id_0)
        assert (
            gt_converted_network.vp.id[edge_0] == "0"
        ), "Id is not converted correctly"
        assert (
            gt_converted_network.vp.age[edge_0] == 30
        ), "Attribute was not converted correctly"
        assert (
            gt_converted_network.vp.fitness[edge_0] == 10
        ), "Attribute was not converted correctly"

        id_3 = id_list.index("3")
        edge_3 = gt_converted_network.vertex(id_3)
        assert (
            gt_converted_network.vp.id[edge_3] == "3"
        ), "Id is not converted correctly"
        assert (
            gt_converted_network.vp.age[edge_3] == 60
        ), "Attribute was not converted correctly"
        assert (
            gt_converted_network.vp.fitness[edge_3] == 3
        ), "Attribute was not converted correctly"

        id_4 = id_list.index("4")
        edge_4 = gt_converted_network.vertex(id_4)
        assert (
            gt_converted_network.vp.id[edge_4] == "4"
        ), "Id is not converted correctly"
        assert (
            gt_converted_network.vp.age[edge_4] == 4
        ), "Attribute was not converted correctly"
        assert (
            gt_converted_network.vp.fitness[edge_4] == 11
        ), "Attribute was not converted correctly"

        # get neighbors of node 0 from networkx
        neighbors = gt_converted_network.get_all_neighbors(id_0)
        assert len(neighbors) == 3, "Number of neighbors is not correct"
        for neighbor in neighbors:
            assert gt_converted_network.vp.id[
                gt_converted_network.vertex(neighbor)
            ] in [
                "1",
                "2",
                "4",
            ], "Id is not converted correctly"

        # get neighbors of node 4 from networkx
        neighbors = gt_converted_network.get_all_neighbors(id_4)
        assert len(neighbors) == 2, "Number of neighbors is not correct"
        for neighbor in neighbors:
            assert gt_converted_network.vp.id[
                gt_converted_network.vertex(neighbor)
            ] in [
                "0",
                "3",
            ], "Id is not converted correctly"

    @pytest.mark.essential
    def test_nx_to_nk(self, create_networkx: nk.Graph) -> None:
        """Test if conversion nx to nk works well."""
        nk_converted_network = NetworkConverter.nx_to_nk(create_networkx)

        nx_adj_matrix = nx.adjacency_matrix(create_networkx)
        nk_adj_matrix = nk.algebraic.adjacencyMatrix(
            nk_converted_network, matrixType="sparse"
        )

        assert (
            nk_adj_matrix != nx_adj_matrix
        ).nnz == 0, "Nx to Nk conversion didn't work properly"

    @pytest.mark.essential
    def test_nk_to_nx(self, create_networkx: nk.Graph) -> None:
        """Test if conversion nk to nx works well."""
        nk_converted_network = NetworkConverter.nx_to_nk(create_networkx)
        nx_converted_network = NetworkConverter.nk_to_nx(nk_converted_network)

        nx_adj_matrix = nx.adjacency_matrix(nx_converted_network)
        nk_adj_matrix = nk.algebraic.adjacencyMatrix(
            nk_converted_network, matrixType="sparse"
        )

        assert (
            nk_adj_matrix != nx_adj_matrix
        ).nnz == 0, "Nx to Nk conversion didn't work properly"

    @pytest.mark.essential
    def test_gt_to_nx(self, create_networkx: nx.Graph) -> None:
        """Tests the conversion from gt.Graph to nx.Graph.

        Thereby, it is important to check if the right nodes have the
        right attribute!
        """
        # First convert known graph to gt
        gt_converted_network = NetworkConverter.nx_to_gt(create_networkx)
        assert isinstance(
            gt_converted_network, gt.Graph
        ), "Graph object is not created properly"

        # Convert back to nx
        nx_converted_network = NetworkConverter.gt_to_nx(
            gt_converted_network, labelname="id"
        )

        assert isinstance(
            nx_converted_network, nx.Graph
        ), "Graph object is not created properly"

        # Check if all the attributes have been copied
        assert list(nx_converted_network.nodes["0"].keys()) == [
            "fitness",
            "age",
            "state",
        ], "Not all attributes are copied"

        # Check if the attributes are correct
        assert (
            nx_converted_network.nodes["2"]["age"] == 20
        ), "Attributes are not copied correct"
        assert (
            nx_converted_network.nodes["3"]["fitness"] == 3
        ), "Attributes are not copied correct"
        assert (
            nx_converted_network.nodes["4"]["age"] == 4
        ), "Attributes are not copied correct"

        # Check if the edges are correct
        assert list(nx_converted_network.edges("1")) == [
            ("1", "0"),
            ("1", "2"),
        ], "Edges were not copied correctly"
        assert list(nx_converted_network.edges("0")) == [
            ("0", "1"),
            ("0", "2"),
            ("0", "4"),
        ], "Edges were not copied correclty"

    @pytest.mark.essential
    def test_gt_to_nk(self, create_networkx: nx.Graph) -> None:
        """Tests the conversion from gt.Graph to nk.Graph.

        Since networkit doesn't work with graph attribute, only the edge connection
        is important.
        """
        # First convert known graph to gt
        gt_converted_network = NetworkConverter.nx_to_gt(create_networkx)
        assert isinstance(
            gt_converted_network, gt.Graph
        ), "Graph object is not created properly"

        # Convert gt to nx
        nk_converted_network = NetworkConverter.gt_to_nk(gt_converted_network)
        assert isinstance(
            nk_converted_network, nk.Graph
        ), "Graph object is not created properly"

        assert nk_converted_network.hasEdge(1, 2), "Edges were not copied correctly"
        assert nk_converted_network.hasEdge(3, 4), "Edges were not copied correctly"
        assert nk_converted_network.hasEdge(0, 2), "Edges were not copied correctly"

    @pytest.mark.essential
    def test_nk_to_gt(self, create_networkx: nx.Graph) -> None:
        """Tests the conversion from gt.Graph to nk.Graph.

        Since networkit doesn't work with graph attribute, only the edge connection
        is important.
        """
        # Create networkit graph
        nk_converted_network = NetworkConverter.nx_to_nk(create_networkx)
        nk_adj_matrix = nk.algebraic.adjacencyMatrix(
            nk_converted_network, matrixType="sparse"
        )

        # Create graph_tool graph
        gt_converted_network = NetworkConverter.nk_to_gt(nk_converted_network)
        assert isinstance(
            gt_converted_network, gt.Graph
        ), "Graph object is not created properly"

        gt_adj_matrix = gt.adjacency(gt_converted_network)
        gt_adj_matrix.data -= 1
        assert (
            nk_adj_matrix != gt_adj_matrix
        ).nnz == 0, "Nk to Gt conversion didn't work properly"


if __name__ == "__main__":
    main()
