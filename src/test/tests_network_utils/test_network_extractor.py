"""Test if the filtering and resetting works correctly."""
from copy import deepcopy
from unittest import main

import graph_tool.all as gt
import numpy as np
import pytest
from network_utils.network_converter import NetworkConverter
from network_utils.network_extractor import NetworkExtractor
from network_utils.network_reader import NetworkReader
from simulators.meta_simulator import MetaSimulator


class TestNetworkExtractor:
    """Class for unit tests for  NetworkExtractor."""

    @pytest.mark.essential
    def test_filter_criminal_network(self) -> None:
        """Test if only the criminal network is returned."""
        # Get stats about network_obj
        network_name = "montagna_calls"
        nx_network = NetworkReader().get_data(network_name)
        gt_network = NetworkConverter.nx_to_gt(nx_network)
        org_n_size = gt_network.num_vertices()
        org_n_edges = gt_network.num_edges()

        # Adding nodes based on preferential attachment
        meta_sim = MetaSimulator(
            network_name=network_name,
            ratio_honest=0.8,
            ratio_wolf=0.1,
            k=2,
            attachment_method="random",
        )

        meta_sim.network.status = np.asarray(list(meta_sim.network.vp.state))
        filtered_network = NetworkExtractor.filter_criminal_network(meta_sim.network)

        assert (
            filtered_network.num_vertices() == org_n_size
        ), "Number should be the same"
        assert filtered_network.num_edges() == org_n_edges, "Number should be the same"
        assert gt.isomorphism(
            gt_network, filtered_network
        ), "Should be the same network"

    @pytest.mark.essential
    def test_un_filter_criminal_network(self) -> None:
        """Test if the un-filtering approach works correctly."""
        network_name = "montagna_calls"
        meta_sim = MetaSimulator(
            network_name=network_name,
            ratio_honest=0.8,
            ratio_wolf=0.1,
            k=2,
            attachment_method="random",
        )

        org_network = deepcopy(meta_sim.network)
        # ensure it is not the same object
        assert org_network != meta_sim.network
        org_n_size = org_network.num_vertices()
        org_n_edges = org_network.num_edges()

        meta_sim.network.status = np.asarray(list(meta_sim.network.vp.state))
        NetworkExtractor.filter_criminal_network(meta_sim.network)
        assert not gt.isomorphism(
            meta_sim.network, org_network
        ), "Should not be the same network"
        assert org_n_size != meta_sim.network.num_vertices()
        assert org_n_edges != meta_sim.network.num_edges()

        # reset filtering
        NetworkExtractor.un_filter_criminal_network(meta_sim.network)
        assert gt.isomorphism(
            meta_sim.network, org_network
        ), "Should be the same network"
        assert org_n_size == meta_sim.network.num_vertices()
        assert org_n_edges == meta_sim.network.num_edges()


if __name__ == "__main__":
    main()
