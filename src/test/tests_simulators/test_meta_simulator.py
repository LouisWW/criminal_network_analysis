"""Test if the MetaSimulator is running correctly."""
from unittest import main

import graph_tool.all as gt
import pytest
from simulators.meta_simulator import MetaSimulator


class TestMetaSimualtor:
    """Class for unit tests for MetaSimulator."""

    @pytest.mark.essential
    def test_meta_simulator(self, gt_network: gt.Graph) -> None:
        """Test if the initialization works."""
        org_size = gt_network.num_vertices()
        """Test if the initialization works."""
        # Keep the ratio small so the test will be faster
        # by adding less nodes
        ratio_honest = 0.3
        ratio_wolf = 0.1

        meta_simulator = MetaSimulator(
            "montagna_calls",
            "random",
            ratio_honest,
            ratio_wolf,
        )

        # Test if the obj is init correctly
        assert isinstance(
            meta_simulator, MetaSimulator
        ), "MetaSimulator hasn't been init correctly"

        # Test if the ratio is caluclated correctly
        assert meta_simulator.ratio_criminal == 0.6, "Ratio is wrong."
        assert (
            meta_simulator.n_criminal == org_size
        ), "Determined number of criminals is wrong."
        assert meta_simulator.total_number_nodes == 158, "Ratio is wrong"
        assert meta_simulator.new_nodes == 63, "Number of nodes to add is wrong"
        assert (
            round(meta_simulator.relative_ratio_honest, 2) == 0.75
        ), "Relative ratio is wrong"
        assert (
            round(meta_simulator.relative_ratio_wolf, 2) == 0.25
        ), "Relative ratio is wrong"

    @pytest.mark.essential
    def test_meta_sim_wrong_init(self) -> None:
        """Test if a wrong init triggers the assert statements."""
        # With ratio_honest == 0
        with pytest.raises(Exception):
            assert MetaSimulator(
                "montagna_calls", ratio_honest=0, attachment_method="preferential"
            )

        # With ratio_wolf == 1.1
        with pytest.raises(Exception):
            assert MetaSimulator(
                "montagna_calls", ratio_wolf=1.1, attachment_method="small-wolrd"
            )

        # With ratios not adding up more than 1
        with pytest.raises(Exception):
            assert MetaSimulator(
                "montagna_calls",
                ratio_honest=0.8,
                ratio_wolf=0.3,
                attachment_method="small-wolrd",
            )
            assert MetaSimulator(
                "montagna_calls",
                ratio_honest=0.8,
                ratio_wolf=0.2,
                attachment_method="small-wolrd",
            )
            assert MetaSimulator(
                "montagna_calls",
                ratio_honest=-0.8,
                ratio_wolf=0.1,
                attachment_method="small-wolrd",
            )
            assert MetaSimulator(
                "montagna_calls",
                ratio_honest=-0.8,
                ratio_wolf=0.2,
                attachment_method="small-wolrd",
            )

    @pytest.mark.essential
    def test_initialise_network(self, gt_network: gt.Graph) -> None:
        """Test if the init process is done correctly.

        More precisely tests if the ratio of c/h/w is correct
        """
        meta_sim = MetaSimulator(
            "montagna_calls",
            ratio_honest=0.2,
            ratio_wolf=0.3,
            attachment_method="small-world",
        )
        network = meta_sim.initialise_network(network=gt_network)

        # Criminal network size should be 95
        # Honest and Wolf ratio should be within a range given the init is stochastic
        assert (
            len(gt.find_vertex(network, network.vp.state, "c")) == 95
        ), "Criminal ratio not correct"
        assert (
            38 - 15 <= len(gt.find_vertex(network, network.vp.state, "h")) <= 38 + 15
        ), "Honest ratio not correct"
        assert (
            57 - 15 <= len(gt.find_vertex(network, network.vp.state, "w")) <= 57 + 15
        ), "Wolf ratio not correct"

    @pytest.mark.essential
    def test_init_fitness_not_rand(self, gt_network: gt.Graph) -> None:
        """Test if the init of the fitness attribute is done correctly.

        Using zero fit for all the nodes.
        """
        meta_sim = MetaSimulator("montagna_calls", attachment_method="random")
        network = meta_sim.init_fitness(gt_network, random_fit=False)
        assert network.vp.fitness, "Fitness attribute doesn't exists..."
        assert not any(network.vp.fitness), "Fitness should all be zero"

    @pytest.mark.essential
    def test_init_fitness_rand(self, gt_network: gt.Graph) -> None:
        """Test if the init of the fitness attribute is done correctly.

        Using radnom fit for all the nodes.
        """
        meta_sim = MetaSimulator("montagna_calls", attachment_method="random")

        network = meta_sim.init_fitness(gt_network, random_fit=True)
        assert network.vp.fitness, "Fitness attribute doesn't exists..."
        assert any(network.vp.fitness), "Fitness should not all be zero"

    @pytest.mark.essential
    def test_init_filtering(self, gt_network: gt.Graph) -> None:
        """Test if the init of the fitness attribute is done correctly."""
        meta_sim = MetaSimulator("montagna_calls", attachment_method="preferential")
        network = meta_sim.init_filtering(gt_network)
        assert network.vp.filtering, "Filtering attribute doesn't exists..."


if __name__ == "__main__":
    main()
