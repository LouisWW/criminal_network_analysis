"""Test if the MetaSimulator is running correctly."""
from unittest import main

import graph_tool.all as gt
import numpy as np
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
            k=5,
        )
        network = meta_sim.initialise_network(network=gt_network, k=5)

        # Criminal network size should be 95
        # Honest and Wolf ratio should be within a range given the init is stochastic
        assert (
            len(gt.find_vertex(network, network.vp.status, "c")) == 95
        ), "Criminal ratio not correct"
        assert (
            38 - 15 <= len(gt.find_vertex(network, network.vp.status, "h")) <= 38 + 15
        ), "Honest ratio not correct"
        assert (
            57 - 15 <= len(gt.find_vertex(network, network.vp.status, "w")) <= 57 + 15
        ), "Wolf ratio not correct"

    @pytest.mark.essential
    def test_create_population(self) -> None:
        """Checks if the create populatio creates different graph each time."""
        # Keep the ratio small so the test will be faster
        # by adding less nodes
        ratio_honest = 0.3
        ratio_wolf = 0.1

        # Test all the different attachment_method
        meta_simulator_pref = MetaSimulator(
            "montagna_calls",
            "preferential",
            ratio_honest,
            ratio_wolf,
        )
        meta_simulator_rand = MetaSimulator(
            "montagna_calls",
            "random",
            ratio_honest,
            ratio_wolf,
        )
        meta_simulator_sw = MetaSimulator(
            "montagna_calls", "small-world", ratio_honest, ratio_wolf, k=5
        )
        population_1_pref = meta_simulator_pref.create_population(
            meta_simulator_pref.criminal_network
        )
        population_2_pref = meta_simulator_pref.create_population(
            meta_simulator_pref.criminal_network
        )
        population_1_rand = meta_simulator_pref.create_population(
            meta_simulator_rand.criminal_network
        )
        population_2_rand = meta_simulator_pref.create_population(
            meta_simulator_rand.criminal_network
        )
        population_1_sw = meta_simulator_pref.create_population(
            meta_simulator_sw.criminal_network
        )
        population_2_sw = meta_simulator_pref.create_population(
            meta_simulator_sw.criminal_network
        )

        assert not gt.isomorphism(
            population_1_pref, population_2_pref
        ), "Should not be the same network"

        assert not gt.isomorphism(
            population_1_rand, population_2_rand
        ), "Should not be the same network"

        assert not gt.isomorphism(
            population_1_sw, population_2_sw
        ), "Should not be the same network"

    @pytest.mark.essential
    @pytest.mark.parametrize(
        "structure", [("preferential"), ("random"), ("small-world")]
    )
    def test_create_list_of_populations(self, structure: str) -> None:
        """Test if the list of populations are working."""
        ratio_honest = 0.3
        ratio_wolf = 0.1

        # Test all the different attachment_method
        meta_simulator = MetaSimulator(
            "montagna_calls", structure, ratio_honest, ratio_wolf, k=5
        )

        repetition = 10
        candidates = np.random.choice(range(repetition), 10, replace=False)

        list_of_population = meta_simulator.create_list_of_populations(repetition)
        assert not gt.isomorphism(
            list_of_population[candidates[0]], list_of_population[candidates[1]]
        ), "Should not be the same network"

    @pytest.mark.essential
    def test_avg_play(self) -> None:
        """Test if avg_play function is working correctly."""
        ratio_honest = 0.3
        ratio_wolf = 0.1
        rounds = 20

        # Test all the different attachment_method
        meta_simulator_pref = MetaSimulator(
            "montagna_calls",
            "random",
            ratio_honest,
            ratio_wolf,
        )

        data = meta_simulator_pref.avg_play(
            rounds=rounds, repetition=10, ith_collect=1, execute="parallel"
        )

        assert "mean_ratio_criminal" in data.keys(), "Key not found"
        assert "mean_ratio_honest" in data.keys(), "Key not found"
        assert "mean_ratio_wolf" in data.keys(), "Key not found"
        assert "std_ratio_criminal" in data.keys(), "Key not found"
        assert "std_ratio_honest" in data.keys(), "Key not found"
        assert "std_ratio_wolf" in data.keys(), "Key not found"

        assert len(data["mean_ratio_criminal"]) != 0, "Key not found"
        assert len(data["mean_ratio_honest"]) != 0, "Key not found"
        assert len(data["mean_ratio_wolf"]) != 0, "Key not found"
        assert len(data["std_ratio_criminal"]) != 0, "Key not found"
        assert len(data["std_ratio_honest"]) != 0, "Key not found"
        assert len(data["std_ratio_wolf"]) != 0, "Key not found"

        sim_1 = data["0"]["ratio_criminal"]
        sim_2 = data["2"]["ratio_criminal"]
        assert not np.array_equal(
            np.array(sim_1), np.array(sim_2)
        ), "Two simulations were identical...."


if __name__ == "__main__":
    main()
