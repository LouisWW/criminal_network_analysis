"""Test if the plotting function is working correctly."""
from collections import defaultdict
from typing import Any
from typing import DefaultDict
from typing import List
from unittest import main
from unittest.mock import Mock
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from utils.plotter import Plotter


class TestPlotter:
    """Class for unit tests for Plotter."""

    @pytest.mark.essential
    def test_initialisation(self) -> None:
        """Test if the plotter class is initiated correctly."""
        plotter = Plotter()
        assert isinstance(plotter, Plotter), "Init didn't work correctly"

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_lines(self, mock_show: Mock) -> None:
        """Test if the plot_lines function is working correctly."""
        plotter = Plotter()

        # create fake data
        data_collector = defaultdict(list)  # type: DefaultDict[str, List[Any]]
        data_collector["honest_ratio"] = list(np.random.rand(200))
        data_collector["criminal_ratio"] = list(np.random.rand(200))
        data_collector["wolf_ratio"] = list(np.random.rand(200))

        ax = plotter.plot_lines(
            dict_data=data_collector,
            y_data_to_plot=["honest_ratio", "wolf_ratio"],
            xlabel="rounds",
            ylabel="ratio (per)",
            title="This is a test",
        )
        assert isinstance(ax, plt.Axes)

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_lines_with_std(self, mock_show: Mock) -> None:
        """Test if the plot_lines function is working correctly."""
        plotter = Plotter()

        # create fake data
        data_collector = defaultdict(list)  # type: DefaultDict[str, List[Any]]
        data_collector["mean_honest_ratio"] = list(np.random.rand(200))
        data_collector["mean_criminal_ratio"] = list(np.random.rand(200))
        data_collector["mean_wolf_ratio"] = list(np.random.rand(200))

        data_collector["std_honest_ratio"] = list(np.random.normal(0.1, 5, size=200))
        data_collector["std_criminal_ratio"] = list(np.random.normal(0.1, 5, size=200))
        data_collector["std_wolf_ratio"] = list(np.random.normal(0.1, 5, size=200))

        ax = plotter.plot_lines(
            dict_data=data_collector,
            y_data_to_plot=["mean_honest_ratio", "mean_criminal_ratio"],
            xlabel="rounds",
            ylabel="ratio (per)",
            title="This is a test",
            plot_std=True,
        )
        assert isinstance(ax, plt.Axes)

    @pytest.mark.essential
    def test_plot_lines_wrong_key(self) -> None:
        """Test if the plot_lines function raises error with wrong key."""
        plotter = Plotter()

        # create fake data
        data_collector = defaultdict(list)  # type: DefaultDict[str, List[Any]]
        data_collector["honest_ratio"] = list(np.random.rand(200))
        data_collector["criminal_ratio"] = list(np.random.rand(200))
        data_collector["wolf_ratio"] = list(np.random.rand(200))

        with pytest.raises(Exception):
            plotter.plot_lines(
                dict_data=data_collector,
                data_to_plot=["honest_ratio", "ratio"],
                xlabel="rounds",
                ylabel="ratio (per)",
                title="This is a test",
            )

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_hist(self, mock_show: Mock) -> None:
        """Test if the plot_hist function is working correctly."""
        plotter = Plotter()

        # create fake data
        data_collector = defaultdict(list)  # type: DefaultDict[str, List[Any]]
        data_collector["honest_ratio"] = list(np.random.normal(size=1000))
        data_collector["criminal_ratio"] = list(np.random.poisson(5, 1000))
        data_collector["wolf_ratio"] = list(np.random.power(5, 1000))

        ax = plotter.plot_hist(
            dict_data=data_collector,
            data_to_plot=["honest_ratio", "wolf_ratio", "criminal_ratio"],
            n_bins=100,
            xlabel="group size",
            ylabel="count",
            title="This is a test",
        )
        assert isinstance(ax, plt.Axes)

    @pytest.mark.essential
    # @patch("matplotlib.pyplot.show")
    def test_plot_lines_comparative(self) -> None:
        """Test if the plotting function is wokring correctly."""
        plotter = Plotter()

        # create fake data
        fake_data = {}
        for key in ["preferential_attachment", "small-world", "random attachment"]:
            data_collector = defaultdict(list)  # type: DefaultDict[str, List[Any]]
            loc, scale = np.random.randint(0, 10), np.random.randint(0, 10)
            data_collector["mean_security_efficiency"] = list(
                np.random.logistic(loc, scale, 10) * np.random.randint(0, 10)
            )
            data_collector["mean_information"] = list(np.ones(10))
            data_collector["mean_gcs"] = list(np.ones(10) * 5)
            data_collector["mean_iteration"] = list(range(0, 10))
            data_collector["std_security_efficiency"] = list(
                np.random.normal(1, 50, size=10)
            )
            data_collector["std_information"] = list(np.random.normal(1, 50, size=10))
            data_collector["std_gcs"] = list(np.random.normal(1, 50, size=10))

            fake_data[key] = data_collector

        ax = plotter.plot_lines_comparative(
            fake_data,
            y_data_to_plot="mean_" + "security_efficiency",
            x_data_to_plot="mean_iteration",
            title="Testing the simulation",
            xlabel="rounds",
            ylabel="security_efficiency",
            plot_std="True",
        )

        ax = plotter.plot_lines_comparative(
            fake_data,
            y_data_to_plot="mean_" + "information",
            x_data_to_plot="mean_iteration",
            title="Testing the simulation",
            xlabel="rounds",
            ylabel="information",
            plot_std="True",
        )

        assert isinstance(ax, plt.Axes)


if __name__ == "__main__":
    main()
