"""Test if the plotting function is working correctly."""
from collections import defaultdict
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
        data_collector = defaultdict(list)
        data_collector["honest_ratio"] = np.random.rand(200)
        data_collector["criminal_ratio"] = np.random.rand(200)
        data_collector["wolf_ratio"] = np.random.rand(200)

        ax = plotter.plot_lines(
            dict_data=data_collector,
            data_to_plot=["honest_ratio", "wolf_ratio"],
            xlabel="rounds",
            ylabel="ratio (per)",
            title="This is a test",
        )

        assert isinstance(ax, plt.Axes)
        assert plt.show(), "An error occurred with plt.show()"

    @pytest.mark.essential
    def test_plot_lines_wrong_key(self) -> None:
        """Test if the plot_lines function raises error with wrong key."""
        plotter = Plotter()

        # create fake data
        data_collector = defaultdict(list)
        data_collector["honest_ratio"] = np.random.rand(200)
        data_collector["criminal_ratio"] = np.random.rand(200)
        data_collector["wolf_ratio"] = np.random.rand(200)

        with pytest.raises(Exception):
            plotter.plot_lines(
                dict_data=data_collector,
                data_to_plot=["honest_ratio", "ratio"],
                xlabel="rounds",
                ylabel="ratio (per)",
                title="This is a test",
            )
