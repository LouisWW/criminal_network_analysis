"""Test if the plotting function is working correctly."""
from collections import defaultdict
from typing import Any
from typing import DefaultDict
from typing import Dict
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
        data_collector["rounds"] = list(range(0, 200))

        ax = plotter.plot_lines(
            dict_data={"test": data_collector},
            y_data_to_plot=["honest_ratio", "wolf_ratio"],
            x_data_to_plot="rounds",
            xlabel="rounds",
            ylabel="ratio (per)",
            title="This is a test",
        )
        assert isinstance(ax, (np.ndarray, np.generic))

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_lines_with_std(self, mock_show: Mock) -> None:
        """Test if the plot_lines function is working correctly."""
        plotter = Plotter()

        # create fake data
        data_collector = defaultdict(list)  # type: DefaultDict[str, List[Any]]
        data_collector["mean_honest_ratio"] = list(np.random.rand(2000))
        data_collector["mean_criminal_ratio"] = list(np.random.rand(2000))
        data_collector["mean_wolf_ratio"] = list(np.random.rand(2000))

        data_collector["std_honest_ratio"] = list(np.random.normal(0.1, 5, size=2000))
        data_collector["std_criminal_ratio"] = list(np.random.normal(0.1, 5, size=2000))
        data_collector["std_wolf_ratio"] = list(np.random.normal(0.1, 5, size=2000))

        data_collector["rounds"] = list(range(0, 2000))

        ax = plotter.plot_lines(
            dict_data={"preferential": data_collector, "random": data_collector},
            y_data_to_plot=["mean_honest_ratio", "mean_criminal_ratio"],
            x_data_to_plot="rounds",
            xlabel="rounds",
            ylabel="ratio (per)",
            title=True,
            plot_deviation="std",
        )
        assert isinstance(ax, (plt.Axes, np.ndarray, np.generic))

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
                dict_data={"random": data_collector},
                data_to_plot=["honest_ratio", "ratio"],
                xlabel="rounds",
                ylabel="ratio (per)",
                title="This is a test",
            )

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_hist(
        self,
        mock_show: Mock,
        fake_topological_data: Dict[str, DefaultDict[str, List[Any]]],
    ) -> None:
        """Test if the plot_hist function is working correctly."""
        plotter = Plotter()
        ax = plotter.plot_hist(
            dict_data=fake_topological_data,
            y_data_to_plot=["mean_security_efficiency", "mean_information", "mean_gcs"],
            ylabel=True,
            xlabel=True,
        )
        assert isinstance(ax, (np.ndarray, np.generic))

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_lines_comparative(
        self,
        mock_show: Mock,
        fake_topological_data: Dict[str, DefaultDict[str, List[Any]]],
    ) -> None:
        """Test if the plotting function is working correctly."""
        plotter = Plotter()

        ax = plotter.plot_lines_comparative(
            fake_topological_data,
            y_data_to_plot=["mean_security_efficiency", "mean_information", "mean_gcs"],
            x_data_to_plot="mean_iteration",
            plot_deviation="sem",
        )

        assert isinstance(ax, (plt.Axes, np.ndarray, np.generic))

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_lines_correlation(
        self,
        mock_show: Mock,
        fake_correlation_data: Dict[str, DefaultDict[str, List[Any]]],
    ) -> None:
        """Test if the plt lines correlation function works."""
        plotter = Plotter()
        ax = plotter.plot_lines_correlation(
            dict_data=fake_correlation_data,
            y_data_to_plot=[
                "degree",
                "betweenness",
                "katz",
                "closeness",
                "eigen vector",
            ],
            x_data_to_plot="criminal_likelihood",
        )

        assert isinstance(ax, (np.ndarray, np.generic))

    @pytest.mark.essential
    @patch("matplotlib.pyplot.show")
    def test_plot_lines_correlation_grid(
        self,
        mock_show: Mock,
        fake_correlation_data: Dict[str, DefaultDict[str, List[Any]]],
    ) -> None:
        """Test if the plt lines correlation function works."""
        plotter = Plotter()
        ax = plotter.plot_lines_correlation_grid(
            dict_data=fake_correlation_data,
            y_data_to_plot=[
                "degree",
                "betweenness",
                "katz",
                "closeness",
                "eigen vector",
            ],
            x_data_to_plot="criminal_likelihood",
        )

        assert isinstance(ax, (np.ndarray, np.generic))


if __name__ == "__main__":
    main()
