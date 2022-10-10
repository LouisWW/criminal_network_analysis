"""This script test the functions form stats.py.

__author__ = Louis
__date__ = 10/05/2022
"""
from collections import defaultdict
from typing import DefaultDict
from typing import List
from unittest import main
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
from utils.stats import compare_time_series
from utils.stats import dict_mean
from utils.stats import get_mean_std_over_list


class TestStats:
    """Class for unit tests for the stats functions."""

    @pytest.mark.essential
    def test_get_mean_std_over_list(self) -> None:
        """Test if the mean and std returned is correct."""
        data_collector = defaultdict(
            list,
            {
                "0": {
                    "ratio_honest": [0, 2, 4, 6, 8],
                    "ratio_wolf": [3, 4, 5, 1, 2],
                    "network_katz": [4.4, 7, 10],
                },
                "1": {
                    "ratio_honest": [0, 4, 10, 16, 18],
                    "ratio_wolf": [3, -6, 5, 5, 7],
                    "network_katz": [4, 8, 11],
                },
                "2": {
                    "ratio_honest": [0, 3, 40, 68, 7],
                    "ratio_wolf": [13, 4, 5, 11, 22],
                    "network_katz": [4, 6, 12],
                },
            },
        )

        new_data_collector = get_mean_std_over_list(data_collector)

        assert (
            np.array(new_data_collector["mean_ratio_honest"])
            == np.array([0, 3, 18, 30, 11])
        ).all(), "Mean is not computed correctly"
        assert (
            np.isclose(
                new_data_collector["std_ratio_wolf"],
                [4.71, 4.71, 0, 4.10, 8.49],
                rtol=1e-02,
                atol=1e-03,
            )
        ).all(), "Mean is not computed correctly"
        assert (
            np.isclose(
                new_data_collector["std_network_katz"],
                [0.188, 0.81, 0.81],
                rtol=1e-02,
                atol=1e-03,
            )
        ).all(), "Mean is not computed correctly"

    @pytest.mark.essential
    @patch("builtins.print")
    def test_compare_time_series(
        self,
        mocked_print: Mock,
        fake_topological_data: DefaultDict[str, DefaultDict[str, List[int]]],
    ) -> None:
        """Test if the time-series comparison/anova test is working."""
        compare_time_series(fake_topological_data)
        assert (
            "('preferential', 'small-world')                    : pcm"
            in mocked_print.mock_calls[1].args[0]
        )

    def test_dict_mean(self) -> None:
        """Test if the dict mean function is working correctly."""
        dicts = [
            {"X": 5, "value": 200},
            {"X": -2, "value": 100},
            {"X": 3, "value": 400},
        ]
        value = dict_mean(dicts)
        assert value["X"] == 2.0
        assert pytest.approx(value["value"]) == 233.333333


if __name__ == "__main__":
    main()
