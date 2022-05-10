"""This script test the functions form stats.py.

__author__ = Louis
__date__ = 10/05/2022
"""
from collections import defaultdict

import numpy as np
import pytest
from src.utils.stats import get_mean_std_over_list


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
