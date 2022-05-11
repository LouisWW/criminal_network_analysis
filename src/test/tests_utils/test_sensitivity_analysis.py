"""This file test if the SensitivityAnalyser works correctly."""
from unittest import main
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from utils.sensitivity_analysis import SensitivityAnalyser


class TestSensitivityAnalyser:
    """Test the different functions of the SensitivityAnalyser."""

    @pytest.mark.essential
    def test_initialisation(self) -> None:
        """Test if the plotter class is initiated correctly."""
        sa = SensitivityAnalyser()
        assert isinstance(sa, SensitivityAnalyser), "Init didn't work correctly"

    @pytest.mark.essential
    @patch("src.utils.sensitivity_analysis.open")
    def test_sim_mart_vaq_sa(self, mock_open_file: Mock) -> None:
        """Test with a small sample the SA on the Mart-Vaq simulation."""
        # Define the search space
        problem = {
            "num_vars": 2,
            "names": ["tau", "ratio_wolf"],
            "bounds": [[0, 1], [0.1, 0.28]],
        }
        sa = SensitivityAnalyser()
        # Define network
        sa.args.read_data = "montagna_calls"
        sa.args.save = True
        sa.args.n_samples = 10

        sobol_indices = sa.sim_mart_vaq_sa(
            output_value="ratio_criminal", n_samples=4, rounds=10, problem=problem
        )

        assert any(sobol_indices["S1"]), "List contains only zeros/False"
        assert any(sobol_indices["ST"]), "List contains only zeros/False"

        # To mock the saving part
        # Check if the right dir and right contnent is saved
        assert (
            "/results/data/sensitivity_analysis/sim_mart_vaq_sa_ratio_criminal_"
            in mock_open_file.call_args[0][0]
        )
        assert "ST" in mock_open_file.return_value.mock_calls[1].args[0]


if __name__ == "__main__":
    main()
