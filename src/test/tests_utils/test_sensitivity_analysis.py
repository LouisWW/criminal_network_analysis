"""This file test if the SensitivityAnalyser works correctly."""
import pytest
from src.utils.sensitivity_analysis import SensitivityAnalyser


class TestSensitivityAnalyser:
    """Test the different functions of the SensitivityAnalyser."""

    @pytest.mark.essential
    def test_initialisation(self) -> None:
        """Test if the plotter class is initiated correctly."""
        sa = SensitivityAnalyser()
        assert isinstance(sa, SensitivityAnalyser), "Init didn't work correctly"

    @pytest.mark.essential
    def test_sim_mart_vaq_sa(self) -> None:
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

        sobol_indices = sa.sim_mart_vaq_sa(
            output_value="ratio_criminal", search_space=problem, n_samples=4
        )

        assert any(sobol_indices["S1"]), "List contains only zeros/False"
        assert any(sobol_indices["ST"]), "List contains only zeros/False"
