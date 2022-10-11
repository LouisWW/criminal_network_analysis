"""
This script's intention is generate json which includes np.arrays.

More specifically, generated data is visualized.

__author__ = Louis Weyland
__date__   = 14/09/2022
"""
import json
from typing import Any
from typing import Union

import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    """Convert numpy to list for json encoding."""

    def default(self, obj: Any) -> Union[np.integer, np.floating, np.ndarray, Any]:
        """Encode the numpy array to the respective type."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)
