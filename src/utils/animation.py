"""This script's contains the animation object.

The animation is based on the code examples from
https://graph-tool.skewed.de/static/doc/demos/animation/animation.html


__author__ = Louis Weyland
__date__   = 6/72/2022
"""
import logging
import os.path
from typing import List
from typing import Tuple

from network_utils.network_combiner_helper_c import (
    combine_by_small_world_attachment_helper,
)
from network_utils.network_combiner_helper_c import random_attachment_c
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from src.config.config import ConfigParser

logger = logging.getLogger("logger")


class Animateur(ConfigParser):
    """Create a video of the network initialisation and simulation."""

    def __init__(self) -> None:
        """Set the save path and network."""
        super().__init__()

        path = os.path.dirname(os.path.realpath(__file__))
        par_dir = os.path.abspath(os.path.join(path, "../"))
        # par_dir = ../src/
        self.savig_dir = par_dir + "/results/video/"

        # network
        # Get actual criminal network
        nx_network = NetworkReader().get_data(self.args.network_name)
        logger.info(f"The data used is {nx_network.name}")

        # Convert to gt.Graph
        self.network = NetworkConverter.nx_to_gt(nx_network)
        self.n_nodes = self.network.num_vertices()
        self.new__nodes = 50
        self.prob = 0.3
        self.k = 6

    def animate_attachment_process(self, attach_meth: str) -> None:
        """Create a video of the attachment process."""
        if attach_meth == "preferential":
            pass
        elif attach_meth == "random":
            accepted_edges = random_attachment_c(
                self.n_nodes, self.new_nodes, self.prob
            )
            self.do_animation(accepted_edges)

        elif attach_meth == "small-world":
            accepted_edges = combine_by_small_world_attachment_helper(
                self.n_nodes, self.new__nodes, self.k, self.prob
            )
            self.do_animation(accepted_edges)

    def do_animation(self, accepted_edges: List[Tuple[int, int]]) -> None:
        """Simulate the adding of the node."""
        pass
