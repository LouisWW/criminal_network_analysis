"""This script's contains the animation object.

The animation is based on the code examples from
https://graph-tool.skewed.de/static/doc/demos/animation/animation.html


__author__ = Louis Weyland
__date__   = 6/72/2022
"""
import logging
import os.path
import sys
from copy import deepcopy
from typing import Literal

import graph_tool.all as gt
import numpy as np
from config.config import ConfigParser
from gi.repository import GLib
from gi.repository import Gtk
from network_utils.network_combiner import NetworkCombiner
from network_utils.network_combiner_helper_c import (
    combine_by_small_world_attachment_helper,
)
from network_utils.network_combiner_helper_c import random_attachment_c
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader

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
        nx_network = NetworkReader().get_data(self.args.read_data)
        logger.info(f"The data used is {nx_network.name}")

        # Convert to gt.Graph
        self.network = NetworkConverter.nx_to_gt(nx_network)
        self.network_dummy = deepcopy(self.network)
        self.n_nodes = self.network.num_vertices()
        self.new_nodes = 50
        self.k = 6
        self.m = 2

        self.count = 0

    def create_animation(self) -> None:
        """Simulate the adding of the node."""
        self.win = Gtk.OffscreenWindow()
        self.win.set_default_size(500, 400)
        self.pos = gt.sfdp_layout(self.network)
        self.network, _ = self.get_color_map(
            self.network, color_vertex_property="state_color"
        )
        self.win.graph = gt.GraphWidget(
            self.network,
            self.pos,
            vertex_fill_color=self.network.vertex_properties["state_color"],
        )
        self.win.add(self.win.graph)

        if self.args.animate_attachment_process:
            if self.args.attach_meth == "preferential":
                (
                    _,
                    self.accepted_edges,
                ) = NetworkCombiner.combine_by_preferential_attachment_faster(
                    self.network_dummy, self.new_nodes, self.m
                )
            elif self.args.attach_meth == "random":
                self.accepted_edges = random_attachment_c(
                    self.n_nodes, self.new_nodes, self.prob
                )
            elif self.args.attach_meth == "small-world":
                self.accepted_edges = combine_by_small_world_attachment_helper(
                    self.n_nodes, self.new_nodes, self.k, self.prob
                )
            else:
                raise KeyError(
                    "Please define which attachment method to use! -attach-meth"
                )

            # Bind the function above as an 'idle' callback.

            self.accepted_edges = iter(self.accepted_edges)
            GLib.idle_add(self.do_animation_attachment_process)

        elif self.args.animate_simulation:
            pass

        # We will give the user the ability to stop the program by closing the window.
        self.win.connect("delete_event", Gtk.main_quit)
        # Actually show the window, and start the main loop.
        self.win.show_all()
        Gtk.main()

    def do_animation_attachment_process(self) -> Literal[True]:
        """Add the nodes listen in the accepted edges."""
        v_x, v_y = next(self.accepted_edges)
        self.network.add_edge(v_x, v_y)
        self.network.vp.state[self.network.vertex(v_y)] = np.random.choice(
            ["w", "h"], p=[0.7, 0.3]
        )
        self.pos[self.network.vertex(v_y)] = (
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20),
        )
        self.win.graph.regenerate_surface()
        self.win.graph.queue_draw()
        self.pixbuf = self.win.get_pixbuf()
        self.pixbuf.savev(
            f"{self.savig_dir}/{self.args.attach_meth}+{self.count}.png", "png", [], []
        )
        if self.count > self.new_nodes:
            sys.exit(0)
        self.count += 1

        return True

    def get_color_map(
        self, network: gt.Graph, color_vertex_property: str = None
    ) -> gt.PropertyMap:
        """Define the color of the vertex based on the vertex property."""
        if color_vertex_property == "state_color":
            # c = red, h = blue, w = green
            color_map = {"c": (1, 0, 0, 1), "h": (0, 0, 1, 1), "w": (0, 1, 0, 1)}
            color_code = network.new_vertex_property("vector<double>")
            network.vertex_properties["state_color"] = color_code
            for v in network.vertices():
                color_code[v] = color_map[network.vertex_properties["state"][v]]
            return network, color_code
