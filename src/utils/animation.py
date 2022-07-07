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
        self.k = 4
        self.m = 2
        self.prob = 0.01

        self.count = 0

        # To color the vertices
        self.color_map = {"c": (1, 0, 0, 1), "h": (0, 0, 1, 1), "w": (0, 1, 0, 1)}

    def create_animation(self) -> None:
        """Simulate the adding of the node."""
        self.win = Gtk.OffscreenWindow()
        self.win.set_default_size(500, 400)
        self.pos = gt.sfdp_layout(self.network)
        self.color_code = self.network.new_vertex_property("vector<double>")
        self.network.vertex_properties["state_color"] = self.color_code
        for v in self.network.vertices():
            self.color_code[v] = self.color_map[
                self.network.vertex_properties["state"][v]
            ]
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
                self.accepted_edges.sort(key=lambda y: y[1])
            elif self.args.attach_meth == "random":
                self.accepted_edges = random_attachment_c(
                    self.n_nodes + self.new_nodes, self.new_nodes, self.prob
                )
                self.accepted_edges.sort(key=lambda y: y[1])
            elif self.args.attach_meth == "small-world":
                self.accepted_edges = combine_by_small_world_attachment_helper(
                    self.n_nodes + self.new_nodes, self.new_nodes, self.k, self.prob
                )
                self.accepted_edges.sort(key=lambda y: y[1])

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
        try:
            v_x, v_y = next(self.accepted_edges)
            try:
                # Check if new node exist
                self.network.add_edge(v_x, v_y, add_missing=False)
            except ValueError:
                # Otherwise add it, define its position and status and color the
                # node
                self.network.add_edge(v_x, v_y, add_missing=True)
                self.pos[self.network.vertex(v_y)] = (
                    np.random.choice(
                        [np.random.uniform(-20, -10), np.random.uniform(10, 20)]
                    ),
                    np.random.choice(
                        [np.random.uniform(-20, -10), np.random.uniform(10, 20)]
                    ),
                )
                self.network.vp.state[self.network.vertex(v_y)] = np.random.choice(
                    ["w", "h"], p=[0.6, 0.4]
                )
                self.color_code[self.network.vertex(v_y)] = self.color_map[
                    self.network.vertex_properties["state"][self.network.vertex(v_y)]
                ]

            self.win.graph.regenerate_surface()
            self.win.graph.queue_draw()
            self.pixbuf = self.win.get_pixbuf()
            self.pixbuf.savev(
                f"{self.savig_dir}/{self.args.attach_meth}+{self.count}.png",
                "png",
                [],
                [],
            )
            self.count += 1
        except StopIteration:
            logger.info("All the nodes with their edges have been added")
            # Create the gif and delete the png
            os.system(
                f"convert -delay 20 -loop 0 results/video/*.png results/video/{self.args.attach_meth}.gif"
            )
            os.system("rm results/video/*.png")
            sys.exit(0)

        return True
