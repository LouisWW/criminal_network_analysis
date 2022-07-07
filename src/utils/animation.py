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
from network_utils.network_extractor import NetworkExtractor
from network_utils.network_reader import NetworkReader
from simulators.meta_simulator import MetaSimulator
from simulators.sim_mart_vaq import SimMartVaq

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

        # network and metasimulator
        # Get actual criminal network
        nx_network = NetworkReader().get_data(self.args.read_data)
        logger.info(f"The data used is {nx_network.name}")

        # Convert to gt.Graph
        self.network = NetworkConverter.nx_to_gt(nx_network)
        self.network_dummy = deepcopy(self.network)
        self.meta_sim = MetaSimulator(
            network_name=nx_network.name,
            attachment_method=self.args.attach_meth,
            ratio_honest=0.3,
            ratio_wolf=0.3,
        )
        self.simulator = SimMartVaq(
            network=self.meta_sim.network,
            delta=0.4,  # no acting for wolfs
            gamma=0.1,
            tau=0.8,  # no fitness sharing between wolf to criminal
            beta_s=1,
            beta_h=1,
            beta_c=15,
            c_c=10,  # no benefits from criminals/ they still act
            r_c=10,
            c_w=1,
            r_w=1,
            r_h=1,
            temperature=10,
            mutation_prob=0.0001,  # only fermi function
        )

        self.n_nodes = self.network.num_vertices()
        self.new_nodes = 50
        self.k = 4
        self.m = 2
        self.prob = 0.01
        self.count = 0
        self.rounds = 600
        # To color the vertices
        self.color_map = {"c": (1, 0, 0, 1), "h": (0, 0, 1, 1), "w": (0, 1, 0, 1)}

    def create_animation(self) -> None:
        """Simulate the adding of the node."""
        self.win = Gtk.OffscreenWindow()
        self.win.set_default_size(500, 400)

        if self.args.animate_attachment_process:
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
            self.pos = gt.sfdp_layout(self.simulator.network)
            self.color_code = self.simulator.network.new_vertex_property(
                "vector<double>"
            )
            self.simulator.network.vertex_properties["state_color"] = self.color_code
            for v in self.simulator.network.vertices():
                self.color_code[v] = self.color_map[
                    self.simulator.network.vertex_properties["state"][v]
                ]

            if self.args.animate_simulation == "filtered":
                NetworkExtractor.filter_criminal_network(self.simulator.network)
            self.win.graph = gt.GraphWidget(
                self.simulator.network,
                self.pos,
                vertex_fill_color=self.simulator.network.vertex_properties[
                    "state_color"
                ],
            )
            self.win.add(self.win.graph)

            if self.args.animate_simulation == "filtered":
                NetworkExtractor.un_filter_criminal_network(self.simulator.network)
            # Bind the function above as an 'idle' callback.
            GLib.idle_add(self.do_animation_simulation)

        # We will give the user the ability to stop the program by closing the window.
        self.win.connect("delete_event", Gtk.main_quit)
        # Actually show the window, and start the main loop.
        self.win.show_all()
        Gtk.main()

    def do_animation_simulation(self) -> Literal[True]:
        """Play the simulation one round each time."""
        self.simulator.network, _ = self.simulator.play(
            network=self.simulator.network,
            rounds=1,
            n_groups=1,
            ith_collect=1,
            measure_topology=False,
        )

        # Update the color
        for v in self.simulator.network.vertices():
            self.color_code[v] = self.color_map[
                self.simulator.network.vertex_properties["state"][v]
            ]

        if self.args.animate_simulation == "filtered":
            NetworkExtractor.filter_criminal_network(self.simulator.network)

        self.win.graph.regenerate_surface()
        self.win.graph.queue_draw()
        self.pixbuf = self.win.get_pixbuf()
        self.pixbuf.savev(
            f"{self.savig_dir}/simulation+{self.args.animate_simulation}+{self.args.attach_meth}+{self.count}.png",
            "png",
            [],
            [],
        )

        if self.args.animate_simulation == "filtered":
            NetworkExtractor.un_filter_criminal_network(self.simulator.network)
        self.count += 1

        if self.count > self.rounds:
            logger.info("All the nodes with their edges have been added")
            # Create the gif and delete the png
            os.system(
                f"convert -delay 20 -loop 0 results/video/*.png \
                results/video/simulation_{self.args.animate_simulation}_{self.args.attach_meth}.gif"
            )
            os.system("rm results/video/*.png")
            sys.exit(0)

        return True

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
                f"convert -delay 20 -loop 0 results/video/*.png \
                results/video/{self.args.attach_meth}.gif"
            )
            os.system("rm results/video/*.png")
            sys.exit(0)

        return True
