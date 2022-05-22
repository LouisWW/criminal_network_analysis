"""
This script's intention is to get the convert graphs between packages.

It converts graph objects from networkx to networkit
vice versa and networkx to graph-tool.
So as for now to convert networkit graphs to graph-tool,
the graphs need to be firstly converted to networkx.

__author__ = Louis Weyland
__date__   = 14/02/2022
"""
import logging
from typing import Optional
from typing import Tuple
from typing import Union

import graph_tool.all as gt
import networkit as nk
import networkx as nx
import numpy as np
import pyintergraph


logger = logging.getLogger("logger")


def get_prop_type(
    value: Union[int, str, float, bool, dict], key: Optional[str] = None
) -> Tuple[str, Union[float, str, dict], str]:
    """
    Perform typing and value conversion for the graph_tool PropertyMap class.

    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, str):
        # Encode the key as ASCII
        key = str(key)

    # Deal with the value
    if isinstance(value, bool):
        tname = "bool"

    elif isinstance(value, int):
        tname = "float"
        value = float(value)

    elif isinstance(value, float):
        tname = "float"

    elif isinstance(value, str):
        tname = "string"
        value = str(value)

    elif isinstance(value, dict):
        tname = "object"

    else:
        tname = "string"
        value = str(value)

    return tname, value, key


class NetworkConverter:
    """Converting graph object from one package to another."""

    def __init__(self) -> None:
        """No specific info for now."""
        pass

    @staticmethod
    def nx_to_nk(network: nx.graph) -> nk.Graph:
        """Convert graph from networkx to networkit."""
        return nk.nxadapter.nx2nk(network)

    @staticmethod
    def nk_to_nx(network: nk.graph) -> nx.Graph:
        """Convert graph from networkx to networkit."""
        return nk.nxadapter.nk2nx(network)

    @staticmethod
    def nx_to_gt(network: nx.graph) -> gt.Graph:
        """
        Convert a networkx graph to a graph-tool graph.

        Copied from
        https://bbengfort.github.io/2016/06/graph-tool-from-networkx/

        Important notice, nodes start at 1 in graph-tool in comparison
        to networkit/x where node enumeration starts with 0.
        """
        # Phase 0: Create a directed or undirected graph-tool Graph
        gtG = gt.Graph(directed=network.is_directed())

        # Add the Graph properties as "internal properties"
        for key, value in network.graph.items():
            # Convert the value and key into a type for graph-tool
            tname, value, key = get_prop_type(value, key)

            prop = gtG.new_graph_property(tname)  # Create the PropertyMap
            gtG.graph_properties[key] = prop  # Set the PropertyMap
            gtG.graph_properties[key] = value  # Set the actual value

        # Phase 1: Add the vertex and edge property maps
        # Go through all nodes and edges and add seen properties
        # Add the node properties first
        nprops = set()  # cache keys to only add properties once
        for node, data in network.nodes(data=True):

            # Go through all the properties if not seen and add them.
            for key, val in data.items():
                if key in nprops:
                    continue  # Skip properties already added

                # Convert the value and key into a type for graph-tool
                tname, _, key = get_prop_type(val, key)

                prop = gtG.new_vertex_property(tname)  # Create the PropertyMap
                gtG.vertex_properties[key] = prop  # Set the PropertyMap

                # Add the key to the already seen properties
                nprops.add(key)

        # Also add the node id: in NetworkX a node can be any hashable type, but
        # in graph-tool node are defined as indices. So we capture any strings
        # in a special PropertyMap called 'id' -- modify as needed!
        gtG.vertex_properties["id"] = gtG.new_vertex_property("string")

        # Add the edge properties second
        eprops = set()  # cache keys to only add properties once
        for src, dst, data in network.edges(data=True):

            # Go through all the edge properties if not seen and add them.
            for key, val in data.items():
                if key in eprops:
                    continue  # Skip properties already added

                # Convert the value and key into a type for graph-tool
                tname, _, key = get_prop_type(val, key)

                prop = gtG.new_edge_property(tname)  # Create the PropertyMap
                gtG.edge_properties[key] = prop  # Set the PropertyMap

                # Add the key to the already seen properties
                eprops.add(key)

        # Phase 2: Actually add all the nodes and vertices with their properties
        # Add the nodes
        vertices = {}  # vertex mapping for tracking edges later
        for node, data in network.nodes(data=True):

            # Create the vertex and annotate for our edges later
            v = gtG.add_vertex()
            vertices[node] = v

            # Set the vertex properties, not forgetting the id property
            data["id"] = str(node)
            for key, value in data.items():
                gtG.vp[key][v] = value  # vp is short for vertex_properties

        # Add the edges
        for src, dst, data in network.edges(data=True):

            # Look up the vertex structs from our vertices mapping and add edge.
            e = gtG.add_edge(vertices[src], vertices[dst])

            # Add the edge properties
            for key, value in data.items():
                gtG.ep[key][e] = value  # ep is short for edge_properties

        # Done, finally!
        return gtG

    @staticmethod
    def gt_to_nx(network: gt.Graph, labelname: str = "id") -> nx.Graph:
        """Convert graph from graph_tool to networkx."""
        logger.warning("labelname should be set to 'id'!")
        return pyintergraph.gt2nx(network, labelname)

    @staticmethod
    def gt_to_nk(network: gt.Graph) -> nk.Graph:
        """Convert graph from graph_tool to networkit."""
        adj_matrix = gt.adjacency(network)
        col, row = adj_matrix.nonzero()

        # create nk.Graph with number of vertices
        graph = nk.Graph(len(network.get_vertices()))

        for r, c in zip(row, col):
            graph.addEdge(r, c)

        return graph

    @staticmethod
    def nk_to_gt(network: nk.Graph, directed: bool = False) -> gt.Graph:
        """Convert graph from networkit to graph_tool."""
        nk_adj_matrix = nk.algebraic.adjacencyMatrix(network, matrixType="sparse")
        graph = gt.Graph(directed=directed)
        graph.add_edge_list(np.transpose(nk_adj_matrix.nonzero()))

        return graph
