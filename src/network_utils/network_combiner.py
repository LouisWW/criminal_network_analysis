"""This script's intention is to combined a given network with a synthetical network.

__author__ = Louis Weyland
__date__   = 6/02/2022
"""

import graph_tool.all as gt
import networkx as nx

class NetworkCombiner:
    """Combines two network together.
    
    In other words, it creates/attach nodes to an existing network.
    """
    
    def __init__(self):
        """Init parameters"""
        
      
    @staticmethod  
    def combine_by_prefertial_attachment(network : nx.Graph) -> nx.Graph:
        
        