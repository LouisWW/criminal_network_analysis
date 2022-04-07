"""
This script's intention is to properly load the various data from the data/ folder.

__author__ = Louis Weyland
__date__   = 5/02/2022
"""
import os

import networkx as nx
import pandas as pd


class NetworkReader:
    """The NetworkReader reads the data from various files and return a networkx graph."""

    def __init__(self) -> None:
        """Set the directory right."""
        # Get current directory
        path = os.getcwd()
        par_dir = os.path.abspath(path)
        self.directory = par_dir + "/data/"

    def read_cunha(self) -> nx.Graph:
        """Get data from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6214327/ ."""
        data = pd.read_csv(
            self.directory + "Cunha2018.csv",
            delimiter=";",
            names=["vertex1", "vertex2"],
        )
        graph_obj = nx.Graph(list(zip(data["vertex1"], data["vertex2"])))
        return graph_obj

    def read_montagna_meetings(self) -> nx.Graph:
        """Get data from https://zenodo.org/record/3938818#.Yf64mPso9FE ."""
        data = pd.read_csv(
            self.directory + "Montagna_Meetings_Edgelist.csv", sep="\\s+"
        )

        graph_obj = nx.from_pandas_edgelist(
            data, source="Source", target="Target", edge_attr=["Weight"]
        )
        return graph_obj

    def read_montagna_phone_calls(self) -> nx.Graph:
        """Get data from https://zenodo.org/record/3938818#.Yf64mPso9FE ."""
        data = pd.read_csv(
            self.directory + "Montagna_Phone_Calls_Edgelist.csv", sep=","
        )

        graph_obj = nx.from_pandas_edgelist(
            data, source="Source", target="Target", edge_attr=["Weight"]
        )
        return graph_obj
