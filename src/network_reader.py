"""
This script's intention is to properly load the various data from
the data/ folder

__author__ = Louis Weyland
__date__   = 5/02/2022
"""
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class NetworkReader:
    """
    The NetworkReader reads the data from various files which have
    various structures and returns an a readable structure for the
    networkit
    """

    def __init__(self):
        """set the directory right"""
        self.directory = os.path.dirname(os.path.realpath(__file__)) + "/data/"

    def read_cunha(self):
        """
        Reads the data from the following paper
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6214327/
        """
        data = pd.read_csv(
            self.directory + "Cunha2018.csv",
            delimiter=";",
            names=["vertex1", "vertex2"],
        )
        graph_obj = nx.Graph(list(zip(data["vertex1"], data["vertex2"])))
        return graph_obj

    def read_montagna_meetings(self):
        """
        Reads the data from the following paper
        https://zenodo.org/record/3938818#.Yf64mPso9FE
        Pieces of code has been taken from the according github repository
        https://github.com/lcucav/networkdisruption
        """
        data = pd.read_csv(
            self.directory + "Montagna_Meetings_Edgelist.csv", sep="\\s+"
        )

        graph_obj = nx.from_pandas_edgelist(
            data, source="Source", target="Target", edge_attr=["Weight"]
        )
        return graph_obj

    def read_montagna_phone_calls(self):
        """
        Reads the data from the following paper
        https://zenodo.org/record/3938818#.Yf64mPso9FE
        Pieces of code has been taken from the according github repository
        https://github.com/lcucav/networkdisruption
        """
        data = pd.read_csv(
            self.directory + "Montagna_Phone_Calls_Edgelist.csv", sep="\\s+"
        )

        graph_obj = nx.from_pandas_edgelist(
            data, source="Source", target="Target", edge_attr=["Weight"]
        )
        return graph_obj


if __name__ == "__main__":

    network_reader = NetworkReader()
    print(network_reader.directory)
    network_obj = network_reader.read_cunha()
    nx.draw(network_obj)
    plt.show()
