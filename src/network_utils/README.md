# network_utils

In the [network_utils](network_utils/) folder, all the code is contained to create/read and analyse the network. Thereby, different libraries are used for each purpose.

* In [network_combiner.py](network_combiner.py), an initial network is used to build around it a population using various methods such as preferential, random or small-world.
***If a new attachment method needs to be added, it is in this file*!**

* In [network_converter.py](network_converter.py), the networks can be converted between networkx, networkit, graph_tool

* In [network_extractor](network_extractor.py), a graph-tool network is given to extract the nodes with status c (member of a criminal network) and also to un-filter the network back to its origin

* In [network_generator.py](netowrk_generator.py) is a prototype file which is not substantial for the project. It allows for generating a specific type of network (random/preferential). ***No populations are created!***

* [network_reader.py](network_reader.py) has all the functions to load the criminal network contained in folder [data](../data/).

* [network_stats.py](network_stats.py) computes the overall characteristics of a network.

* In [node_stats.py](node_stats.py), the centrality of the nodes is computed such as the betweenness and Katz,...
