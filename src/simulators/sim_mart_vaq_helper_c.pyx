"""This file contains function writen in cython.

The idea is to shift intense loops on c to increase the speed

__date__ = 18/04/2022
"""

import numpy as np
import random


cpdef dict divide_network_fast_loop(dict network_dict, int n_groups, int network_size):
    """Divides the network into groups and return a dict.

    The dict contains the node as key and group number as value
    """
    cdef:
        list key_to_del
        dict nodes_status
        int node
        int neighbour
        int k
        list nodes

    nodes_status = dict.fromkeys(range(0,network_size))

    # Set the seed, number needs to start from 1 because default is 0
    for v,group_number in zip(np.random.choice(list(range(0,network_size)), n_groups,),range(1,n_groups+1)):
        nodes_status[v] = group_number

    while len(network_dict) > 0:
        key_to_del = []
        # random order in order to avoid an group to grow too much
        nodes = list(network_dict.keys())
        random.shuffle(nodes)
        for node in nodes:
            # if node has group number
            if nodes_status[node] != None:
                neighbours = network_dict[node]
                for neighbour in neighbours:
                    if nodes_status[neighbour] == None:
                        nodes_status[neighbour] = nodes_status[node]
                # del key since all neighbours have a group number
                key_to_del.append(node)

        if len(key_to_del) == 0 :
            break
        else:
            for k in key_to_del:
                del network_dict[k]

    return nodes_status
