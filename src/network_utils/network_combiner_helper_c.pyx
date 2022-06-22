"""This script's intention is to speed up the random attachment

__author__ = Louis Weyland
__date__   = 21/06/2022
"""
import cython
import random
import itertools

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
cpdef list random_attachment_c(int network_size,int n_new_nodes,float prob):
    """Generate a Erdös-Rény Random Network around the given network.

        The code is based on the pseudo-code described in
        https://www.frontiersin.org/articles/10.3389/fncom.2011.00011/full
    """
    cdef:
        int i
        int j
        list possible_links

    possible_links = []
    for i in range(0,network_size- n_new_nodes):
        for j in range(network_size-n_new_nodes,network_size):
            if prob > drand48():
                possible_links.append((i,j))

    for i in range(network_size- n_new_nodes,network_size):
        for j in range(i+1,network_size):
            if prob > drand48():
                possible_links.append((i,j))

    return possible_links


cdef extern from "stdlib.h":
    double drand48()


@cython.wraparound(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
cpdef list combine_by_small_world_attachment_helper(int network_size,int new_nodes,int k,float prob):
    """Generate a Erdös-Rény Random Network around the given network.

        The code is based on the pseudo-code described in
        https://www.frontiersin.org/articles/10.3389/fncom.2011.00011/full
    """
    cdef:
        int i
        int j
        list possible_links

    possible_links = []
    # First create a ring lattice
    # The criminal network with th new nodes
    for i in range(0,network_size-new_nodes):
        for j in range(network_size-new_nodes+i+1,network_size-new_nodes+i+int(k/2)):
                if j > network_size:
                    break
                possible_links.append((i,j))

    # New nodes amongst them
    for i in range(network_size-new_nodes,network_size):
        for j in range(i+1,i+int(k/2)):
                if j > network_size:
                    break
                possible_links.append((i,j))

    # Second rewire edges randomly with probability pw
    for i in range(0,network_size-new_nodes):
        for j in range(network_size-new_nodes+i+1,network_size-new_nodes+i+int(k/2)):
                if j > network_size:
                    break

                if prob > drand48():
                    to_exclude =[item for item in possible_links if item[0] == i or item[1] == i]
                    to_exclude = list(set(itertools.chain(*to_exclude)))
                    new_candidate= random.choice(list(set([x for x in range(network_size-new_nodes, network_size)]) - set(to_exclude)))
                    # rewire
                    possible_links.append((i,new_candidate))
                    # delete old
                    try:
                        possible_links.remove((i,j))
                    except:
                        possible_links.remove((j,i))

    for i in range(network_size-new_nodes,network_size):
        for j in range(i+1,i+int(k/2)):
                if j > network_size:
                    break

                if prob > drand48():
                    to_exclude =[item for item in possible_links if item[0] == i or item[1] == i]
                    to_exclude = list(set(itertools.chain(*to_exclude)))
                    new_candidate= random.choice(list(set([x for x in range(network_size-new_nodes, network_size)]) - set(to_exclude)))
                    # rewire
                    possible_links.append((i,new_candidate))
                    # delete old
                    try:
                        possible_links.remove((i,j))
                    except:
                        possible_links.remove((j,i))

    return possible_links
