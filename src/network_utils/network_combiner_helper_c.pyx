"""This script's intention is to speed up the random attachment

__author__ = Louis Weyland
__date__   = 21/06/2022
"""
import cython
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
