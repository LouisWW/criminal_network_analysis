"""This script's intention is to test some snippets of code.

__author__ = Louis Weyland
__date__   = 18/03/2022
"""
import os
import timeit

from network_generator import NetworkGenerator
from network_stats import NetworkStats
from network_stats_c import NetworkStats as NetworkStatsC

network_generator = NetworkGenerator()
network = network_generator.generate_barabasi_albert(n_nodes=1000)

networkstats = NetworkStats(network)
networkstatsc = NetworkStatsC()

print("Network ",networkstats.get_overview())
print("Networkc ",networkstatsc.get_overview(network))

print("Network ",networkstats.get_radius())
print("Networkc ",networkstatsc.get_radius(network))

"""

def fib_w_o(n: int = 50) -> int:
        if n <= 1:
            return n
        else:
            return fib_w_o(n - 2) + fib_w_o(n - 1)

py = timeit.timeit('networkstatsc.fib()',\
    setup="from network_stats_c import NetworkStats as NetworkStatsC;\
    networkstatsc = NetworkStatsC()",number=200)


cy = timeit.timeit('networkstatsc.fib_c()',\
    setup="from network_stats_c import NetworkStats as NetworkStatsC;\
        networkstatsc = NetworkStatsC()",number=200)

without_opt  = timeit.timeit('fib_w_o()',number=200)


print(cy, py, without_opt)
print('Cython is {}x faster'.format(py/cy))


py = timeit.timeit(
    "NetworkConverter().nx_to_gt(create_networkx)",
    setup="from utils.graph_converter import NetworkConverter;\
        import networkx as nx;create_networkx = nx.barabasi_albert_graph(10000, 3)",
    number=10,
)
cy = timeit.timeit(
    "nx_to_gt(create_networkx)",
    setup="from nx_to_gt_c import nx_to_gt; \
        import networkx as nx;create_networkx = nx.barabasi_albert_graph(10000, 3)",
    number=10,
)


print(
    cy,
    py,
)
print(f"Cython is {py / cy}x faster")

"""