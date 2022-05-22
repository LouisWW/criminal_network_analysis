"""This script's intention is to test some snippets of code.

__author__ = Louis Weyland
__date__   = 18/03/2022
"""
import timeit


setup = """
from simulators.meta_simulator import MetaSimulator
from simulators.sim_mart_vaq import SimMartVaq
meta_sim = MetaSimulator(
        network_name='montagna_calls', ratio_honest=0.9, ratio_wolf=0.01
    )
simulator = SimMartVaq(meta_sim.network)
"""

setup_c = """
from simulators.meta_simulator import MetaSimulator
from simulators.sim_mart_vaq_faster import SimMartVaq as SimMartVaqC
meta_sim = MetaSimulator(
        network_name='montagna_calls', ratio_honest=0.9, ratio_wolf=0.01
    )
simulator_c = SimMartVaqC(meta_sim.network)
"""

cy = timeit.timeit(
    """simulator.play(network=simulator.network, rounds= 200, n_groups= 20)""",
    setup=setup,
    number=10,
)
py = timeit.timeit(
    """simulator_c.play(network=simulator_c.network, rounds= 200, n_groups= 20)""",
    setup=setup_c,
    number=10,
)

print(cy, py)
print(f"Cython is {py / cy}x faster")
