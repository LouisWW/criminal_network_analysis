"""
This script contains the ConfigParser.

The ConfigParser catch all the flags given in the command line
and will create python accessible variables

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
import argparse


class ConfigParser:
    """Catches all the flags."""

    def __init__(self) -> None:
        """Make sure when inherited everything works fine."""
        super().__init__()

        parser = argparse.ArgumentParser(
            prog="CRIMINAL NETWORK ANALYSIS",
            description="Flags needed to run a given Pipeline and generate the desired results",
        )

        parser.add_argument(
            "-draw-network",
            type=str,
            default="None",
            nargs="?",
            const="c",
            choices=["c", "n"],
            help="""Define if the network should be visualized.
                            c = circular network,
                            n = normal/random""",
        )

        parser.add_argument(
            "-execute",
            type=str,
            choices=["parallel", "sequential"],
            help="""Run the simulation in parallel or sequential.""",
        )

        parser.add_argument(
            "-create-population",
            action="store_true",
            help="""Create and save the population (-save flag not needed)""",
        )

        parser.add_argument(
            "-get-network-stats",
            action="store_true",
            help="""Returns the mean characteristics of a population
                    (preferential/random/small-world)""",
        )

        parser.add_argument(
            "-compare-w-rnd-init",
            action="store_true",
            help="""Compare the outcome of the simulations if a random init was used or not.""",
        )

        parser.add_argument(
            "-read-data",
            type=str,
            default=None,
            nargs="?",
            const="montagna_calls",
            choices=["cunha", "montagna_calls", "montagna_meetings"],
            help="""Define which network to read; cunha, montagna_meetings, montagna_calls.""",
        )

        parser.add_argument(
            "-attach-meth",
            type=str,
            default=None,
            nargs="?",
            const="preferential",
            choices=["preferential", "random", "small-world"],
            help="""Define the attachment methods around the criminal network""",
        )

        parser.add_argument(
            "-k",
            type=int,
            default=None,
            help="""Define how many new connection a
                    new node is making while generating a population.""",
        )

        parser.add_argument(
            "-animate-attachment-process",
            action="store_true",
            help="""Create an animation of the attachment process.""",
        )

        parser.add_argument(
            "-sim-mart-vaq",
            action="store_true",
            help="""Define if the simulation based on Martinez-Vaquero is run.
                    Thereby, for each repetition the another network is used.""",
        )

        parser.add_argument(
            "-case",
            type=str,
            choices=["const", "growth", "decline"],
            help="""Define which case is simulated""",
        )

        parser.add_argument(
            "-whole-pipeline",
            action="store_true",
            help="""Run the whole simulation for the different structures in one go.
                    Caution: Might be unstable!!!""",
        )
        parser.add_argument(
            "-criminal-likelihood-corr",
            action="store_true",
            help="""Define if a correlation between criminal and node centrality exists.""",
        )
        parser.add_argument(
            "-sim-mart-vaq-w-net",
            action="store_true",
            help="""Define if the simulation based on Martinez-Vaquero is run.
                    Thereby, for each repetition an new network is created.""",
        )

        parser.add_argument(
            "-animate-simulation",
            type=str,
            default=None,
            nargs="?",
            const="unfiltered",
            choices=["unfiltered", "filtered"],
            help="""Create an animation of the simulation.""",
        )

        parser.add_argument(
            "-sa",
            "--sensitivity-analysis",
            type=str,
            default=None,
            nargs="?",
            const="sim-mart-vaq",
            choices=["sim-mart-vaq"],
            help="""Define to run a sensitivity analysis on one of the choices.""",
        )

        parser.add_argument(
            "--sensitivity-analysis-links",
            action="store_true",
            help="""Define to run a sensitivity analysis on one of the choices.""",
        )

        parser.add_argument(
            "-phase-diag",
            "--phase-diagram",
            action="store_true",
            help="""Create a phase diagram with the defined parameters.""",
        )

        parser.add_argument(
            "-topo-meas",
            action="store_true",
            help="""Define to run a comparative analysis of the different simulations.
                    Thereby, for each repetition the same network is used.""",
        )

        parser.add_argument(
            "-topo-meas-w-net",
            action="store_true",
            help="""Define to run a comparative analysis of the different simulations.
                    Thereby, for each repetition an new network is created.""",
        )
        parser.add_argument(
            "-n-samples",
            type=int,
            default=15,
            help="""Define the sampling number for the saltelli method.(default: %(default)s) """,
        )

        parser.add_argument(
            "-output-value",
            type=str,
            default=None,
            help="""Define on which output value to focus for the sensitivity analysis.""",
        )

        parser.add_argument(
            "-save",
            action="store_true",
            help="""Define if the results should be saved.""",
        )

        parser.add_argument(
            "-plot",
            action="store_true",
            help="""Define if the results should be plotted.ss""",
        )

        parser.add_argument(
            "-r",
            "--rounds",
            type=int,
            default=10,
            help="""Define the numbers of rounds played. Can be applied to SimMartVaq.play and
            SensitivityAnalyser.sim_mart_vaq_sa
            """,
        )

        parser.add_argument(
            "-n-groups",
            type=int,
            default=1,
            help="""Define the number of groups for each round.
            """,
        )
        parser.add_argument(
            "-ratio-honest",
            type=float,
            help="""Define the initial ratio of honest in a population.
            """,
        )

        parser.add_argument(
            "-ratio-wolf",
            type=float,
            help="""Define the initial ratio of wolves in a population.
            """,
        )

        parser.add_argument(
            "-delta",
            type=float,
            help="""Define the influence of criminals on the acting of the wolf (SimMartVaq)
            """,
        )

        parser.add_argument(
            "-tau",
            type=float,
            help="""Influence of wolf's action on criminals (SimMartVaq)""",
        )

        parser.add_argument(
            "-gamma",
            type=float,
            help="""Punishment ratio for the members of a criminal organization  (SimMartVaq)""",
        )

        parser.add_argument(
            "-beta-s", type=int, help="""State punishment value (SimMartVaq)"""
        )

        parser.add_argument(
            "-beta-h", type=int, help="""Civil punishment value (SimMartVaq)"""
        )

        parser.add_argument(
            "-beta-c", type=int, help="""Criminal punishment value (SimMartVaq)"""
        )

        parser.add_argument(
            "-c-w", type=int, help="""Damage caused by wolf (SimMartVaq)"""
        )

        parser.add_argument(
            "-c-c", type=int, help="""Damage caused by criminal (SimMartVaq)"""
        )

        parser.add_argument(
            "-r-w", type=int, help="""Reward ratio for wolf (SimMartVaq)"""
        )

        parser.add_argument(
            "-r-c", type=int, help="""Reward ratio for criminal (SimMartVaq)"""
        )

        parser.add_argument(
            "-r-h", type=int, help="""Reward ratio for honest (SimMartVaq)"""
        )

        parser.add_argument(
            "-temp",
            "--temperature",
            type=int,
            help="""Temperature for the fermi function (SimMartVaq)""",
        )

        parser.add_argument(
            "-mutation-prob", type=float, help="""Mutation probability (SimMartVaq)"""
        )

        parser.add_argument("-verbose", action="store_true", help="Print extra info")

        parser.add_argument(
            "-running-chunk",
            action="store_true",
            help="""Running the analysis in chunk
                                                                    by saving in between""",
        )

        # compile the flags
        self.args = parser.parse_known_args()[0]
