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
            help="""Defines if the network should be visualized.
                            c = circular network,
                            n = normal/random""",
        )

        parser.add_argument(
            "-read-data",
            type=str,
            default=None,
            nargs="?",
            const="cunha",
            choices=["cunha", "montagna_calls", "montagna_meetings"],
            help="""Defines which network to read; cunha, montagna_meetings, montagna_calls.""",
        )

        parser.add_argument(
            "-sim-mart-vaq",
            action="store_true",
            help="""Defines if the simulation based on Martiez-Vaquero is run.""",
        )

        parser.add_argument(
            "-sa",
            "--sensitivity-analysis",
            type=str,
            default=None,
            nargs="?",
            const="sim-mart-vaq",
            choices=["sim-mart-vaq"],
            help="""Defines to run a sensitivity analysis on one of the choices.""",
        )

        parser.add_argument(
            "-n-samples",
            type=int,
            default=15,
            help="""Defines the sampling number for the saltelli method.(default: %(default)s) """,
        )

        parser.add_argument(
            "-output-value",
            type=str,
            default=None,
            help="""Defines on which output value to focus for the sensitivity analysis.""",
        )

        parser.add_argument(
            "-save",
            action="store_true",
            help="""Defines if the results should be saved.""",
        )

        parser.add_argument("-verbose", action="store_true", help="Print extra info")

        # compile the flags
        self.args = parser.parse_known_args()[0]
