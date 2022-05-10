"""This script contains all the functions for performing some statistics.

__author__ = Louis Weyland
__date__ = 10/09/2022
"""
from typing import Any
from typing import DefaultDict
from typing import List
from typing import Union

import numpy as np


def get_mean_std_over_list(
    data_collector: DefaultDict,
) -> DefaultDict[str, Union[DefaultDict, List[Any]]]:
    """Take the average and mean over multiple lists from a defaultdict(list).

    Args:
        data_collector (DefaultDict): Contains all the data,each main key respresent the
                                        repetition number. In other words, the data is
                                        averaged over all the repetition.

    Returns:
        DefaultDict[Union[int, str], Union[DefaultDict, List[Any]]]:
                            Returns next to the data also the mean and float for each data
    """
    try:
        for k in data_collector.keys():
            if k.isalpha():
                raise RuntimeError(
                    "Sorry, only numeric keys allowed, for example: round '0','1','2',..."
                )
    except RuntimeError:
        print("Wrong keys were given to the function.")
        raise

    repetition = len(data_collector.keys())
    for key in data_collector["0"].keys():
        m = np.zeros((repetition, len(data_collector["0"][key])))
        for i in range(0, repetition):
            # Matrix repetition x rounds
            m[i, :] = data_collector[str(i)][key]
        # Get mean and std
        data_collector["mean_" + key] = np.mean(m, axis=0)
        data_collector["std_" + key] = np.std(m, axis=0)

    return data_collector
