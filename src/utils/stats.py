"""This script contains all the functions for performing some statistics.

__author__ = Louis Weyland
__date__ = 10/09/2022
"""
import itertools
from copy import deepcopy
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import similaritymeasures
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def get_mean_std_over_list(
    data_collector: DefaultDict,
) -> DefaultDict[str, Union[DefaultDict, List[Any]]]:
    """Take the average and st/sem over multiple lists from a defaultdict(list).

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

    repetition = len([s for s in data_collector.keys() if s.isdigit()])
    keys = list(data_collector["0"].keys())
    if "df" in keys:
        keys.remove("df")
    for key in keys:
        m = np.zeros((repetition, len(data_collector["0"][key])))
        for i in range(0, repetition):
            # Matrix repetition x rounds
            m[i, :] = data_collector[str(i)][key]
        # Get mean and std
        data_collector["mean_" + key] = np.mean(m, axis=0)
        data_collector["std_" + key] = np.std(m, axis=0)
        data_collector["sem_" + key] = stats.sem(m, axis=0)
        data_collector["m_" + key] = m

    return data_collector


def concat_df(
    data_collector: DefaultDict, rounds: int
) -> DefaultDict[str, Union[DefaultDict, List[Any]]]:
    """Concates the different pandasDataFrame to one.

    Args:
        data_collector (DefaultDict): Contains all the data,each main key respresent the
                                        repetition number. In other words, the data is
                                        averaged over all the repetition.

    Returns:
        DefaultDict[Union[int, str], Union[DefaultDict, List[Any]]]:
                            Returns next to the data also the mean and float for each data
    """
    # Get the number of repetition
    repetition = len([s for s in data_collector.keys() if s.isdigit()])

    # Check if the df is not empty
    if len(data_collector["0"]["df"]) > 0:
        list_of_dfs = []
        for i in range(0, repetition):
            list_of_dfs.append(data_collector[str(i)]["df"])
        df_total = pd.concat(list_of_dfs)
        data_collector["df_total"] = df_total
        # divide the criminal_likelihood by the number of rounds
        data_collector["df_total"]["criminal_likelihood"] = data_collector["df_total"][
            "criminal_likelihood"
        ].div(rounds)

    return data_collector


def get_correlation(x: List[Union[float, int]], y: List[Union[float, int]]) -> str:
    """Get the correlation between x and y."""
    # only look at the criminal_likelihood
    corr, p_val = pearsonr(x, y)

    p_star = "".join(["*" for t in [0.01, 0.05, 0.1] if p_val <= t])
    corr_with_p = corr.round(2).astype(str) + p_star
    return corr_with_p


def bootstrapping(data: np.ndarray) -> np.ndarray:
    """Return a bootstrapping of the matrix data.

    Data needs to be a 2 dimensional matrix
    """
    shuffle_array = deepcopy(data)
    for row in range(0, shuffle_array.shape[0]):
        np.random.shuffle(shuffle_array[row, :].reshape((-1, 4)))
    return shuffle_array


def compare_time_series(
    whole_data: DefaultDict[str, DefaultDict[str, List[Any]]]
) -> None:
    """Compare the time series by fitting a model and perform a anova-test on it."""
    attachment_methods = list(whole_data.keys())
    attachment_methods_comb = list(itertools.combinations(attachment_methods, 2))
    metrics = [
        metric
        for metric in list(whole_data[attachment_methods[0]].keys())
        if metric.startswith("mean_")
        if "fitness" not in metric
        if "ratio" not in metric
    ]

    for metric in metrics:
        print("----" + metric + "----")
        for method in ["pcm", "frechet_dist", "area_between_two_curves", "dtw"]:
            for comb in attachment_methods_comb:

                time_serie_a = np.zeros((len(whole_data[comb[0]][metric]), 2))
                time_serie_a[:, 0] = whole_data[comb[0]][metric]
                time_serie_a[:, 1] = whole_data[comb[0]]["mean_iteration"]

                time_serie_b = np.zeros((len(whole_data[comb[1]][metric]), 2))
                time_serie_b[:, 0] = whole_data[comb[1]][metric]
                time_serie_b[:, 1] = whole_data[comb[1]]["mean_iteration"]

                if method == "pcm":
                    print(
                        f"{str(comb):50} : {'pcm':12} \
                        {similaritymeasures.pcm(time_serie_a,time_serie_b)}"
                    )
                elif method == "frechet_dist":
                    print(
                        f"{str(comb):50} : {'frechet_dist':12} \
                            {similaritymeasures.frechet_dist(time_serie_a,time_serie_b)}"
                    )
                elif method == "area_between_two_curves":
                    print(
                        f"{str(comb):50} : {'area':12} \
                            {similaritymeasures.area_between_two_curves(time_serie_a,time_serie_b)}"
                    )
                elif method == "dtw":
                    print(
                        f"{str(comb):50} : {'dtw':12} \
                            {similaritymeasures.dtw(time_serie_a,time_serie_b)[0]}"
                    )

                # https://stats.stackexchange.com/questions/156864/comparing-2-sets-of-longitudinal-data
                series_1, series_2 = comb
                final_diff = []
                for i in range(0, 10000):
                    series_1_boot_samp_1 = bootstrapping(
                        data=whole_data[series_1][metric.replace("mean", "m")]
                    )
                    series_1_boot_samp_2 = bootstrapping(
                        data=whole_data[series_1][metric.replace("mean", "m")]
                    )

                    series_2_boot_samp_1 = bootstrapping(
                        data=whole_data[series_2][metric.replace("mean", "m")]
                    )

                    # average aboslute differences
                    aad = np.abs(
                        np.mean(series_1_boot_samp_1) - np.mean(series_2_boot_samp_1)
                    )
                    var = np.abs(
                        np.mean(series_1_boot_samp_1) - np.mean(series_1_boot_samp_2)
                    )

                    # Final Diff" can be interpreted to be the final difference between
                    # the AAD of series 1 and series 2 after accounting for natural
                    # variation in series 1.
                    final_diff.append(aad - var)

                print(
                    f"Bootstrapping :  {str(comb):50}, value {np.percentile(final_diff, 1)}"
                )

        print(30 * "-")

        # stats f_oneway functions takes the groups as input and returns F and P-value
        fvalue, pvalue = stats.f_oneway(
            whole_data["preferential"][metric.replace("mean", "m")].flatten(),
            whole_data["random"][metric.replace("mean", "m")].flatten(),
            whole_data["small-world"][metric.replace("mean", "m")].flatten(),
        )
        print(
            f"Results of ANOVA test:\nThe F-statistic is: {fvalue}\nThe p-value is: {pvalue}"
        )

        df_pref = pd.DataFrame(
            {
                "score": whole_data["preferential"][
                    metric.replace("mean", "m")
                ].flatten(),
                "group": "preferential",
            }
        )
        df_rand = pd.DataFrame(
            {
                "score": whole_data["random"][metric.replace("mean", "m")].flatten(),
                "group": "random",
            }
        )
        df_sw = pd.DataFrame(
            {
                "score": whole_data["small-world"][
                    metric.replace("mean", "m")
                ].flatten(),
                "group": "small-world",
            }
        )

        df = pd.concat([df_pref, df_rand, df_sw])
        # perform Tukey's test
        tukey = pairwise_tukeyhsd(endog=df["score"], groups=df["group"], alpha=0.05)
        print(tukey)
        print("\n")


def dict_mean(dict_list: Dict) -> Dict[str, Union[int, float]]:
    """Return an average/std value of a list of dictionaries."""
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key + "_mean"] = np.mean([d[key] for d in dict_list], axis=0)
        mean_dict[key + "_std"] = 1.96 * np.std([d[key] for d in dict_list], axis=0)

    return mean_dict
