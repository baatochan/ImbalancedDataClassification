import numpy as np
from scipy.stats import rankdata, ranksums
from tabulate import tabulate


def prepare_results(results):
    algos = list(results.keys())
    numberOfDatasets = len(results[algos[0]])
    meanResults = np.zeros((numberOfDatasets, len(algos)))

    i = 0
    for algo in algos:
        for j in range(0, numberOfDatasets):
            meanResults[j][i] = np.mean(results[algo][j])
        i += 1

    return algos, meanResults


def calculate_ranks(meanResults):
    ranks = []

    for ms in meanResults:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)

    return ranks


def calculate_wilcoxon_test(ranks):
    numberOfClassifiers = len(ranks[0])

    w_statistic = np.zeros((numberOfClassifiers, numberOfClassifiers))
    p_value = np.zeros((numberOfClassifiers, numberOfClassifiers))

    for i in range(numberOfClassifiers):
        for j in range(numberOfClassifiers):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    return w_statistic, p_value


def format_wilcoxon_test_tables(raw_w_statistic, raw_p_value, algos):
    names_column = np.expand_dims(np.array(algos), axis=1)
    w_statistic = np.concatenate((names_column, raw_w_statistic), axis=1)
    w_statistic = tabulate(w_statistic, algos, floatfmt=".2f")
    p_value = np.concatenate((names_column, raw_p_value), axis=1)
    p_value = tabulate(p_value, algos, floatfmt=".2f")

    return w_statistic, p_value


def calculate_advantage(raw_w_statistic, algos):
    raw_advantage = np.zeros((len(algos), len(algos)))
    raw_advantage[raw_w_statistic > 0] = 1
    names_column = np.expand_dims(np.array(algos), axis=1)
    advantage = tabulate(np.concatenate(
        (names_column, raw_advantage), axis=1), algos)

    return advantage


def calculate_significance(raw_p_value, alfa, algos):
    raw_significance = np.zeros((len(algos), len(algos)))
    raw_significance[raw_p_value <= alfa] = 1
    names_column = np.expand_dims(np.array(algos), axis=1)
    significance = tabulate(np.concatenate(
        (names_column, raw_significance), axis=1), algos)

    return significance


def calculate_analysis(results):
    alfa = .05

    algos, meanResults = prepare_results(results)

    ranks = calculate_ranks(meanResults)
    mean_ranks = np.mean(ranks, axis=0)
    print("\nRanks:\n", ranks)
    print("\nMean ranks:\n", mean_ranks)

    raw_w_statistic, raw_p_value = calculate_wilcoxon_test(ranks)
    w_statistic, p_value = format_wilcoxon_test_tables(raw_w_statistic, raw_p_value, algos)
    print("\nw-statistic:\n", w_statistic, "\n\np-value:\n", p_value)

    advantage = calculate_advantage(raw_w_statistic, algos)
    print("\nAdvantage:\n", advantage)

    significance = calculate_significance(raw_p_value, alfa, algos)
    print("\nStatistical significance (alpha = 0.05):\n", significance)
