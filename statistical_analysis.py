import numpy as np
from scipy.stats import rankdata, ranksums, ttest_rel
from tabulate import tabulate


def prepare_results_for_wilcoxon(results):
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

    return raw_advantage, advantage


def calculate_significance(raw_p_value, alfa, algos):
    raw_significance = np.zeros((len(algos), len(algos)))
    raw_significance[raw_p_value <= alfa] = 1
    names_column = np.expand_dims(np.array(algos), axis=1)
    significance = tabulate(np.concatenate(
        (names_column, raw_significance), axis=1), algos)

    return raw_significance, significance


def calculate_stat_better(raw_significance, raw_advantage, algos):
    names_column = np.expand_dims(np.array(algos), axis=1)
    raw_stat_better = raw_significance * raw_advantage
    stat_better = tabulate(np.concatenate(
        (names_column, raw_stat_better), axis=1), algos)

    return raw_stat_better, stat_better

def print_results_wilcoxon(results, algos):
    names_column = np.expand_dims(np.array(algos), axis=1)
    results_table = np.expand_dims(np.mean(results, axis=0), axis=1)
    results_table_formated = np.transpose(np.concatenate((names_column, results_table), axis=1))
    results_table_formated = tabulate(results_table_formated, floatfmt=".2f")

    print("\n", results_table_formated, "\n")

def calculate_global_wilcoxon_analysis(results, name):
    print("\n\n\n################################# Test Wilcoxona dla wszystkich zbiorow (base:", str(name), ") #################################")

    alfa = .05

    algos, meanResults = prepare_results_for_wilcoxon(results)

    print_results_wilcoxon(meanResults, algos)

    ranks = calculate_ranks(meanResults)
    mean_ranks = np.mean(ranks, axis=0)
    # print("\nRanks:\n", ranks)
    # print("\nMean ranks:\n", mean_ranks)

    raw_w_statistic, raw_p_value = calculate_wilcoxon_test(ranks)
    # w_statistic, p_value = format_wilcoxon_test_tables(raw_w_statistic, raw_p_value, algos)
    # print("\nw-statistic:\n", w_statistic, "\n\np-value:\n", p_value)

    raw_advantage, advantage = calculate_advantage(raw_w_statistic, algos)
    # print("\nAdvantage:\n", advantage)

    raw_significance, significance = calculate_significance(raw_p_value, alfa, algos)
    # print("\nStatistical significance (alpha = 0.05):\n", significance)

    raw_stat_better, stat_better = calculate_stat_better(raw_significance, raw_advantage, algos)
    print("\nStatistically significantly better:\n", stat_better)

def calculate_global_wilcoxon_analysis_with_basesplit(results):
    # Forgive me my future self for I have sinned. This should never be done like that, but I need to deliver it fast.
    # This is hardcoded, change this method when using something different from 3 ensembles x 3 bases
    results_0 = {}
    results_1 = {}
    results_2 = {}
    i = 0
    for algo in results.keys():
        j = i % 3
        match j:
            case 0:
                results_0[algo] = results[algo]
            case 1:
                results_1[algo] = results[algo]
            case 2:
                results_2[algo] = results[algo]
        i += 1

    name_0 = (str(list(results_0.keys())[0]).split('-'))[1]  # don't try to understand it, just kill me for that
    name_1 = (str(list(results_1.keys())[0]).split('-'))[1]
    name_2 = (str(list(results_2.keys())[0]).split('-'))[1]

    # calculate_global_wilcoxon_analysis(results, "ALL (just for reference)")
    calculate_global_wilcoxon_analysis(results_0, name_0)
    calculate_global_wilcoxon_analysis(results_1, name_1)
    calculate_global_wilcoxon_analysis(results_2, name_2)

def calculate_tstudent_test(results):
    t_statistic = np.zeros((len(results), len(results)))
    p_value = np.zeros((len(results), len(results)))

    for i in range(len(results)):
        for j in range(len(results)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(results[i], results[j])

    return t_statistic, p_value


def format_tstudent_test_tables(raw_t_statistic, raw_p_value, algos):
    names_column = np.expand_dims(np.array(algos), axis=1)
    t_statistic = np.concatenate((names_column, raw_t_statistic), axis=1)
    t_statistic = tabulate(t_statistic, algos, floatfmt=".2f")
    p_value = np.concatenate((names_column, raw_p_value), axis=1)
    p_value = tabulate(p_value, algos, floatfmt=".2f")

    return t_statistic, p_value

def print_results_tstudent(results, algos):
    names_column = np.expand_dims(np.array(algos), axis=1)
    results_table = np.expand_dims(np.mean(results, axis=1), axis=1)
    results_table_formated = np.transpose(np.concatenate((names_column, results_table), axis=1))
    results_table_formated = tabulate(results_table_formated, floatfmt=".2f")

    print("\n", results_table_formated, "\n")

def calculate_tstudent_analysis_for_dataset(nameOfDataset, datasetResults, algos):
    print("\n\n\n################################# Test Tstudenta dla zbioru ", str(nameOfDataset), " #################################")
    alfa = .05

    print_results_tstudent(datasetResults, algos)

    raw_t_statistic, raw_p_value = calculate_tstudent_test(datasetResults)
    # print("t-statistic:\n", raw_t_statistic, "\n\np-value:\n", raw_p_value)
    # t_statistic, p_value = format_tstudent_test_tables(raw_t_statistic, raw_p_value, algos)
    # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    raw_advantage, advantage = calculate_advantage(raw_t_statistic, algos)
    # print("\nAdvantage:\n", advantage)

    raw_significance, significance = calculate_significance(raw_p_value, alfa, algos)
    # print("\nStatistical significance (alpha = 0.05):\n", significance)

    raw_stat_better, stat_better = calculate_stat_better(raw_significance, raw_advantage, algos)
    print("Statistically significantly better:\n", stat_better)

def calculate_tstudent_analysis_for_all_datasets(results, name):
    print("\n\n\n################################# Test Tstudenta dla kazdego zbioru (base:", str(name), ") #################################")
    algos = list(results.keys())
    numberOfDatasets = len(results[algos[0]])

    for i in range(numberOfDatasets):
        numberOfFolds = len(results[algos[0]][i])
        datasetResults = np.zeros((len(algos), numberOfFolds))

        j = 0
        for algo in algos:
            for k in range(numberOfFolds):
                datasetResults[j][k] = results[algo][i][k]
            j += 1

        calculate_tstudent_analysis_for_dataset(i, datasetResults, algos)

def calculate_tstudent_analysis_for_all_datasets_with_basesplit(results):
    # Forgive me my future self for I have sinned. This should never be done like that, but I need to deliver it fast.
    # This is hardcoded, change this method when using something different from 3 ensembles x 3 bases
    results_0 = {}
    results_1 = {}
    results_2 = {}
    i = 0
    for algo in results.keys():
        j = i % 3
        match j:
            case 0:
                results_0[algo] = results[algo]
            case 1:
                results_1[algo] = results[algo]
            case 2:
                results_2[algo] = results[algo]
        i += 1

    name_0 = (str(list(results_0.keys())[0]).split('-'))[1]  # don't try to understand it, just kill me for that
    name_1 = (str(list(results_1.keys())[0]).split('-'))[1]
    name_2 = (str(list(results_2.keys())[0]).split('-'))[1]

    # calculate_tstudent_analysis_for_all_datasets(results, "ALL (just for reference)")
    calculate_tstudent_analysis_for_all_datasets(results_0, name_0)
    calculate_tstudent_analysis_for_all_datasets(results_1, name_1)
    calculate_tstudent_analysis_for_all_datasets(results_2, name_2)
