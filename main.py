import numpy as np

import balance_dataset
import cross_validation
import load_data
import print_helpers
import statistical_analysis

data_dir = "data/"
random_state = 420

# load all the datasets from the data_dir
datasets = load_data.load_data_from_files(data_dir)

# print the content and imbalanceness of loaded datasets
# print_helpers.printAllDatasets(datasets)
# print_helpers.printClassImbalanceForAllDatasets(datasets)

# balance the Xth dataset and print distribution of class to confirm that the set is balanced
# set = 18
# print_helpers.printDataset(set, datasets[set][0], datasets[set][1])
# balancedDataset = balance_dataset.balance_dataset(datasets[set][0], datasets[set][1], random_state)
# print_helpers.printClassImbalanceForDataset(balancedDataset)

# run each of crossvalid function as 5 times repeated 2-fold cross validation and print the results for Xth dataset
# set = 18
# scores = cross_validation.run_every_crossvalid(datasets[set][0], datasets[set][1], 2, 1, random_state)
# print_helpers.printAlgoResults(set, scores)

# load saved results from classifiers analysis or run classifiers analysis if file is not present and save results
try:
    results = np.load('results.npy', allow_pickle=True)
    # print("\nScores:\n", results)
    results = results.item()  # item() is needed to properly load a dict using np
except FileNotFoundError:
    # run every crossvalid as 5 times repeated 2-fold cross validation and print the results for all datasets
    results = cross_validation.run_every_crossvalid_for_every_dataset(datasets, 2, 5, random_state)
    np.save('results', results)

statistical_analysis.calculate_global_wilcoxon_analysis_with_basesplit(results) # item() is needed to properly load a dict using np

statistical_analysis.calculate_tstudent_analysis_for_all_datasets_with_basesplit(results) # item() is needed to properly load a dict using np
