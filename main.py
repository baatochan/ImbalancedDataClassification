import balance_dataset
import cross_validation
import load_data
import print_helpers

data_dir = "data/"
random_state = 420

# load all the datasets from the data_dir
datasets = load_data.load_data_from_files(data_dir)

# print the content and imbalanceness of loaded datasets
# print_helpers.printAllDatasets(datasets)
# print_helpers.printClassImbalanceForAllDatasets(datasets)

# balance the first dataset and print distribution of class to confirm that the set is balanced
# balancedDataset = balance_dataset.balance_dataset(datasets[0][0], datasets[0][1], random_state)
# print_helpers.printClassImbalanceForDataset(balancedDataset)

# run each of crossvalid function as 5 times reapeted 2-fold cross validation and print the results for 2nd dataset
# scores = cross_validation.run_every_crossvalid(datasets[1][0], datasets[1][1], 2, 5, random_state)
# print_helpers.printAlgoResults("2", scores)

# run every crossvalid as 5 times reapeted 2-fold cross validation and print the results for all datasets
cross_validation.run_every_crossvalid_for_every_dataset(datasets, 2, 5, random_state)
