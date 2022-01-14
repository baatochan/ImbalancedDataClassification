import balance_dataset
import load_data
import print_helpers

data_dir = "data/"


# load all the datasets from the data_dir
datasets = load_data.load_data_from_files(data_dir)

# print the content and imbalanceness of loaded datasets
# print_helpers.printAllDatasets(datasets)
print_helpers.printClassImbalanceForAllDatasets(datasets)

# balance the first dataset and print distribution of class to confirm that the set is balanced
balancedDataset = balance_dataset.balance_dataset(datasets[0][0], datasets[0][1])
print_helpers.printClassImbalanceForDataset(balancedDataset)
