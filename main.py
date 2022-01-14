import load_data
import print_helpers

data_dir = "data/"


# load all the datasets from the data_dir
datasets = load_data.load_data_from_files(data_dir)

# print the content and imbalanceness of loaded datasets
# print_helpers.printAllDatasets(datasets)
print_helpers.printClassImbalanceForAllDatasets(datasets)
