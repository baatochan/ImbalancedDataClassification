# Function that takes a list of loaded datasets (in the form of List(DataFrame, DataFrame)) and prints all dfs.
# Params: List(DataFrame, DataFrame)
# Returns: None
def printAllDatasets(datasets):
    i = 1
    for dataset in datasets:
        print("#### DATASET " + str(i) + " ####")
        print(dataset[0])
        print()
        print(dataset[1])
        print()
        i += 1

def printClassImbalanceForAllDatasets(datasets):
    for dataset in datasets:
        print(dataset[1].value_counts())
        print()
