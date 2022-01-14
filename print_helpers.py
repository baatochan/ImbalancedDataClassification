# Function that takes a list of loaded datasets (in the form of List(DataFrame, DataFrame)) and prints all dfs.
# Params: List(DataFrame, DataFrame)
# Returns: None
from statistics import mean


def printAllDatasets(datasets):
    i = 1
    for dataset in datasets:
        print("#### DATASET " + str(i) + " ####")
        print(dataset[0])
        print()
        print(dataset[1])
        print()
        i += 1


# Function for printing the imbalance of y_class for every dataset.
# Params: List(DataFrame, DataFrame)
# Returns: None
def printClassImbalanceForAllDatasets(datasets):
    for dataset in datasets:
        printClassImbalanceForDataset(dataset)


# Function for printing the imbalance of y_class.
# Params: (DataFrame, DataFrame)
# Returns: None
def printClassImbalanceForDataset(dataset):
    print(dataset[1].value_counts())
    print()


# Function for printing the crossvalid score for the model. Var model_name is the name of the model, Var scores is a
# dictionary with results for each algo
# Params: String/Int, Dictionary
# Returns: None
def printAlgoResults(model_name, scores):
    print("#### Model " + str(model_name) + " ####")
    for algo, score in scores.items():
        print(algo)
        print(score)
        print(mean(score))
        print()
