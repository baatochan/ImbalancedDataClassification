from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

import balance_dataset
import print_helpers


# Function that trains selected classifier for train data and validates it with test data. Var classifier should be one
# of ['adaboost', 'bagging', 'forest'] value.
# Params: string, DataFrame, DataFrame, DataFrame, DataFrame, int/None
# Returns: float
def train_model(classifier, X_train, Y_train, X_test, Y_test, random_state):
    match classifier:
        case 'adaboost':
            clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        case 'bagging':
            clf = BaggingClassifier(n_estimators=50, random_state=random_state)
        case 'forest':
            clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model = clf.fit(X_train, Y_train.values.ravel())
    Y_pred = model.predict(X_test)

    return metrics.f1_score(Y_test, Y_pred, pos_label='positive')


# Function that does n time repeated k-fold cross validation of selected Classifier. Var classifier should be one of
# ['adaboost', 'bagging', 'forest'] value, k is represented by n_splits, n is represented by n_repeats, random_state
# is used to ensure that every run data is split in the same way each run the random_state is the same. Random_state
# can be set to None for random run. Function returns a list of F1 scores.
# Params: string, DataFrame, DataFrame, int, int, int/None
# Returns: List
def run_crossvalid(classifier, X_features, Y_class, n_splits, n_repeats, random_state):
    scores = []

    split_algorithm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_class):
        X_train = X_features.iloc[train_samples_indexes]
        X_test = X_features.iloc[test_samples_indexes]
        Y_train = Y_class.iloc[train_samples_indexes]
        Y_test = Y_class.iloc[test_samples_indexes]

        oversampled_X, oversampled_Y = balance_dataset.balance_dataset(X_train, Y_train, random_state)

        f1Score = train_model(classifier, oversampled_X, oversampled_Y, X_test, Y_test, random_state)

        scores.append(f1Score)

    return scores


# Function that runs n time repeated k-fold cross validation for every Classifier. K is represented by n_splits,
# n is represented by n_repeats, random_state is used to ensure that every run data is split in the same way each run
# the random_state is the same. Random_state can be set to None for random run. Function returns a list of F1 scores.
# Params: DataFrame, DataFrame, int, int, int/None
# Returns: Dictionary
def run_every_crossvalid(X_features, Y_class, n_splits, n_repeats, random_state):
    scores = {}
    for classifier in ['adaboost', 'bagging', 'forest']:
        scoreArray = run_crossvalid(classifier, X_features, Y_class, n_splits, n_repeats, random_state)
        scores[classifier] = scoreArray

    return scores


# Function that runs n time repeated k-fold cross validation for every Classifier for all provided datasets. K is
# represented by n_splits, n is represented by n_repeats, random_state is used to ensure that every run data is split
# in the same way each run the random_state is the same. Random_state can be set to None for random run. Function
# prints the results for every trained model.
# Params: List(DataFrame, DataFrame), int, int, int/None
# Returns: None
def run_every_crossvalid_for_every_dataset(datasets, n_splits, n_repeats, random_state):
    i = 1
    for dataset in datasets:
        score = run_every_crossvalid(dataset[0], dataset[1], n_splits, n_repeats, random_state)
        print_helpers.printAlgoResults(i, score)
        i += 1
