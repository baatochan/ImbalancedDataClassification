from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import balance_dataset
import print_helpers


# Function that trains selected classifier (setting the selected base_estimator) with the train data and validates it
# with the test data. Var classifier should be one of ['adaboost', 'bagging', 'subspace'] value, base_est one of
# ['decisionTree', 'logisticRegression', 'gaussianNB'] values.
# Params: string, string, DataFrame, DataFrame, DataFrame, DataFrame, int/None
# Returns: float
def train_model(classifier, base_est, X_train, Y_train, X_test, Y_test, random_state):
    match base_est:
        case 'decisionTree':
            est = DecisionTreeClassifier(criterion="entropy")
        case 'logisticRegression':
            est = LogisticRegression(max_iter=3500)
        case 'gaussianNB':
            est = GaussianNB()

    match classifier:
        case 'adaboost':
            clf = AdaBoostClassifier(base_estimator=est, n_estimators=50, learning_rate=1)
        case 'bagging':
            clf = BaggingClassifier(base_estimator=est, n_estimators=50, random_state=random_state)
        case 'subspace':
            # clf = RandomSubspaceClassifier()
            halfOfFeatures = int(len(X_train.columns)/2)  # int() to round it down
            clf = BaggingClassifier(base_estimator=est, n_estimators=50, random_state=random_state, bootstrap=False, max_features=halfOfFeatures)

    model = clf.fit(X_train, Y_train.values.ravel())
    Y_pred = model.predict(X_test)

    return metrics.f1_score(Y_test, Y_pred, pos_label='positive')


# Function that does n time repeated k-fold cross validation of selected classifier with selected base_estimator. Var
# classifier should be one of ['adaboost', 'bagging', 'subspace'] value, base_est one of ['decisionTree',
# 'logisticRegression', 'gaussianNB'] values, k is represented by n_splits, n is represented by n_repeats,
# random_state is used to ensure that every run data is split in the same way each run the random_state is the same.
# Random_state can be set to None for random run. Function returns a list of F1 scores.
# Params: string, string, DataFrame, DataFrame, int, int, int/None
# Returns: List
def run_crossvalid(classifier, base_est, X_features, Y_class, n_splits, n_repeats, random_state):
    scores = []

    split_algorithm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_class):
        X_train = X_features.iloc[train_samples_indexes]
        X_test = X_features.iloc[test_samples_indexes]
        Y_train = Y_class.iloc[train_samples_indexes]
        Y_test = Y_class.iloc[test_samples_indexes]

        oversampled_X, oversampled_Y = balance_dataset.balance_dataset(X_train, Y_train, random_state)

        f1Score = train_model(classifier, base_est, oversampled_X, oversampled_Y, X_test, Y_test, random_state)

        scores.append(f1Score)

    return scores


# Function that runs n time repeated k-fold cross validation for every classifier. K is represented by n_splits,
# n is represented by n_repeats, random_state is used to ensure that every run data is split in the same way each run
# the random_state is the same. Random_state can be set to None for random run. Function returns a list of F1 scores.
# Params: DataFrame, DataFrame, int, int, int/None
# Returns: Dictionary
def run_every_crossvalid(X_features, Y_class, n_splits, n_repeats, random_state):
    scores = {}
    for classifier in ['adaboost', 'bagging', 'subspace']:
        for base_est in ['decisionTree', 'logisticRegression', 'gaussianNB']:
            scoreArray = run_crossvalid(classifier, base_est, X_features, Y_class, n_splits, n_repeats, random_state)
            scores[classifier + ' ' + base_est] = scoreArray

    return scores


# Function that runs n time repeated k-fold cross validation for every classifier for all provided datasets. K is
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
