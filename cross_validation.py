from imblearn.ensemble import EasyEnsembleClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

import balance_dataset


# Function that does n time repeated k-fold cross validation of AdaBoostClassifier. K is represented by n_splits,
# n is represented by n_repeats, random_state is used to ensure that every run data is split in the same way each run
# the random_state is the same. Random_state can be set to None for random run. Function returns a list of F1 scores.
# Params: DataFrame, DataFrame, int, int, int/None
# Returns: List
def run_adaboost_crossvalid(X_features, Y_class, n_splits, n_repeats, random_state):
    scores = []

    split_algorithm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_class):
        X_train = X_features.iloc[train_samples_indexes]
        X_test = X_features.iloc[test_samples_indexes]
        Y_train = Y_class.iloc[train_samples_indexes]
        Y_test = Y_class.iloc[test_samples_indexes]

        oversampled_X, oversampled_Y = balance_dataset.balance_dataset(X_train, Y_train, random_state)

        abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        model = abc.fit(oversampled_X, oversampled_Y.values.ravel())
        Y_pred = model.predict(X_test)

        scores.append(metrics.f1_score(Y_test, Y_pred, pos_label='positive'))

    return scores


# Function that does n time repeated k-fold cross validation of BaggingClassifier. K is represented by n_splits,
# n is represented by n_repeats, random_state is used to ensure that every run data is split in the same way each run
# the random_state is the same. Random_state can be set to None for random run. Function returns a list of F1 scores.
# Params: DataFrame, DataFrame, int, int, int/None
# Returns: List
def run_bagging_crossvalid(X_features, Y_class, n_splits, n_repeats, random_state):
    scores = []

    split_algorithm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_class):
        X_train = X_features.iloc[train_samples_indexes]
        X_test = X_features.iloc[test_samples_indexes]
        Y_train = Y_class.iloc[train_samples_indexes]
        Y_test = Y_class.iloc[test_samples_indexes]

        oversampled_X, oversampled_Y = balance_dataset.balance_dataset(X_train, Y_train, random_state)

        bc = BaggingClassifier(n_estimators=50, random_state=random_state)
        model = bc.fit(oversampled_X, oversampled_Y.values.ravel())
        Y_pred = model.predict(X_test)

        scores.append(metrics.f1_score(Y_test, Y_pred, pos_label='positive'))

    return scores


# Function that does n time repeated k-fold cross validation of RandomForestClassifier. K is represented by n_splits,
# n is represented by n_repeats, random_state is used to ensure that every run data is split in the same way each run
# the random_state is the same. Random_state can be set to None for random run. Function returns a list of F1 scores.
# Params: DataFrame, DataFrame, int, int, int/None
# Returns: List
def run_forest_crossvalid(X_features, Y_class, n_splits, n_repeats, random_state):
    scores = []

    split_algorithm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_class):
        X_train = X_features.iloc[train_samples_indexes]
        X_test = X_features.iloc[test_samples_indexes]
        Y_train = Y_class.iloc[train_samples_indexes]
        Y_test = Y_class.iloc[test_samples_indexes]

        oversampled_X, oversampled_Y = balance_dataset.balance_dataset(X_train, Y_train, random_state)

        rfc = RandomForestClassifier(n_estimators=50, random_state=random_state)
        model = rfc.fit(oversampled_X, oversampled_Y.values.ravel())
        Y_pred = model.predict(X_test)

        scores.append(metrics.f1_score(Y_test, Y_pred, pos_label='positive'))

    return scores
