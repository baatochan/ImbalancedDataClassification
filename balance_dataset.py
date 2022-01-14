from imblearn.over_sampling import SMOTE


def balance_dataset(X_features, Y_class, random_state):
    sm = SMOTE(sampling_strategy='minority', random_state=random_state, n_jobs=-1)
    oversampled_X, oversampled_Y = sm.fit_resample(X_features, Y_class)

    return oversampled_X, oversampled_Y
