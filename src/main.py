from data_prep import load_csv, check_data, build_features
from data_prep import stratified_split, stratified_kfold
from data_prep import StandardScaler, FEATURE_NAMES
import numpy as np
from knn import KNN

SEED = 67
DATA_PATH = "../data/high_diamond_ranked_10min.csv"

def accuracy(y_true, y_pred):
    """fraction of correct predictions (temp until metrics.py is made)"""
    return np.mean(y_true == y_pred)


def cv_accuracy(make_model, X, y):
    """
    average accuracy over 5 stratified folds
    refit scaler every fold so we don't leak into validation.
    """
    scores = []
    for tr, va in stratified_kfold(y, 5, SEED):
        scaler = StandardScaler()
        scaler.fit(X[tr])
        X_tr = scaler.transform(X[tr])
        X_va = scaler.transform(X[va])

        model = make_model()
        model.fit(X_tr, y[tr])
        pred = model.predict(X_va)
        scores.append(accuracy(y[va], pred))

    return np.mean(scores), np.std(scores)

def main():
    #load raw csv
    header, rows = load_csv(DATA_PATH)

    #checks
    info = check_data(header, rows)
    print("=== Data Check ===")
    for key, value in info.items():
        print(f"{key:<20} {value}")

    #build features
    X, y = build_features(rows)

    # train/test split (temp 80/20 split, stratified)
    train_idx, test_idx = stratified_split(y, 0.2, SEED)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    print()
    print(f"Train: {len(y_train)} games | Test: {len(y_test)} games")

    # find the best k using 5-fold cv on train data
    print()
    print("=== Tuning KNN (5-fold CV) ===")
    best_k = None
    best_k_acc = 0
    for k in [1, 5, 11, 21, 51, 101, 151, 201, 251, 301]:
        mean, std = cv_accuracy(lambda: KNN(k), X_train, y_train)
        print(f"  k = {k:<4} accuracy {mean:.4f} +/- {std:.4f}")
        if mean > best_k_acc:
            best_k = k
            best_k_acc = mean

    print()
    print(f"Chosen: k = {best_k}")

    # train on the full training set and test once
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNN(best_k)
    knn.fit(X_train_s, y_train)
    knn_acc = accuracy(y_test, knn.predict(X_test_s))

    print()
    print("=== Test set accuracy ===")
    print(f"  KNN (k = {best_k}): {knn_acc:.4f}")

main()