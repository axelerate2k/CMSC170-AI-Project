"""
main.py  (Logistic Regression only)
Loads and preprocesses the data, tunes Logistic
Regression with 5-fold CV, then tests it.

Run from the repository root:  python src/main.py
"""
import numpy as np

from data_prep import load_csv, check_data, build_features
from data_prep import stratified_split, stratified_kfold
from data_prep import StandardScaler, FEATURE_NAMES
from logistic_regression import LogisticRegression

SEED = 42
DATA_PATH = "../data/high_diamond_ranked_10min.csv"


def accuracy(y_true, y_pred):
    """Fraction of correct predictions.
    (Temporary: full metrics come in metrics.py.)"""
    return np.mean(y_true == y_pred)


def cv_accuracy(make_model, X, y):
    """
    Average accuracy over 5 stratified folds.
    The scaler is re-fit in every fold, so the
    validation fold is never seen during training.
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
    # 1. Load and check the data.
    header, rows = load_csv(DATA_PATH)
    info = check_data(header, rows)
    print("=== Data check ===")
    for key, value in info.items():
        print(f"  {key:<20}{value}")

    # 2. Build the 13 Blue - Red features.
    X, y = build_features(rows)

    # 3. Stratified 80/20 train/test split.
    train_idx, test_idx = stratified_split(y, 0.2, SEED)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    print()
    print(f"Train: {len(y_train)} games | Test: {len(y_test)} games")

    # 4. Tune the learning rate with 5-fold CV.
    print()
    print("=== Tuning Logistic Regression (5-fold CV) ===")
    best_lr = None
    best_lr_acc = 0
    for lr in [0.001, 0.01, 0.1, 0.5]:
        def make_logreg():
            return LogisticRegression(lr, 1000)
        mean, std = cv_accuracy(make_logreg, X_train, y_train)
        print(f"  lr = {lr:<6} accuracy {mean:.4f} +/- {std:.4f}")
        if mean > best_lr_acc:
            best_lr = lr
            best_lr_acc = mean

    print()
    print(f"Chosen: learning rate = {best_lr}")

    # 5. Train on the full training set, test once.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(best_lr, 1000)
    logreg.fit(X_train_s, y_train)
    logreg_acc = accuracy(y_test, logreg.predict(X_test_s))

    print()
    print("=== Test set accuracy ===")
    print(f"  Logistic Regression: {logreg_acc:.4f}")

    # 6. Which features matter most?
    # Features are z-scored, so weights are comparable.
    print()
    print("=== Logistic Regression weights ===")
    order = np.argsort(-np.abs(logreg.w))
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<20}{logreg.w[i]:+.4f}")


if __name__ == "__main__":
    main()
