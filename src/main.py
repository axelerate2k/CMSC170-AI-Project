from data_prep import load_csv, check_data, build_features
from data_prep import stratified_split, stratified_kfold
from data_prep import StandardScaler, FEATURE_NAMES

SEED = 67
DATA_PATH = "data/high_diamond_ranked_10min.csv"

def main():
    #load raw csv
    header, rows = load_csv(DATA_PATH)

    #checks
    info = check_data(header, rows)
    print("=== Data Check ===")
    for key, value in info.items():
        print(f"{key}:<20 {value}")

    #build features
    X, y = build_features(rows)
    n_games, n_features = X.shape
    print()
    print(f"Feature matrix: {n_games} games x {n_features} features")
    print("Features:", ", ".join(FEATURE_NAMES))

    #stratified split
    train_idx, test_idx = stratified_split(y, 0.2, SEED)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    #5-fold stratified cross-validation
    print()
    print("=== 5-Fold Stratified Cross-Validation ===")
    folds = stratified_kfold(y_train, 5, SEED)
    fold_num = 1

    for tr, va in folds:
        val_rate = y_train[va].mean()
        print(f"Fold {fold_num}: Train {len(tr)}, "
              f"validate {len(va)}, win rate {val_rate:.3f}")
        fold_num += 1

    #z-score scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print()
    print("=== Z-Score Check ===")
    train_mean = abs(X_train_scaled.mean(axis=0)).max()
    train_std = X_train_scaled.std(axis=0).mean()
    test_mean = abs(X_test_scaled.mean(axis=0)).max()

    print(f"    Train mean after scaling (should be 0): {train_mean:.6f}")
    print(f"    Train std after scaling (should be 1): {train_std:.6f}")

    print(f"    Test mean after scaling (close to 0): {test_mean:.4f}")

main()