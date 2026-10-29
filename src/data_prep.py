import csv
import random
import numpy as np

FEATURE_NAMES = [
    "firstBlood",
    "killsDiff",
    "assistsDiff",
    "wardsPlacedDiff",
    "wardsDestroyedDiff",
    "dragonsDiff",
    "heraldsDiff",
    "towersDiff",
    "goldDiff",
    "expDiff",
    "avgLevelDiff",
    "csDiff",
    "jungleCsDiff",
]

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            rows.append(row)
    return header, rows

def check_data(header, rows):
    #count empty cells
    missing = 0
    for row in rows:
        for col in header:
            value = row[col]
            if value is None or value.strip() == "":
                missing += 1

    #count repeated gameIds
    game_ids = []
    for row in rows:
        game_ids.append(row["gameId"])
    duplicates = len(game_ids) - len(set(game_ids))

    #count blue wins
    wins = 0
    for row in rows:
        wins += int(row["blueWins"])

    return {
        "n_rows": len(rows),
        "n_cols": len(header),
        "missing_values": missing,
        "duplicate_gameIds": duplicates,
        "blue_wins": wins,
        "blue_losses": len(rows) - wins,
    }

def diff(row, stat):
    blue = float(row["blue" + stat])
    red = float(row["red" + stat])
    return blue - red

def build_features(rows):
    X = []
    y = []

    for row in rows:
        features = [
            float(row["blueFirstBlood"]),
            diff(row, "Kills"),
            diff(row, "Assists"),
            diff(row, "WardsPlaced"),
            diff(row, "WardsDestroyed"),
            diff(row, "Dragons"),
            diff(row, "Heralds"),
            diff(row, "TowersDestroyed"),
            float(row["blueGoldDiff"]),
            float(row["blueExperienceDiff"]),
            diff(row, "AvgLevel"),
            diff(row, "TotalMinionsKilled"),
            diff(row, "TotalJungleMinionsKilled")
        ]
        X.append(features)
        y.append(int(row["blueWins"]))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    return X, y

def indices_of_class(y, label):
    result = []
    for i in range(len(y)):
        if y[i] == label:
            result.append(i)
    return result

def stratified_split(y, test_ratio=0.2, seed=67):
    rng = random.Random(seed)
    train_idx = []
    test_idx = []

    for label in (0, 1):
        idx = indices_of_class(y, label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)

def stratified_kfold(y, k=5, seed=67):
    rng = random.Random(seed)
    folds = []
    for i in range(k):
        folds.append([])

    for label in (0, 1):
        idx = indices_of_class(y, label)
        rng.shuffle(idx)
        for pos in range(len(idx)):
            fold = pos % k
            folds[fold].append(idx[pos])

    pairs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = []
        for m in range(k):
            if m != i:
                train_idx.extend(folds[m])
        pairs.append((np.array(train_idx), np.array(val_idx)))

    return pairs

class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0

        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)