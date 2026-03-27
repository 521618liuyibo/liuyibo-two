import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, test_size=0.1, val_size=0.1, random_state=179):
    train_val, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data["label"]
    )
    val_rel = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=random_state, stratify=train_val["label"]
    )
    return train, val, test

def prepare_features(train, val, test):
    X_train = train.loc[:, "docking_score":]
    y_train = train["label"].values
    X_val = val.loc[:, "docking_score":]
    y_val = val["label"].values
    X_test = test.loc[:, "docking_score":]
    y_test = test["label"].values

    num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()

    for df in [X_val, X_test]:
        missing = set(X_train.columns) - set(df.columns)
        for c in missing:
            df[c] = 0
    return X_train, y_train, X_val, y_val, X_test, y_test, num_cols, cat_cols

def generate_undersampled_datasets(X, y, n=100, random_state=179):
    X_min = X[y==1]
    X_maj = X[y==0]
    n_min = len(X_min)
    datasets = []
    for i in range(n):
        np.random.seed(random_state + i)
        idx = np.random.choice(len(X_maj), n_min, replace=False)
        Xm = X_maj.iloc[idx]
        ym = np.zeros(len(idx))
        Xu = pd.concat([X_min, Xm])
        yu = np.concatenate([np.ones(n_min), ym])
        datasets.append((Xu, yu))
    return datasets