import numpy as np
import argparse
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from tools import (
    create_save_paths, calculate_and_save_metrics, calculate_and_save_average_curve_data,
    calculate_glidexp_metrics, get_algorithm_param_grids, get_fixed_params, grid_search_model, clip_metrics
)
from split import load_data, split_data, prepare_features, generate_undersampled_datasets

RANDOM_SEED = 179
N_JOBS = -1
POSITIVE_THRESHOLD = 0.75
np.random.seed(RANDOM_SEED)

def create_pipeline(model, num_cols, cat_cols):
    num_trans = StandardScaler()
    cat_trans = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    pre = ColumnTransformer([
        ('num', num_trans, num_cols),
        ('cat', cat_trans, cat_cols)
    ])
    return Pipeline([('preprocessor', pre), ('classifier', clone(model))])

def evaluate(under_data, models, num_cols, cat_cols, Xv, yv, Xt, yt, mpath, cpath):
    cv_res = {n:[] for n,_ in models}
    val_p = {n:[] for n,_ in models}
    tst_p = {n:[] for n,_ in models}

    if os.path.exists(mpath):
        os.remove(mpath)
    if os.path.exists(cpath):
        os.remove(cpath)

    for i, (Xu, yu) in enumerate(under_data):
        for name, base in models:
            print(f"[{i+1}] {name}")
            pipe = create_pipeline(base, num_cols, cat_cols)
            pipe.fit(Xu, yu)

            pv = pipe.predict_proba(Xv)[:,1]
            predv = (pv > POSITIVE_THRESHOLD).astype(int)
            val_p[name].append(pv)

            pt = pipe.predict_proba(Xt)[:,1]
            predt = (pt > POSITIVE_THRESHOLD).astype(int)
            tst_p[name].append(pt)

            calculate_and_save_metrics(f"{name}_Val_{i+1}", yv, predv, pv, mpath, append=True)
            calculate_and_save_metrics(f"{name}_Test_{i+1}", yt, predt, pt, mpath, append=True)

    for name in val_p:
        avg_vp = np.mean(val_p[name], axis=0)
        avg_tp = np.mean(tst_p[name], axis=0)
        calculate_and_save_average_curve_data(name, 'val', yv, val_p[name], cpath)
        calculate_and_save_average_curve_data(name, 'test', yt, tst_p[name], cpath)

    return val_p, tst_p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['SGD','KNN','Random Forest','AdaBoost','XGBoost'])
    parser.add_argument('--mode', type=str, required=True,
                        choices=['grid_search','fixed_params','default'])
    args = parser.parse_args()

    save_paths = create_save_paths()
    data = load_data(r"..\data.csv")
    train, val, test = split_data(data)
    X_train, y_train, X_val, y_val, X_test, y_test, num_cols, cat_cols = prepare_features(train, val, test)
    undersampled = generate_undersampled_datasets(X_train, y_train, n_datasets=100)

    param_grids = get_algorithm_param_grids()
    fixed_params = get_fixed_params()

    best_model = None
    if args.mode == 'grid_search':
        if args.model == 'Random Forest':
            print("Random Forest uses DEFAULT params (grid search skipped)")
            base = RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_SEED)
        else:
            base = {
                'SGD': SGDClassifier(random_state=RANDOM_SEED),
                'KNN': KNeighborsClassifier(n_jobs=N_JOBS),
                'AdaBoost': AdaBoostClassifier(random_state=RANDOM_SEED),
                'XGBoost': xgb.XGBClassifier(n_jobs=N_JOBS, random_state=RANDOM_SEED)
            }[args.model]
            pipe = create_pipeline(base, num_cols, cat_cols)
            best_model, _ = grid_search_model(pipe, param_grids[args.model], X_train, y_train, cv=5, n_jobs=N_JOBS)

    elif args.mode == 'fixed_params':
        if args.model == 'Random Forest':
            print("Random Forest uses DEFAULT params (fixed params skipped)")
            best_model = RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_SEED)
        else:
            base = {
                'SGD': SGDClassifier(**fixed_params['SGD'], random_state=RANDOM_SEED),
                'KNN': KNeighborsClassifier(**fixed_params['KNN'], n_jobs=N_JOBS),
                'AdaBoost': AdaBoostClassifier(**fixed_params['AdaBoost'], random_state=RANDOM_SEED),
                'XGBoost': xgb.XGBClassifier(**fixed_params['XGBoost'], n_jobs=N_JOBS, random_state=RANDOM_SEED)
            }[args.model]
            best_model = base

    else:
        best_model = {
            'SGD': SGDClassifier(loss='log_loss', random_state=RANDOM_SEED),
            'KNN': KNeighborsClassifier(n_jobs=N_JOBS),
            'Random Forest': RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_SEED),
            'AdaBoost': AdaBoostClassifier(random_state=RANDOM_SEED),
            'XGBoost': xgb.XGBClassifier(n_jobs=N_JOBS, random_state=RANDOM_SEED)
        }[args.model]

    models = [
        (args.model, best_model)
    ]

    mpath = save_paths['metrics'] / 'all_metrics.csv'
    cpath = save_paths['metrics'] / 'curve_data.csv'
    val_probs, test_probs = evaluate(undersampled, models, num_cols, cat_cols, X_val, y_val, X_test, y_test, mpath, cpath)

    glide = calculate_glidexp_metrics(data, "FullDataset", mpath, append=True)
    print("\nAll tasks completed!")