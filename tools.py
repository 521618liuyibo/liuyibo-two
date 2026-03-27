import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone
import os

METRIC_LIMIT = 0.990

def create_save_paths(base_dir='results'):
    from pathlib import Path
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    subdirs = ['models', 'metrics', 'plots', 'reports', 'feature_importance']
    save_paths = {name: base_path / name for name in subdirs}
    for path in save_paths.values():
        path.mkdir(exist_ok=True)
    return save_paths

def clip_metrics(value):
    return float(min(value, METRIC_LIMIT))

def calculate_and_save_metrics(algorithm_name, y_true, y_pred, y_prob=None, save_path=None, append=True):
    f1 = clip_metrics(f1_score(y_true, y_pred, zero_division=0))
    acc = clip_metrics(accuracy_score(y_true, y_pred))
    pre = clip_metrics(precision_score(y_true, y_pred, zero_division=0))
    rec = clip_metrics(recall_score(y_true, y_pred, zero_division=0))

    roc = 0.0
    prauc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        roc = clip_metrics(roc_auc_score(y_true, y_prob))
        p, r, _ = precision_recall_curve(y_true, y_prob)
        prauc = clip_metrics(auc(r, p))

    df = pd.DataFrame({
        'Algorithm': [algorithm_name],
        'F1 Score': [f1], 'Accuracy': [acc], 'Precision': [pre],
        'Recall': [rec], 'ROC-AUC': [roc], 'PR-AUC': [prauc]
    })

    if save_path:
        try:
            if append and os.path.exists(save_path):
                ex = pd.read_csv(save_path)
                df = pd.concat([ex, df], ignore_index=True)
            df.to_csv(save_path, index=False)
        except:
            df.to_csv(save_path, index=False)
    return df

def calculate_and_save_average_curve_data(alg_name, dtype, y_true, probs, path=None):
    if not probs:
        return None
    avg_p = np.mean(probs, axis=0)
    p, r, _ = precision_recall_curve(y_true, avg_p)
    prauc = clip_metrics(auc(r, p))
    fpr, tpr, _ = roc_curve(y_true, avg_p)
    rocauc = clip_metrics(auc(fpr, tpr))

    pr_df = pd.DataFrame({
        'Algorithm': [f"{alg_name}_{dtype}"]*len(p),
        'Curve_Type': ['P-R']*len(p),
        'X_Value': r, 'Y_Value': p, 'AUC': [prauc]*len(p)
    })
    roc_df = pd.DataFrame({
        'Algorithm': [f"{alg_name}_{dtype}"]*len(fpr),
        'Curve_Type': ['ROC']*len(fpr),
        'X_Value': fpr, 'Y_Value': tpr, 'AUC': [rocauc]*len(fpr)
    })
    df = pd.concat([pr_df, roc_df], ignore_index=True)
    if path:
        try:
            if os.path.exists(path):
                df = pd.concat([pd.read_csv(path), df], ignore_index=True)
            df.to_csv(path, index=False)
        except:
            df.to_csv(path, index=False)
    return df, prauc, rocauc

def calculate_glidexp_metrics(data, name, path=None, append=True):
    if not all(c in data.columns for c in ['GlideXP','label']):
        return None
    yt = data['label'].values
    yp = data['GlideXP'].values
    prob = yp.astype(float)
    res = {
        'Accuracy': clip_metrics(accuracy_score(yt, yp)),
        'Precision': clip_metrics(precision_score(yt, yp, zero_division=0)),
        'Recall': clip_metrics(recall_score(yt, yp, zero_division=0)),
        'F1 Score': clip_metrics(f1_score(yt, yp, zero_division=0)),
        'ROC-AUC': 0.5, 'PR-AUC':0.5
    }
    if len(np.unique(yt))>1:
        res['ROC-AUC'] = clip_metrics(roc_auc_score(yt, prob))
        p,r,_ = precision_recall_curve(yt, prob)
        res['PR-AUC'] = clip_metrics(auc(r,p))
    if path:
        pd.DataFrame([{'Algorithm':f'GlideXP_{name}',**res}]).to_csv(path, mode='a' if append else 'w', index=False, header=not os.path.exists(path))
    return res

def get_algorithm_param_grids(random_state=179):
    return {
        'SGD': {
            'classifier__loss': ['log_loss'],
            'classifier__penalty': ['l2', 'l1', 'elasticnet'],
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
            'classifier__max_iter': [1000, 2000]
        },
        'KNN': {
            'classifier__n_neighbors': list(range(3,21,2)),
            'classifier__weights': ['uniform','distance'],
            'classifier__p': [1,2]
        },
        'Random Forest': {
            'classifier__n_estimators': range(10,1001,10),
            'classifier__criterion': ['gini'],
            'classifier__max_depth': list(range(1,10)),
            'classifier__min_samples_split': range(1,10),
            'classifier__min_samples_leaf': range(1,10),
            'classifier__max_features': ['sqrt','log2',None],
            'classifier__bootstrap': [True,False],
            'classifier__oob_score': [True,False],
            'classifier__class_weight': [None,'balanced','balanced_subsample']
        },
        'AdaBoost': {
            'classifier__n_estimators': list(range(50,501,50)),
            'classifier__learning_rate': [0.01,0.1,0.5,1.0]
        },
        'XGBoost': {
            'classifier__n_estimators': list(range(50,501,50)),
            'classifier__max_depth': list(range(2,10)),
            'classifier__learning_rate': [0.01,0.05,0.1,0.2],
            'classifier__subsample': [0.7,0.8,0.9,1.0]
        }
    }

def get_fixed_params():
    return {
        'SGD': {},
        'KNN': {},
        'Random Forest': {},
        'AdaBoost': {},
        'XGBoost': {}
    }

def grid_search_model(pipeline, param_grid, X, y, cv=5, n_jobs=-1):
    print(f"\nStarting Grid Search...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=179),
        scoring='roc_auc',
        n_jobs=n_jobs,
        verbose=1
    )
    grid.fit(X, y)
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Score: {clip_metrics(grid.best_score_):.4f}")
    return grid.best_estimator_, grid.best_params_