"""
IFOSS: Isolation Forest + One-Sided Selection

Standalone implementation of the IFOSS preprocessing framework.

Input:
    X : pandas DataFrame
    y : pandas Series or NumPy array

Output:
    X_resampled
    y_resampled
    best_params
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve

from imblearn import FunctionSampler
from imblearn.under_sampling import OneSidedSelection

from catboost import CatBoostClassifier

# ============================================================
# INPUT DATA
# ============================================================

# Replace with your dataset

X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv").iloc[:,0]
# ============================================================

SEED = 42
N_TRIALS = 200

optuna.logging.set_verbosity(optuna.logging.WARNING)


def to_numpy(y):

    if hasattr(y, "values"):
        y = y.values

    return np.asarray(y).reshape(-1)


y = to_numpy(y)

# ------------------------------------------------------------
# Detect categorical columns automatically
# ------------------------------------------------------------

cat_cols = [

    c for c in X.columns

    if X[c].dtype == "object"

    or str(X[c].dtype) == "category"

]

cat_idx = [

    X.columns.get_loc(c)

    for c in cat_cols

]

# ------------------------------------------------------------
# Train / Validation Split
# ------------------------------------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    random_state=SEED

)


# ------------------------------------------------------------
# CatBoost model
# ------------------------------------------------------------

catboost = CatBoostClassifier(

    iterations=1000,

    loss_function="Logloss",

    random_seed=SEED,

    verbose=0,

    thread_count=-1,

    task_type="CPU",

    early_stopping_rounds=200

)


# ------------------------------------------------------------
# G-Mean
# ------------------------------------------------------------

def gmean_score(y_true, probability):

    fpr, tpr, thr = roc_curve(y_true, probability)

    J = tpr - fpr

    idx = np.argmax(J)

    sensitivity = tpr[idx]

    specificity = 1 - fpr[idx]

    return np.sqrt(sensitivity * specificity)


# ------------------------------------------------------------
# IFOSS
# ------------------------------------------------------------

def apply_ifoss(

        X,

        y,

        max_samples,

        contamination,

        max_features,

        n_neighbors,

        n_seeds_S

):

    def remove_outliers(X_, y_):

        isolation = IsolationForest(

            max_samples=max_samples,

            contamination=contamination,

            max_features=max_features,

            bootstrap=True,

            random_state=SEED

        )

        isolation.fit(X_)

        mask = isolation.predict(X_) == 1

        return X_[mask], y_[mask]

    sampler = FunctionSampler(

        func=remove_outliers

    )

    X_clean, y_clean = sampler.fit_resample(

        X,

        y

    )

    oss = OneSidedSelection(

        sampling_strategy="majority",

        n_neighbors=n_neighbors,

        n_seeds_S=n_seeds_S,

        random_state=SEED,

        n_jobs=-1

    )

    X_resampled, y_resampled = oss.fit_resample(

        X_clean,

        y_clean

    )

    if isinstance(X_resampled, np.ndarray):

        X_resampled = pd.DataFrame(

            X_resampled,

            columns=X.columns

        )

    return X_resampled, y_resampled
# ------------------------------------------------------------
# OPTUNA OBJECTIVE
# ------------------------------------------------------------

def objective(trial):

    params = {

        "max_samples": trial.suggest_int(
            "max_samples",
            1000,
            len(X_train)
        ),

        "contamination": trial.suggest_float(
            "contamination",
            0.01,
            0.50
        ),

        "max_features": trial.suggest_float(
            "max_features",
            0.30,
            1.00
        ),

        "n_neighbors": trial.suggest_int(
            "n_neighbors",
            1,
            10
        ),

        "n_seeds_S": trial.suggest_int(
            "n_seeds_S",
            100,
            1000
        )

    }

    X_res, y_res = apply_ifoss(

        X_train,
        y_train,

        max_samples=params["max_samples"],
        contamination=params["contamination"],
        max_features=params["max_features"],
        n_neighbors=params["n_neighbors"],
        n_seeds_S=params["n_seeds_S"]

    )

if len(cat_idx) > 0:

    catboost.fit(
        X_res,
        y_res,
        eval_set=(X_valid, y_valid),
        cat_features=cat_idx,
        use_best_model=True
    )

else:

    catboost.fit(
        X_res,
        y_res,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    probability = catboost.predict_proba(X_valid)[:, 1]

    return gmean_score(
        y_valid,
        probability
    )


# ------------------------------------------------------------
# RUN OPTUNA
# ------------------------------------------------------------

print("=" * 60)
print("Optimizing IFOSS...")
print("=" * 60)

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)

best_params = study.best_trial.params

print("\nBest IFOSS Parameters\n")
print(best_params)


# ------------------------------------------------------------
# APPLY BEST IFOSS TO FULL DATASET
# ------------------------------------------------------------

X_resampled, y_resampled = apply_ifoss(

    X,
    y,

    max_samples=best_params["max_samples"],
    contamination=best_params["contamination"],
    max_features=best_params["max_features"],
    n_neighbors=best_params["n_neighbors"],
    n_seeds_S=best_params["n_seeds_S"]

)


# ------------------------------------------------------------
# SAVE DATASETS
# ------------------------------------------------------------

X_resampled.to_csv(
    "X_IFOSS.csv",
    index=False
)

pd.Series(y_resampled).to_csv(
    "y_IFOSS.csv",
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\nOptimization Complete")
print("-" * 60)

print(f"Original samples : {len(X)}")
print(f"Resampled samples: {len(X_resampled)}")

print("\nSaved files")
print("  X_IFOSS.csv")
print("  y_IFOSS.csv")
print("\nResampled Feature Matrix (first 5 rows)")
print(X_resampled.head())

print("\nResampled Labels (first 5 rows)")
print(pd.Series(y_resampled).head())
