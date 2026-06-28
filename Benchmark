
"""
Benchmark.py

Benchmark six machine learning classifiers
with and without IFOSS preprocessing.

Training Data
-------------
Original dataset:
    X_train
    y_train

IFOSS-resampled dataset:
    X_ifoss
    y_ifoss

Test Data
---------
Independent external test dataset:
    X_test
    y_test

Outputs
-------
Table 1: Baseline Results (No IFOSS)
Table 2: Results with IFOSS
Table 3: Absolute % Point Change
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


SEED = 42


# ============================================================
# INPUT DATA
# ============================================================

"""
Option 1
--------
Run immediately after IFOSS.py
"""

try:

    X_train = Xl
    y_train = yl

    X_ifoss = X_resampled
    y_ifoss = y_resampled

    X_test = Xl_t
    y_test = yl_t

    print("Using datasets from memory.")

except NameError:

    """
    Option 2
    --------
    Load datasets from CSV files.
    """

    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]

    X_ifoss = pd.read_csv("X_IFOSS.csv")
    y_ifoss = pd.read_csv("y_IFOSS.csv").iloc[:, 0]

    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]

    print("Loaded datasets from CSV files.")


# ============================================================
# Detect categorical variables
# ============================================================

cat_cols = [

    c

    for c in X_train.columns

    if X_train[c].dtype == "object"

    or str(X_train[c].dtype) == "category"

]

num_cols = [

    c

    for c in X_train.columns

    if c not in cat_cols

]

cat_idx = [

    X_train.columns.get_loc(c)

    for c in cat_cols

]


# ============================================================
# SVM preprocessing
# ============================================================

svm_preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            StandardScaler(),

            num_cols

        ),

        (

            "cat",

            OrdinalEncoder(

                handle_unknown="use_encoded_value",

                unknown_value=-1

            ),

            cat_cols

        )

    ],

    remainder="drop"

)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(

            solver="liblinear",

            random_state=SEED,

            max_iter=2000

        ),

    "SVM (RBF)":

        Pipeline(

            steps=[

                (

                    "prep",

                    svm_preprocessor

                ),

                (

                    "clf",

                    SVC(

                        kernel="rbf",

                        probability=True,

                        C=1.0,

                        gamma="scale",

                        class_weight="balanced",

                        random_state=SEED

                    )

                )

            ]

        ),

    "Random Forest":

        RandomForestClassifier(

            n_estimators=1000,

            random_state=SEED,

            n_jobs=-1

        ),

    "XGBoost":

        xgb.XGBClassifier(

            n_estimators=1000,

            eval_metric="logloss",

            random_state=SEED,

            n_jobs=-1,

            enable_categorical=True,

            use_label_encoder=False

        ),

    "LightGBM":

        lgb.LGBMClassifier(

            n_estimators=1000,

            random_state=SEED,

            verbose=-1,

            n_jobs=-1

        ),

    "CatBoost":

        cb.CatBoostClassifier(

            iterations=1000,

            loss_function="Logloss",

            random_seed=SEED,

            verbose=0,

            task_type="CPU",

            thread_count=-1,

            early_stopping_rounds=200

        )

}


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def youden_best(y_true, probability):

    fpr, tpr, thr = roc_curve(

        y_true,

        probability

    )

    J = tpr - fpr

    idx = np.argmax(J)

    threshold = thr[idx]

    sensitivity = tpr[idx]

    specificity = 1 - fpr[idx]

    gmean = np.sqrt(

        sensitivity *

        specificity

    )

    return threshold, gmean


def evaluate(y_true, probability):

    auc = roc_auc_score(

        y_true,

        probability

    )

    threshold, gmean = youden_best(

        y_true,

        probability

    )

    prediction = (

        probability >= threshold

    ).astype(int)

    return {

        "AUC":

            auc,

        "Weighted F1":

            f1_score(

                y_true,

                prediction,

                average="weighted",

                zero_division=0

            ),

        "Accuracy":

            accuracy_score(

                y_true,

                prediction

            ),

        "Balanced Accuracy":

            balanced_accuracy_score(

                y_true,

                prediction

            ),

        "G-Mean":

            gmean

    }


# ============================================================
# BENCHMARK
# ============================================================

results_without_ifoss = {}
results_with_ifoss = {}

print("\n" + "=" * 60)
print("Running Benchmark")
print("=" * 60)

for name, model in models.items():

    print(f"\n{name}")

    # ========================================================
    # BASELINE (WITHOUT IFOSS)
    # ========================================================

    if name == "CatBoost":

        if len(cat_idx) > 0:

            model.fit(

                X_train,
                y_train,

                eval_set=(X_test, y_test),

                cat_features=cat_idx,

                use_best_model=True

            )

        else:

            model.fit(

                X_train,
                y_train,

                eval_set=(X_test, y_test),

                use_best_model=True

            )

    else:

        model.fit(

            X_train,
            y_train

        )

    probability = model.predict_proba(

        X_test

    )[:, 1]

    results_without_ifoss[name] = evaluate(

        y_test,

        probability

    )


    # ========================================================
    # WITH IFOSS
    # ========================================================

    if name == "CatBoost":

        if len(cat_idx) > 0:

            model.fit(

                X_ifoss,
                y_ifoss,

                eval_set=(X_test, y_test),

                cat_features=cat_idx,

                use_best_model=True

            )

        else:

            model.fit(

                X_ifoss,
                y_ifoss,

                eval_set=(X_test, y_test),

                use_best_model=True

            )

    else:

        model.fit(

            X_ifoss,
            y_ifoss

        )

    probability = model.predict_proba(

        X_test

    )[:, 1]

    results_with_ifoss[name] = evaluate(

        y_test,

        probability

    )

    print("Done")


# ============================================================
# RESULTS TABLES
# ============================================================

metrics_cols = [

    "AUC",

    "Weighted F1",

    "Accuracy",

    "Balanced Accuracy",

    "G-Mean"

]

df_base = pd.DataFrame(

    results_without_ifoss

).T[metrics_cols]

df_ifoss = pd.DataFrame(

    results_with_ifoss

).T[metrics_cols]

df_improvement = (

    df_ifoss -

    df_base

) * 100


# ============================================================
# PRINT TABLES
# ============================================================

print("\n===== Table 1: Baseline Results (No IFOSS) =====")

print(

    df_base.round(6).to_markdown()

)

print("\n===== Table 2: With IFOSS =====")

print(

    df_ifoss.round(6).to_markdown()

)

print("\n===== Table 3: Absolute % Point Change (With IFOSS − No IFOSS) =====")

print(

    df_improvement.round(2).to_markdown()

)


# ============================================================
# SAVE RESULTS
# ============================================================

with pd.ExcelWriter(

    "Benchmark_Results.xlsx"

) as writer:

    df_base.to_excel(

        writer,

        sheet_name="Without IFOSS"

    )

    df_ifoss.to_excel(

        writer,

        sheet_name="With IFOSS"

    )

    df_improvement.to_excel(

        writer,

        sheet_name="Improvement"

    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("Benchmark Completed")
print("=" * 60)



print("\nResults saved to:")
print("  Benchmark_Results.xlsx")

