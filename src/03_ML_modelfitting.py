# -*- coding: utf-8 -*-
"""
Script: 04_ML_Modeling_with_TEST_and_CHANCE.py
Project: Motivational Salience Index (MSI)
Author: Marie Pittet, adapted by Malika Tapparel, re-adapted by Marie
Description:
- Benchmarks ElasticNet, SVR, Ridge, and HistGB on item-level behavioral data.
- Evaluates generalization on held-out data
- Adds a chance benchmark by permuting predictions within each person.
"""

# %%
# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# ------------------------------------------------------------
# 1) CONFIG & TRANSFORMERS
# ------------------------------------------------------------
TRAIN_PATH = "../data/preprocessed/by_item/training.csv"
TEST_PATH  = "../data/preprocessed/by_item/test.csv"

TARGET, PERSON_ID, ITEM_ID = "vas_score", "sbj", "item_id"
ID_COLS = [PERSON_ID, ITEM_ID]
DROP_SUBSTRINGS = ["n_trials"]

N_SPLITS, RANDOM_STATE = 5, 42

# Human-Readable Mapping
RENAME_DICT = {
    'n_correct': 'Correct Responses (Total)',
    'n_miss': 'Missed Responses (Total)',
    'mean_rt_towards': 'Mean Correct Towards RT',
    'median_rt_towards': 'Median Correct Towards RT',
    'mean_rt_away': 'Mean Correct Away RT',
    'median_rt_away': 'Median Correct Away RT',
    'acc_all': 'Overall Accuracy (%)',
    'acc_towards': 'Towards Accuracy (%)',
    'acc_away': 'Away Accuracy (%)',
    'iw_mean': 'Mean Implicit Wanting',
    'iw_median': 'Median Implicit Wanting',
    'mean_rt_towards_relative': 'Rel. Mean Correct Towards RT',
    'median_rt_towards_relative': 'Rel. Median Correct Towards RT',
    'mean_rt_away_relative': 'Rel. Mean Correct Away RT',
    'median_rt_away_relative': 'Rel. Median Correct Away RT',
    'iw_mean_relative': 'Rel. Mean Implicit Wanting',
    'iw_median_relative': 'Rel. Median Implicit Wanting',
    'missing_indicator': 'Any Data Imputation Flag'
}

class BehavioralCleaner(BaseEstimator, TransformerMixin):
    """Hard-removes all non-behavioral IDs and substrings to prevent leakage."""
    def __init__(self, substrings, ids_to_kill):
        self.substrings = substrings
        self.ids_to_kill = ids_to_kill

    def fit(self, X, y=None):
        cols = list(X.columns)
        self.cols_to_drop_ = [
            c for c in cols
            if any(s in c for s in self.substrings)
            or any(i in c for i in self.ids_to_kill)
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors="ignore")

# ------------------------------------------------------------
# 2) METRICS
# ------------------------------------------------------------
def get_spearman(df, target_col, pred_col):
    """Within-person Spearman averaged across persons."""
    rhos = []
    for _, g in df.groupby(PERSON_ID):
        if len(g) < 3:
            continue
        yt, yp = g[target_col].values, g[pred_col].values
        if np.nanstd(yt) < 1e-12 or np.nanstd(yp) < 1e-12:
            continue
        rho, _ = spearmanr(yt, yp)
        rhos.append(rho)
    return np.mean(rhos) if rhos else 0.0

def add_within_person_chance(df, pred_col="y_pred", out_col="y_rand", seed=RANDOM_STATE):
    """
    Chance benchmark: permute predictions within each person.
    Preserves each person's prediction distribution but destroys item mapping.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    def _permute(arr):
        arr = np.asarray(arr)
        return rng.permutation(arr)

    df[out_col] = df.groupby(PERSON_ID)[pred_col].transform(_permute)
    return df

# ------------------------------------------------------------
# 3) MODELING
# ------------------------------------------------------------
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

y_train = df_train[TARGET].astype(float)
X_train = df_train.drop(columns=[TARGET]).copy()

y_test = df_test[TARGET].astype(float)
X_test = df_test.drop(columns=[TARGET]).copy()

mlflow.set_experiment("food_liking_no_leakage_with_test")

pipeline = Pipeline([
    ("cleaner", BehavioralCleaner(substrings=DROP_SUBSTRINGS, ids_to_kill=ID_COLS)),
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

models = {
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE),
    "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=200)
}

cv = GroupKFold(n_splits=N_SPLITS)

for model_name, reg in models.items():
    with mlflow.start_run(run_name=f"{model_name}_CV_and_TEST"):

        # ------------------------------------------------------------
        # A) Cross-validation (held-out items)
        # ------------------------------------------------------------
        all_cv_res = []

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups=X_train[ITEM_ID]), 1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            # Center target within-person (train means applied to validation persons)
            mu = y_tr.groupby(X_tr[PERSON_ID]).mean()
            y_tr_c = (y_tr - X_tr[PERSON_ID].map(mu)).fillna(0)
            y_va_c = (y_va - X_va[PERSON_ID].map(mu)).fillna(0)

            Xtr_p = pipeline.fit_transform(X_tr)
            Xva_p = pipeline.transform(X_va)

            reg.fit(Xtr_p, y_tr_c)

            fold_df = pd.DataFrame({
                PERSON_ID: X_va[PERSON_ID],
                "y_true": y_va_c,
                "y_pred": reg.predict(Xva_p)
            }).dropna()

            all_cv_res.append(fold_df)

        cv_results = pd.concat(all_cv_res, ignore_index=True)

        rho_cv = get_spearman(cv_results, "y_true", "y_pred")

        cv_results = add_within_person_chance(cv_results, pred_col="y_pred", out_col="y_rand", seed=RANDOM_STATE)
        rho_cv_chance = get_spearman(cv_results, "y_true", "y_rand")

        # ------------------------------------------------------------
        # B) Final train-on-all + test evaluation
        # ------------------------------------------------------------
        mu_full = y_train.groupby(X_train[PERSON_ID]).mean()
        y_train_c_full = (y_train - X_train[PERSON_ID].map(mu_full)).fillna(0)

        X_train_full_p = pipeline.fit_transform(X_train)
        reg.fit(X_train_full_p, y_train_c_full)

        # IMPORTANT: test target centering uses test's own within-person mean
        mu_test = y_test.groupby(X_test[PERSON_ID]).mean()
        y_test_c = (y_test - X_test[PERSON_ID].map(mu_test)).fillna(0)

        X_test_p = pipeline.transform(X_test)
        y_test_pred = reg.predict(X_test_p)

        test_df = pd.DataFrame({
            PERSON_ID: X_test[PERSON_ID],
            "y_true": y_test_c,
            "y_pred": y_test_pred
        }).dropna()

        rho_test = get_spearman(test_df, "y_true", "y_pred")

        test_df = add_within_person_chance(test_df, pred_col="y_pred", out_col="y_rand", seed=RANDOM_STATE)
        rho_test_chance = get_spearman(test_df, "y_true", "y_rand")

        print(
            f"\n[{model_name}] "
            f"CV Spearman: {rho_cv:.4f} (Chance: {rho_cv_chance:.4f}) | "
            f"TEST Spearman: {rho_test:.4f} (Chance: {rho_test_chance:.4f})"
        )

        # ------------------------------------------------------------
        # C) Visualization
        # ------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        sns.barplot(
            x=["CV Chance", "CV Model", "TEST Chance", "TEST Model"],
            y=[rho_cv_chance, rho_cv, rho_test_chance, rho_test],
            ax=axes[0],
            palette="coolwarm"
        )
        axes[0].set_title(f"{model_name} Performance (Within-Person Spearman ρ)")
        axes[0].set_ylabel("Correlation Coefficient")
        axes[0].set_xlabel("")

        # Feature Importance (computed on TEST set inputs, for the final trained model)
        feat_names = pipeline.named_steps["imputer"].get_feature_names_out()

        if model_name in ["Ridge", "ElasticNet"]:
            imps = np.abs(reg.coef_)
        else:
            r = permutation_importance(
                reg, X_test_p, y_test_c, n_repeats=5, random_state=RANDOM_STATE
            )
            imps = r.importances_mean

        fi = (
            pd.DataFrame({"Feature": feat_names, "Imp": imps})
            .sort_values("Imp", ascending=False)
            .head(10)
        )
        fi["Feature"] = fi["Feature"].map(lambda x: RENAME_DICT.get(x, x))

        sns.barplot(x="Imp", y="Feature", data=fi, ax=axes[1], palette="viridis")
        axes[1].set_title("Top Predictors (Final Model)")
        axes[1].set_xlabel("Impact on Prediction Score")
        axes[1].set_ylabel("Behavioral Metric")

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------
        # D) MLflow logging
        # ------------------------------------------------------------
        mlflow.log_metric("rho_cv", rho_cv)
        mlflow.log_metric("rho_cv_chance", rho_cv_chance)
        mlflow.log_metric("rho_test", rho_test)
        mlflow.log_metric("rho_test_chance", rho_test_chance)
        mlflow.sklearn.log_model(reg, f"model_{model_name}")