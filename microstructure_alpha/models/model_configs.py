import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def calibrate_model(model, method="isotonic", cv=3):
    return CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
    )


def build_model_summary(all_results):

    df = pd.DataFrame(
        [
            {
                "model": name,
                "mean_auc": np.mean(res["fold_aucs"]),
                "std_auc": np.std(res["fold_aucs"]),
                "min_auc": np.min(res["fold_aucs"]),
            }
            for name, res in all_results.items()
        ]
    )

    return df.set_index("model")


LINEAR_MODELS = {
    "logreg_l2": LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="saga",
        max_iter=5000,
    ),
    "logreg_l2_strong": LogisticRegression(
        C=0.1,
        l1_ratio=0.0,
        solver="saga",
        max_iter=5000,
    ),
    "logreg_l2_very_strong": LogisticRegression(
        C=0.01,
        l1_ratio=0.0,
        solver="saga",
        max_iter=5000,
    ),
    "logreg_l1": LogisticRegression(
        C=0.1,
        l1_ratio=1.0,
        solver="saga",
        max_iter=5000,
    ),
    "logreg_en": LogisticRegression(
        C=0.1,
        l1_ratio=0.5,
        solver="saga",
        max_iter=5000,
    ),
}


TREE_MODELS = {
    "rf": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=50,
        max_features=0.2,
        n_jobs=1,
        random_state=42,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=50,
        max_features=0.2,
        n_jobs=1,
        random_state=42,
    ),
    "xgb": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=42,
        eval_metric="logloss",
    ),
    "xgb_deep": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=42,
        eval_metric="logloss",
    ),
    "xgb_stable": XGBClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        reg_alpha=1.0,
        min_child_weight=40,
        gamma=1.0,
        n_jobs=1,
        random_state=42,
        eval_metric="logloss",
    ),
}


OTHER_MODELS = {
    "gaussian_nb": GaussianNB(),
}
