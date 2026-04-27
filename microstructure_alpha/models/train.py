import copy

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def run_model_parallel(
    X,
    y=None,
    model=None,
    tscv=None,
    n_jobs=4,
    scale=False,
    calibrate=False,
    calib_method="isotonic",
    cv=3,
    clip_quantiles=(0.01, 0.99),
    mask=None,
    hmm=False,
):

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(
            train_idx,
            test_idx,
            X,
            y,
            model,
            scale,
            calibrate,
            calib_method,
            cv,
            clip_quantiles,
            mask,
            hmm,
        )
        for train_idx, test_idx in tscv.split(X)
    )
    if mask is not None:
        oof_full, oof_masked, fold_aucs, fold_coefs = zip(*results)

        return {
            "fold_oof_full": list(oof_full),
            "fold_oof": list(oof_masked),
            "fold_aucs": list(fold_aucs),
            "coefs": list(fold_coefs),
            "mean_auc": np.nanmean(fold_aucs),
        }
    elif hmm:
        regime_oof, regime_prob_oof, test_idx = zip(*results)

        return {
            "fold_regime_oof": list(regime_oof),
            "fold_regime_prob_oof": list(regime_prob_oof),
            "test_idx": list(test_idx),
        }

    else:
        oof_preds, fold_aucs, fold_coefs = zip(*results)

        return {
            "fold_oof": list(oof_preds),
            "fold_aucs": list(fold_aucs),
            "coefs": list(fold_coefs),
            "mean_auc": np.mean(fold_aucs),
        }


def run_fold(
    train_idx,
    test_idx,
    X,
    y=None,
    model=None,
    scale=True,
    calibrate=False,
    calib_method="isotonic",
    cv=3,
    clip_quantiles=(0.01, 0.99),
    mask=None,
    hmm=False,
):
    if not hmm:
        oof = np.full(len(X), np.nan)

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if mask is not None:
            train_mask = mask.iloc[train_idx]
            test_mask = mask.iloc[test_idx]

            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
        lower = X_train.quantile(clip_quantiles[0])
        upper = X_train.quantile(clip_quantiles[1])

        X_train = X_train.clip(lower, upper, axis=1)
        X_test = X_test.clip(lower, upper, axis=1)

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        base_model = copy.deepcopy(model)
        base_model.fit(X_train, y_train)

        if calibrate:
            model_fold = CalibratedClassifierCV(
                copy.deepcopy(model),
                method=calib_method,
                cv=3,
            )
            model_fold.fit(X_train, y_train)
            p = model_fold.predict_proba(X_test)[:, 1]
        else:
            model_fold = base_model
            p = model_fold.predict_proba(X_test)[:, 1]

        oof[test_idx] = p

        if mask is not None:
            auc = roc_auc_score(y_test[test_mask], p[test_mask])
        else:
            auc = roc_auc_score(y_test, p)

        if hasattr(base_model, "coef_"):
            coef = pd.Series(base_model.coef_[0], index=X.columns)

        elif hasattr(base_model, "feature_importances_"):
            coef = pd.Series(base_model.feature_importances_, index=X.columns)

        elif hasattr(base_model, "get_booster"):
            score = base_model.get_booster().get_score(importance_type="gain")
            coef = pd.Series(score).reindex(X.columns, fill_value=0.0)

        else:
            coef = pd.Series(index=X.columns, data=np.nan)

        if mask is not None:
            oof_masked = np.full(len(X), np.nan)

            test_mask = mask.iloc[test_idx].values
            test_index = np.array(test_idx)

            X_test_masked = X_test[test_mask]
            y_test_masked = y_test[test_mask]

            p = model_fold.predict_proba(X_test_masked)[:, 1]

            oof_masked[test_index[test_mask]] = p

            auc = roc_auc_score(y_test_masked, p)
            return oof, oof_masked, auc, coef

        return oof, auc, coef

    elif hmm:
        n = len(X)
        regime_oof = np.full(n, np.nan)
        regime_prob_oof = np.full((n, model.n_components), np.nan)
        oof = np.full(len(X), np.nan)

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)

        lower = X_train.quantile(clip_quantiles[0])
        upper = X_train.quantile(clip_quantiles[1])

        X_train = X_train.clip(lower, upper, axis=1)
        X_test = X_test.clip(lower, upper, axis=1)

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        base_model = copy.deepcopy(model)
        base_model.fit(X_train)

        probs = base_model.predict_proba(X_test)
        regimes = base_model.predict(X_test)
        # =========================
        # STORE OOF
        # =========================
        regime_oof[test_idx] = regimes
        regime_prob_oof[test_idx] = probs
        print(len(X_test), probs.shape)
        return regime_oof, regime_prob_oof, test_idx
