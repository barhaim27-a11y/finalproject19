import os, json, warnings
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, auc
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Optional/soft deps
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from tensorflow import keras
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def _ensure_dirs():
    Path("assets").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    """Load dataset; if missing, synthesize a small demo dataset so app boots."""
    _ensure_dirs()
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    # synthesize tiny dataset (not for grading; only to allow boot)
    rng = np.random.default_rng(42)
    n = 120
    cols = config.FEATURES + [config.TARGET]
    X = rng.normal(0, 1, size=(n, len(config.FEATURES)))
    y = (rng.random(n) > 0.55).astype(int)
    df = pd.DataFrame(X, columns=config.FEATURES); df[config.TARGET] = y
    df.insert(0, config.NAME_COL, [f"s{i:03d}" for i in range(n)])
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df

def validate_training_data(df: pd.DataFrame) -> Tuple[bool, list]:
    errs = []
    for col in config.FEATURES + [config.TARGET]:
        if col not in df.columns:
            errs.append(f"Missing column: {col}")
    if len(df) < 40:
        errs.append("Dataset too small (<40 rows).")
    return (len(errs) == 0, errs)

def _get_model_by_name(name: str, params: dict):
    name = name or config.DEFAULT_MODEL
    if name == "LogisticRegression":
        clf = LogisticRegression(**{k:v for k,v in params.items() if k in ["C","max_iter","penalty"]}, random_state=config.RANDOM_STATE)
    elif name == "RandomForest":
        clf = RandomForestClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","max_depth","min_samples_split"]}, random_state=config.RANDOM_STATE, n_jobs=-1)
    elif name == "SVC":
        clf = SVC(**{k:v for k,v in params.items() if k in ["C","kernel","probability"]}, random_state=config.RANDOM_STATE)
    elif name == "GradientBoosting":
        clf = GradientBoostingClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","learning_rate","max_depth"]}, random_state=config.RANDOM_STATE)
    elif name == "ExtraTrees":
        clf = ExtraTreesClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","max_depth","min_samples_split"]}, random_state=config.RANDOM_STATE, n_jobs=-1)
    elif name == "XGBoost" and HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate",0.05)),
            max_depth=int(params.get("max_depth",3)),
            subsample=float(params.get("subsample",0.9)),
            colsample_bytree=float(params.get("colsample_bytree",0.9)),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            tree_method="hist",
        )
    elif name == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64,32)),
            alpha=float(params.get("alpha", 0.0005)),
            max_iter=int(params.get("max_iter", 400)),
            random_state=config.RANDOM_STATE,
        )
    elif name == "KerasNN" and HAS_KERAS:
        # simple Keras model wrapped in sklearn-like API
        hidden = params.get("hidden", (64,32))
        dropout = float(params.get("dropout", 0.2))
        epochs = int(params.get("epochs", 30))
        batch_size = int(params.get("batch_size", 16))

        def build_fn(input_dim):
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(input_dim,)))
            for h in hidden:
                model.add(keras.layers.Dense(int(h), activation="relu"))
                if dropout>0: model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
            return model

        class KerasWrapper:
            def __init__(self): self.model=None
            def fit(self, X, y):
                self.model = build_fn(X.shape[1])
                self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
                return self
            def predict_proba(self, X):
                proba = self.model.predict(X, verbose=0).reshape(-1,1)
                return np.hstack([1-proba, proba])
        clf = KerasWrapper()
    else:
        # fall back to logistic
        clf = LogisticRegression(max_iter=200, C=1.0, penalty="l2", random_state=config.RANDOM_STATE)

    # preprocessing
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])
    return pipe

def _get_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if proba.ndim == 1:  # rare case
            return proba
        return proba[:,1]
    if hasattr(pipe, "decision_function"):
        z = pipe.decision_function(X)
        # map to [0,1] with a sigmoid-ish transformation
        return 1/(1+np.exp(-z))
    # worst-case: predict labels then cast
    return pipe.predict(X).astype(float)

def _opt_threshold(y_true, y_scores, mode: str = "youden"):
    """Return optimal threshold and (optionally) extra metric for that threshold."""
    if mode == "f1":
        prec, rec, thr = precision_recall_curve(y_true, y_scores)
        f1s = (2*prec*rec/(prec+rec+1e-9))
        i = int(np.nanargmax(f1s[:-1])) if len(f1s)>1 else 0
        t = thr[i] if len(thr)>0 else 0.5
        return float(t), {"f1_opt": float(np.nanmax(f1s))}
    # default Youden
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    j = tpr - fpr
    i = int(np.nanargmax(j)) if len(j)>0 else 0
    t = thr[i] if len(thr)>0 else 0.5
    return float(t), {}

def _compute_metrics(y_true, y_scores, y_pred, model_name: str, thr_mode: str="youden"):
    m = {}
    m["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["f1"] = float(f1_score(y_true, y_pred))
    m["precision"] = float(precision_score(y_true, y_pred))
    m["recall"] = float(recall_score(y_true, y_pred))
    opt_thr, extra = _opt_threshold(y_true, y_scores, mode=thr_mode)
    m["opt_thr"] = float(opt_thr)
    m.update(extra)
    m["n_samples"] = int(len(y_true))
    return m

def _save_plots(y_true, y_scores, model_name: str, tag: str = "run"):
    assets = Path("assets"); assets.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {model_name}"); plt.legend()
    roc_path = assets / f"roc_{tag}.png"; plt.savefig(roc_path, dpi=150, bbox_inches="tight"); plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_scores); ap = average_precision_score(y_true, y_scores)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – {model_name}"); plt.legend()
    pr_path = assets / f"pr_{tag}.png"; plt.savefig(pr_path, dpi=150, bbox_inches="tight"); plt.close()
    # CM (0.5 threshold for visualization)
    cm = confusion_matrix(y_true, (y_scores>=0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm); disp.plot(values_format="d")
    plt.title("Confusion Matrix"); cm_path = assets / f"cm_{tag}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()
    return {
        "roc": {"fpr": list(map(float,fpr)), "tpr": list(map(float,tpr)), "auc": float(roc_auc), "path": str(roc_path)},
        "pr": {"prec": list(map(float,prec)), "rec": list(map(float,rec)), "ap": float(ap), "path": str(pr_path)},
        "cm": {"matrix": cm.tolist(), "path": str(cm_path)},
    }

def create_pipeline(model_name: str, model_params: dict):
    return _get_model_by_name(model_name, model_params or {})

def train_model(data_path: str, model_name: str, model_params: dict,
                test_size: float=0.2, do_cv: bool=True, do_tune: bool=True,
                artifact_tag: str = "run", thr_mode: str = "youden"):
    _ensure_dirs()
    df = load_data(data_path)
    ok, errs = validate_training_data(df)
    if not ok:
        return {"ok": False, "errors": errs}

    X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE
    )

    pipe = create_pipeline(model_name, model_params)

    cv_means = None
    if do_cv:
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        scoring = ["roc_auc","accuracy","f1","precision","recall"]
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        cv_means = {m: float(np.mean(scores[f"test_{m}"])) for m in ["roc_auc","accuracy","f1","precision","recall"]}

    if do_tune:
        grid = config.PARAM_GRIDS.get(model_name, None)
        if grid:
            gs = GridSearchCV(pipe, grid, scoring=config.SCORING, cv=3, n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            pipe = gs.best_estimator_

    pipe.fit(X_tr, y_tr)
    y_scores = _get_proba(pipe, X_val)
    thr = 0.5
    if isinstance(y_scores, np.ndarray):
        thr, extra = _opt_threshold(y_val, y_scores, mode=thr_mode)
    y_pred = (y_scores>=thr).astype(int)
    metrics = _compute_metrics(y_val, y_scores, y_pred, model_name, thr_mode=thr_mode)
    curves = _save_plots(y_val, y_scores, model_name, tag=artifact_tag)

    cand_path = f"models/candidate_{artifact_tag}.joblib"
    joblib.dump(pipe, cand_path)

    # log run
    try:
        import datetime as _dt
        runs = Path(config.RUNS_CSV)
        runs.parent.mkdir(parents=True, exist_ok=True)
        row = {"tag":artifact_tag,"model":model_name,"roc_auc":metrics["roc_auc"],"f1":metrics["f1"],"accuracy":metrics["accuracy"],
               "precision":metrics["precision"],"recall":metrics["recall"],"opt_thr":metrics.get("opt_thr",0.5),
               "time":_dt.datetime.utcnow().isoformat()}
        if runs.exists():
            df_runs = pd.read_csv(runs); df_runs = pd.concat([df_runs, pd.DataFrame([row])], ignore_index=True)
        else:
            df_runs = pd.DataFrame([row])
        df_runs.to_csv(runs, index=False)
    except Exception:
        pass

    return {
        "ok": True,
        "candidate_path": cand_path,
        "val_metrics": metrics,
        "cv_means": cv_means,
        "curves": curves,
        "params_used": model_params
    }

def has_production() -> bool:
    return Path(config.MODEL_PATH).exists()

def read_best_meta() -> Dict[str, Any]:
    p = Path("assets/best_model.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def promote_model_to_production(candidate_path: str, metadata: Dict[str, Any] = None) -> str:
    _ensure_dirs()
    dst = Path(config.MODEL_PATH); dst.parent.mkdir(parents=True, exist_ok=True)
    if not Path(candidate_path).exists():
        raise FileNotFoundError(f"Candidate not found: {candidate_path}")
    # copy model
    import shutil; shutil.copyfile(candidate_path, dst)
    # write meta
    meta = read_best_meta()
    meta.update(metadata or {})
    if "opt_thr" not in meta:
        # default to 0.5 if not provided
        meta["opt_thr"] = 0.5
    Path("assets/best_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Promoted {candidate_path} → {dst}"

def _predict_core(model_path: str, X: pd.DataFrame, threshold: float=None) -> pd.DataFrame:
    pipe = joblib.load(model_path)
    scores = _get_proba(pipe, X)
    thr = 0.5 if threshold is None else float(threshold)
    pred = (scores>=thr).astype(int)
    return pd.DataFrame({"proba_PD": scores, "pred": pred})

def predict_with_production(X: pd.DataFrame, threshold: float=None) -> pd.DataFrame:
    if not has_production(): raise FileNotFoundError("No production model found.")
    meta = read_best_meta()
    thr = threshold if threshold is not None else float(meta.get("opt_thr", 0.5))
    return _predict_core(config.MODEL_PATH, X, threshold=thr)

def run_prediction(row_df: pd.DataFrame):
    out = predict_with_production(row_df)
    return int(out.iloc[0]["pred"]), float(out.iloc[0]["proba_PD"])

def batch_predict(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    feats = config.FEATURES
    X = df[feats] if all(f in df.columns for f in feats) else df
    preds = predict_with_production(X)
    return pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

def evaluate_model(model_path: str, data_path: str=None, artifact_tag: str="best_eval"):
    df = load_data(data_path or config.DATA_PATH)
    X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    pipe = joblib.load(model_path)
    scores = _get_proba(pipe, X_te)
    meta = read_best_meta()
    thr = float(meta.get("opt_thr", 0.5))
    y_pred = (scores>=thr).astype(int)
    metrics = _compute_metrics(y_te, scores, y_pred, "production", thr_mode="youden")
    curves = _save_plots(y_te, scores, "production", tag=artifact_tag)
    # also persist static names for Best tab if tag matches
    if artifact_tag.startswith("best_eval"):
        import shutil
        for src, dst in [(curves["roc"]["path"], "assets/roc_best_eval.png"),
                         (curves["pr"]["path"], "assets/pr_best_eval.png"),
                         (curves["cm"]["path"], "assets/cm_best_eval.png")]:
            if Path(src).exists():
                shutil.copyfile(src, dst)
    return {"metrics": metrics, "curves": curves}
