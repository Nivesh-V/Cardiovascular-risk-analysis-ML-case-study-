"""Shared preprocessing + inference helpers (same steps as the three notebooks)."""
import os, json, datetime as dt
import numpy as np, pandas as pd, joblib

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "HeartFailureDataset.csv")
MODEL_DIR = os.path.join(BASE, "models")
LOG_PATH = os.path.join(BASE, "prediction_log.csv")
CAT_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

TASKS = {
    "Classification": {"target": "cardio", "file": "classification.joblib"},
    "Regression": {"target": "ap_hi", "file": "regression.joblib"},
    "Clustering": {"target": None, "file": "clustering.joblib"},
}


def load_raw():
    return pd.read_csv(DATA_PATH)


def clean_supervised(df):
    """Steps shared by the classification and regression notebooks."""
    df = df.drop_duplicates().drop(columns=["id"]).copy()
    for c in df.columns:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    df["age_years"] = (df["age"] / 365.25).round(1)
    df = df.drop(columns=["age"])
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    return df


def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def load_model(task):
    return joblib.load(os.path.join(MODEL_DIR, TASKS[task]["file"]))


def build_row(art, inp):
    """Raw user input dict -> model-ready 1-row DataFrame (same cleaning as training)."""
    r = dict(inp)
    r["gender"] = {1: 0, 2: 1}.get(int(r["gender"]), r["gender"]) if art["task"] != "Clustering" else int(r["gender"])
    r["age_years"] = float(r["age_years"]) if art["task"] == "Clustering" else round(float(r["age_years"]), 1)
    r["bmi"] = float(r["weight"]) / (float(r["height"]) / 100) ** 2
    row = pd.DataFrame([r])
    for c, (lo, hi) in art["clip_bounds"].items():
        if c in row:
            row[c] = row[c].clip(lo, hi)
    return row[art["features"]]


def predict(task, inp):
    """Returns dict with prediction + details using the best model of that task."""
    art = load_model(task)
    X = build_row(art, inp)
    out = {"task": task, "target": art["target"], "algorithm": art["best_name"],
           "algorithm_type": art["algorithm_type"]}
    if task == "Classification":
        Z = art["pca"].transform(art["scaler"].transform(X))
        m = art["model"]
        pred = int(m.predict(Z)[0])
        out["result"] = "Cardiovascular disease likely" if pred else "No cardiovascular disease"
        out["result_code"] = pred
        if hasattr(m, "predict_proba"):
            out["confidence"] = float(m.predict_proba(Z)[0, 1])
    elif task == "Regression":
        Z = art["pca"].transform(art["scaler"].transform(X))
        v = float(art["model"].predict(Z)[0])
        out["result"] = f"{v:.1f} mmHg"
        out["result_code"] = round(v, 2)
        out["rmse"] = art["best_metrics"]["RMSE"]
    else:
        Z = art["scaler"].transform(X)
        c = int(np.argmin(((art["centroids"] - Z[0]) ** 2).sum(axis=1)))
        out["result"] = f"Cluster {c}"
        out["result_code"] = c
        out["point_2d"] = art["pca"].transform(pd.DataFrame(Z, columns=art["features"]))[0].tolist()
    return out


def log_prediction(inp, res):
    row = {"timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "task_type": res["task"], "target_variable": res["target"] or "(none - unsupervised)",
           "algorithm_type": res["algorithm_type"], "algorithm_used": res["algorithm"],
           **{f"in_{k}": v for k, v in inp.items()},
           "result": res["result"]}
    if "confidence" in res:
        row["probability_cvd"] = round(res["confidence"], 4)
    df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


def read_log():
    return pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
