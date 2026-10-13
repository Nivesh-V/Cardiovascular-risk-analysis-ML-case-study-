"""Trains every model from the three notebooks and saves the best per task.
Usage: python train.py classification|regression|clustering|all"""
import sys, time, warnings, json
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                              BaggingClassifier, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
import core
warnings.filterwarnings("ignore")
RS = 42
CONT = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi"]


def prep(target):
    df = core.clean_supervised(core.load_raw())
    bounds = {c: core.iqr_bounds(df[c]) for c in CONT}
    dc = df.copy()
    for c in CONT:
        dc[c] = dc[c].clip(*bounds[c])
    dc["gender"] = dc["gender"].map({1: 0, 2: 1})
    if target == "cardio":
        X, y = dc.drop(columns="cardio"), dc["cardio"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=RS)
    else:
        y = dc["ap_hi"]; X = dc.drop(columns=["ap_hi", "cardio"])
        bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=bins, random_state=RS)
    sc = StandardScaler().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)
    full = PCA(random_state=RS).fit(Str)
    n = int(np.argmax(np.cumsum(full.explained_variance_ratio_) >= .95) + 1)
    pca = PCA(n_components=n, random_state=RS).fit(Str)
    return dict(bounds=bounds, X=X, ytr=ytr, yte=yte, sc=sc, pca=pca,
                Ptr=pca.transform(Str), Pte=pca.transform(Ste))


def run_classification():
    d = prep("cardio"); res = {}
    def fit(name, est, grid, search=15000):
        t = time.time()
        Xs, _, ys, _ = train_test_split(d["Ptr"], d["ytr"], train_size=search, stratify=d["ytr"], random_state=RS)
        gs = GridSearchCV(est, grid, cv=3, scoring="roc_auc", n_jobs=-1).fit(Xs, ys)
        best = clone(est).set_params(**gs.best_params_).fit(d["Ptr"], d["ytr"])
        score = best.predict_proba(d["Pte"])[:, 1] if hasattr(best, "predict_proba") else best.decision_function(d["Pte"])
        res[name] = dict(model=best, params=gs.best_params_, cv=gs.best_score_, pred=best.predict(d["Pte"]), score=score,
                         train_acc=accuracy_score(d["ytr"], best.predict(d["Ptr"])), t=time.time() - t)
        print(name, gs.best_params_, round(gs.best_score_, 4), f"{time.time()-t:.0f}s", flush=True)
    fit("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RS), {"C": [.01, .1, 1, 10], "solver": ["liblinear"]})
    fit("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 10, 15, 20, 30], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]})
    fit("Gaussian Naive Bayes", GaussianNB(), {"var_smoothing": np.logspace(-11, -7, 5)})
    fit("Decision Tree", DecisionTreeClassifier(random_state=RS), {"max_depth": [2, 3, 5, 8, 10, 15, 20, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]})
    fit("SVC", SVC(random_state=RS, cache_size=1000), [{"C": [.1, 1, 10], "kernel": ["linear"]}, {"C": [.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]}], search=6000)
    fit("Random Forest", RandomForestClassifier(random_state=RS, n_jobs=-1), {"n_estimators": [100, 200, 300], "max_depth": [8, 12, 16]})
    fit("AdaBoost", AdaBoostClassifier(random_state=RS), {"n_estimators": [50, 100, 200], "learning_rate": [.1, .5, 1.0]})
    fit("Gradient Boosting", GradientBoostingClassifier(random_state=RS), {"n_estimators": [100, 200], "learning_rate": [.05, .1], "max_depth": [3, 4]})
    fit("Bagging", BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS), random_state=RS, n_jobs=-1), {"n_estimators": [50, 100], "estimator__max_depth": [6, 8, 12]})
    fit("MLP Neural Network", MLPClassifier(max_iter=300, early_stopping=True, random_state=RS), {"hidden_layer_sizes": [(50,), (100,), (64, 32)], "activation": ["relu", "tanh"]})
    yte = d["yte"]; rows = []
    for n, r in res.items():
        rows.append({"Model": n, "Accuracy": accuracy_score(yte, r["pred"]), "Precision": precision_score(yte, r["pred"]),
                     "Recall": recall_score(yte, r["pred"]), "F1-score": f1_score(yte, r["pred"]),
                     "ROC-AUC": roc_auc_score(yte, r["score"]), "CV ROC-AUC": r["cv"], "Train Accuracy": r["train_acc"],
                     "Best params": str(r["params"])})
    tab = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best = tab.loc[0, "Model"]; print(tab.drop(columns="Best params").round(4).to_string())
    roc = {}
    for n, r in res.items():
        f, t_, _ = roc_curve(yte, r["score"]); k = np.linspace(0, len(f) - 1, 150).astype(int)
        roc[n] = (f[k].tolist(), t_[k].tolist())
    save("classification", dict(task="Classification", target="cardio", algorithm_type="Supervised - Classification",
         best_name=best, model=res[best]["model"], scaler=d["sc"], pca=d["pca"], features=list(d["X"].columns),
         clip_bounds=d["bounds"], results=tab, best_metrics=tab.iloc[0].to_dict(), roc=roc,
         confusion=confusion_matrix(yte, res[best]["pred"]).tolist(), n_train=len(d["ytr"]), n_test=len(yte)))


def run_regression():
    d = prep("ap_hi"); res = {}
    def fit(name, est, grid, search=15000, sample_refit=False):
        t = time.time()
        Xs, _, ys, _ = train_test_split(d["Ptr"], d["ytr"], train_size=search, random_state=RS)
        gs = GridSearchCV(est, grid, cv=3, scoring="r2", n_jobs=-1).fit(Xs, ys)
        best = clone(est).set_params(**gs.best_params_)
        fx, fy = (Xs, ys) if sample_refit else (d["Ptr"], d["ytr"])
        best.fit(fx, fy)
        res[name] = dict(model=best, params=gs.best_params_, cv=gs.best_score_, pred=best.predict(d["Pte"]),
                         train_r2=r2_score(fy, best.predict(fx)))
        print(name, gs.best_params_, round(gs.best_score_, 4), f"{time.time()-t:.0f}s", flush=True)
    fit("Linear Regression", LinearRegression(), {"fit_intercept": [True, False]})
    fit("Ridge", Ridge(random_state=RS), {"alpha": [.01, .1, 1, 10, 100]})
    fit("Lasso", Lasso(max_iter=20000, random_state=RS), {"alpha": [.001, .01, .1, 1, 10]})
    fit("ElasticNet", ElasticNet(max_iter=20000, random_state=RS), {"alpha": [.001, .01, .1, 1], "l1_ratio": [.2, .5, .8]})
    fit("Polynomial Regression", Pipeline([("poly", PolynomialFeatures(include_bias=False)), ("scale_poly", StandardScaler()), ("regressor", LinearRegression())]), {"poly__degree": [1, 2, 3]})
    fit("Decision Tree", DecisionTreeRegressor(random_state=RS), {"max_depth": [3, 5, 8, 10, 15, 20, None], "min_samples_split": [2, 5, 10]})
    fit("Random Forest", RandomForestRegressor(random_state=RS, n_jobs=-1), {"n_estimators": [100, 200], "max_depth": [10, 20], "min_samples_split": [2, 5]})
    fit("Gradient Boosting", GradientBoostingRegressor(random_state=RS), {"learning_rate": [.03, .05, .1], "n_estimators": [100, 200], "max_depth": [2, 3]})
    fit("SVR", SVR(cache_size=1000), {"C": [1, 10, 50], "epsilon": [.05, .1, .2], "kernel": ["rbf", "linear"]}, search=5000, sample_refit=True)
    fit("KNN", KNeighborsRegressor(), {"n_neighbors": [3, 5, 10, 15, 20, 30, 50], "weights": ["uniform", "distance"]})
    yte = d["yte"]; rows = []
    for n, r in res.items():
        rows.append({"Model": n, "R2": r2_score(yte, r["pred"]), "RMSE": float(np.sqrt(mean_squared_error(yte, r["pred"]))),
                     "MAE": mean_absolute_error(yte, r["pred"]), "CV R2": r["cv"], "Train R2": r["train_r2"], "Best params": str(r["params"])})
    tab = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    best = tab.loc[0, "Model"]; print(tab.drop(columns="Best params").round(4).to_string())
    idx = np.random.RandomState(RS).choice(len(yte), 1500, replace=False)
    save("regression", dict(task="Regression", target="ap_hi", algorithm_type="Supervised - Regression",
         best_name=best, model=res[best]["model"], scaler=d["sc"], pca=d["pca"], features=list(d["X"].columns),
         clip_bounds=d["bounds"], results=tab, best_metrics=tab.iloc[0].to_dict(),
         actual_vs_pred=(yte.values[idx].tolist(), res[best]["pred"][idx].tolist()), n_train=len(d["ytr"]), n_test=len(yte)))


def run_clustering():
    df = core.load_raw().drop_duplicates().reset_index(drop=True)
    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    X = df.drop(columns=["id", "cardio", "age", "height", "weight"])
    cont = [c for c in X.columns if c not in core.CAT_COLS]
    bounds = {c: core.iqr_bounds(X[c]) for c in cont}
    Xc = X.copy()
    for c in cont:
        Xc[c] = Xc[c].clip(*bounds[c])
    sc = StandardScaler().fit(Xc)
    Xs = pd.DataFrame(sc.transform(Xc), columns=X.columns)
    idx = Xs.sample(n=5000, random_state=RS).index
    Xcl = Xs.loc[idx].reset_index(drop=True)
    cardio = df.loc[idx, "cardio"].reset_index(drop=True)
    K = 3
    km = KMeans(n_clusters=K, random_state=RS, n_init=10).fit(Xcl); kl = km.labels_
    links = []
    for l in ["ward", "complete", "average"]:
        lab = AgglomerativeClustering(n_clusters=K, linkage=l).fit_predict(Xcl)
        links.append((l, silhouette_score(Xcl, lab)))
    bl = max(links, key=lambda t: t[1])[0]
    al = AgglomerativeClustering(n_clusters=K, linkage=bl).fit_predict(Xcl)
    def m(lab): return dict(Silhouette=silhouette_score(Xcl, lab), Davies_Bouldin=davies_bouldin_score(Xcl, lab), Calinski_Harabasz=calinski_harabasz_score(Xcl, lab))
    tab = pd.DataFrame([{"Model": "K-Means", "Parameters": f"k={K}", **m(kl)},
                        {"Model": "Agglomerative Hierarchical", "Parameters": f"k={K}, linkage={bl}", **m(al)}])
    tab = tab.sort_values("Silhouette", ascending=False).reset_index(drop=True)
    best = tab.loc[0, "Model"]; lab = kl if best == "K-Means" else al
    print(tab.round(4).to_string())
    cent = np.vstack([Xcl[lab == c].mean(axis=0) for c in range(K)])   # nearest-centroid rule for new points
    pca = PCA(n_components=2, random_state=RS).fit(Xcl); P = pca.transform(Xcl)
    raw = Xc.loc[idx].reset_index(drop=True).assign(cluster=lab, cardio=cardio.values)
    prof = raw.groupby("cluster").mean().round(2); prof["size"] = raw.groupby("cluster").size()
    save("clustering", dict(task="Clustering", target=None, algorithm_type="Unsupervised - Clustering", best_name=best,
         scaler=sc, pca=pca, features=list(X.columns), clip_bounds=bounds, centroids=cent, results=tab,
         best_metrics=tab.iloc[0].to_dict(), profile=prof, pca_points=P[:2500].tolist(), pca_labels=lab[:2500].tolist(),
         k_curve=None, linkage_scores=links, n_train=len(Xcl)))


def save(name, art):
    joblib.dump(art, f"{core.MODEL_DIR}/{name}.joblib", compress=3)
    tab = art["results"].copy(); tab.to_csv(f"{core.MODEL_DIR}/{name}_results.csv", index=False)
    print("saved", name, "best =", art["best_name"], flush=True)


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("clustering", "all"): run_clustering()
    if w in ("classification", "all"): run_classification()
    if w in ("regression", "all"): run_regression()
