import io, numpy as np, pandas as pd, streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core

st.set_page_config(page_title="Cardio ML Studio", page_icon="🫀", layout="wide")
BLUE, ORANGE, GREY = "#2a6fbb", "#d9822b", "#8a8f98"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "figure.dpi": 110, "axes.titlesize": 11, "axes.titleweight": "bold"})


def render_chart(fig, container, dpi=140):
    """Renders a chart with completely fixed canvas dimensions to prevent resizing."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none")
    buf.seek(0)
    container.image(buf, use_container_width=True)
    plt.close(fig)


@st.cache_data
def get_data():
    df = core.load_raw().drop_duplicates().copy()
    df["age_years"] = (df["age"] / 365.25).round(1)
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    df["gender_label"] = df["gender"].map({1: "Female", 2: "Male"})
    return df


@st.cache_resource
def get_art(task):
    return core.load_model(task)


df = get_data()
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict", "🏆 Model performance", "🗂️ Prediction log"])
st.sidebar.caption(f"Dataset: HeartFailureDataset.csv · {len(df):,} patients (duplicates removed)")


# ------------------------------------------------------------------ DASHBOARD
if page.startswith("📊"):
    st.title("🫀 Cardiovascular dataset dashboard")
    f1, f2 = st.columns(2)
    sex = f1.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
    lo, hi = float(df.age_years.min()), float(df.age_years.max())
    ages = f2.slider("Age range (years)", int(lo), int(hi) + 1, (int(lo), int(hi) + 1))
    d = df[df.gender_label.isin(sex) & df.age_years.between(*ages)]
    if d.empty:
        st.warning("No rows match the filters."); st.stop()

    k = st.columns(5)
    k[0].metric("Patients", f"{len(d):,}")
    k[1].metric("Cardiovascular disease rate", f"{d.cardio.mean():.1%}")
    k[2].metric("Mean age", f"{d.age_years.mean():.1f} y")
    k[3].metric("Mean BP (sys/dia)", f"{d.ap_hi.median():.0f}/{d.ap_lo.median():.0f}")
    k[4].metric("Mean BMI", f"{d.bmi.mean():.1f}")

    t1, t2, t3, t4 = st.tabs(["Distributions", "Risk factors", "Relationships", "Summary table"])
    with t1:
        c_sel, _ = st.columns([1, 1])
        col = c_sel.selectbox("Numeric feature", ["age_years", "bmi", "height", "weight", "ap_hi", "ap_lo"])

        lo_, hi_ = float(df[col].quantile(0.005)), float(df[col].quantile(0.995))
        bin_edges = np.linspace(lo_, hi_, 41)

        c1, c2 = st.columns(2)
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        fig1.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.18)
        for v, lab, colr in [(0, "No CVD", BLUE), (1, "CVD", ORANGE)]:
            sub = d.loc[d.cardio == v, col].clip(lo_, hi_)
            if len(sub) > 0:
                ax1.hist(sub, bins=bin_edges, alpha=0.6, label=lab, color=colr)
        ax1.set_xlim(lo_, hi_)
        ax1.set_xlabel(col)
        ax1.set_ylabel("Patients")
        ax1.set_title(f"{col} by cardiovascular status (outer 1% clipped)")
        ax1.legend(frameon=False)
        render_chart(fig1, c1)

        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        fig2.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.18)
        vc = d.cardio.value_counts().sort_index()
        cats = ["No CVD", "CVD"]
        vals = [int(vc.get(0, 0)), int(vc.get(1, 0))]
        ax2.bar(cats, vals, color=[BLUE, ORANGE], width=0.5)
        for i, v in enumerate(vals):
            ax2.text(i, v, f"{v:,}", ha="center", va="bottom")
        if max(vals) > 0:
            ax2.set_ylim(0, max(vals) * 1.15)
        ax2.set_ylabel("Patients")
        ax2.set_title("Target balance (cardio)")
        ax2.grid(axis="x", visible=False)
        render_chart(fig2, c2)
    with t2:
        feats = ["cholesterol", "gluc", "smoke", "alco", "active", "gender"]
        fig, axes = plt.subplots(2, 3, figsize=(11, 5.5)); 
        for ax, f in zip(axes.ravel(), feats):
            g = d.groupby(f).cardio.agg(["mean", "size"])
            ax.bar(g.index.astype(str), g["mean"], color=BLUE, width=.55)
            for i, (m, n) in enumerate(zip(g["mean"], g["size"])): ax.text(i, m, f"{m:.0%}", ha="center", va="bottom", fontsize=8)
            ax.set_title(f"CVD rate by {f}"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Bars show the share of patients with cardiovascular disease in each category.")
    with t3:
        c1, c2 = st.columns(2)
        num = d[["age_years", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]].corr()
        fig, ax = plt.subplots(figsize=(6, 4.8))
        fig.subplots_adjust(left=0.18, right=0.92, top=0.90, bottom=0.20)
        im = ax.imshow(num, cmap="coolwarm", vmin=-1, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(num))); ax.set_xticklabels(num.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(num))); ax.set_yticklabels(num.columns, fontsize=8)
        for i in range(len(num)):
            for j in range(len(num)): ax.text(j, i, f"{num.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
        ax.set_title("Correlation heatmap"); fig.colorbar(im, shrink=.8)
        render_chart(fig, c1)

        s = d[(d.ap_hi.between(60, 240)) & (d.ap_lo.between(40, 160))].sample(min(4000, len(d)), random_state=1)
        fig, ax = plt.subplots(figsize=(6, 4.8))
        fig.subplots_adjust(left=0.18, right=0.92, top=0.90, bottom=0.20)
        for v, lab, colr in [(0, "No CVD", BLUE), (1, "CVD", ORANGE)]:
            q = s[s.cardio == v]; ax.scatter(q.ap_lo, q.ap_hi, s=6, alpha=.35, color=colr, label=lab)
        ax.set_xlabel("ap_lo (diastolic)"); ax.set_ylabel("ap_hi (systolic)"); ax.legend(frameon=False, markerscale=2)
        ax.set_title("Blood pressure vs disease status (sample)")
        render_chart(fig, c2)
    with t4:
        st.dataframe(d[["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]].describe().T.round(2), use_container_width=True)
        st.dataframe(d.drop(columns=["gender_label"]).head(200), use_container_width=True)


# -------------------------------------------------------------------- PREDICT
elif page.startswith("🔮"):
    st.title("🔮 Predict")
    st.write("Pick the **target variable** you want predicted, then fill in the remaining patient details. "
             "The best-performing algorithm for that task is used automatically.")
    choice = st.selectbox("Target variable to predict", [
        "cardio — has cardiovascular disease? (Classification)",
        "ap_hi — systolic blood pressure (Regression)",
        "Patient segment — no target (Clustering)"])
    task = "Classification" if choice.startswith("cardio") else "Regression" if choice.startswith("ap_hi") else "Clustering"
    art = get_art(task)
    bm = art["best_metrics"]
    if task == "Classification":
        st.info(f"Best algorithm: **{art['best_name']}** (test ROC-AUC {bm['ROC-AUC']:.3f}, accuracy {bm['Accuracy']:.3f}) — chosen from 10 classifiers.")
    elif task == "Regression":
        st.info(f"Best algorithm: **{art['best_name']}** (test R² {bm['R2']:.3f}, RMSE {bm['RMSE']:.1f} mmHg) — chosen from 10 regressors.")
    else:
        st.info(f"Best algorithm: **{art['best_name']}** (silhouette {bm['Silhouette']:.3f}) — chosen from K-Means and Agglomerative.")

    target = art["target"]
    st.subheader("Input features" + (f" (all except `{target}`)" if target else ""))
    with st.form("pred"):
        c = st.columns(3); inp = {}
        inp["age_years"] = c[0].number_input("Age (years)", 18.0, 100.0, 50.0, 1.0)
        inp["gender"] = c[1].selectbox("Gender", [1, 2], format_func=lambda x: {1: "1 – Female", 2: "2 – Male"}[x])
        inp["height"] = c[2].number_input("Height (cm)", 120, 220, 165)
        inp["weight"] = c[0].number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5)
        if target != "ap_hi":
            inp["ap_hi"] = c[1].number_input("Systolic BP – ap_hi (mmHg)", 70, 250, 120)
        inp["ap_lo"] = c[2].number_input("Diastolic BP – ap_lo (mmHg)", 40, 150, 80)
        inp["cholesterol"] = c[0].selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {1: "1 – Normal", 2: "2 – Above normal", 3: "3 – Well above normal"}[x])
        inp["gluc"] = c[1].selectbox("Glucose", [1, 2, 3], format_func=lambda x: {1: "1 – Normal", 2: "2 – Above normal", 3: "3 – Well above normal"}[x])
        inp["smoke"] = c[2].selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
        inp["alco"] = c[0].selectbox("Alcohol intake", [0, 1], format_func=lambda x: "Yes" if x else "No")
        inp["active"] = c[1].selectbox("Physically active", [1, 0], format_func=lambda x: "Yes" if x else "No")
        go = st.form_submit_button("Predict", type="primary")
    if go:
        res = core.predict(task, inp)
        core.log_prediction(inp, res)
        st.subheader("Result")
        r = st.columns(4)
        r[0].metric("Prediction", res["result"])
        r[1].metric("Algorithm type", res["algorithm_type"].split(" - ")[0] + " / " + res["algorithm_type"].split(" - ")[1])
        r[2].metric("Algorithm used", res["algorithm"])
        if "confidence" in res: r[3].metric("Probability of CVD", f"{res['confidence']:.1%}")
        elif "rmse" in res: r[3].metric("Typical error (RMSE)", f"± {res['rmse']:.1f} mmHg")
        if task == "Classification":
            (st.error if res["result_code"] else st.success)("Model output only — this is a student ML project, not medical advice.")
        if task == "Clustering":
            prof = art["profile"]
            st.write("**Cluster profiles** (mean values of the 5,000-patient clustering sample):")
            st.dataframe(prof.rename_axis("cluster"), use_container_width=True)
            P = np.array(art["pca_points"]); L = np.array(art["pca_labels"])
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            for cl, colr in zip(sorted(set(L)), [BLUE, ORANGE, GREY]):
                ax.scatter(P[L == cl, 0], P[L == cl, 1], s=6, alpha=.35, color=colr, label=f"Cluster {cl}")
            ax.scatter(*res["point_2d"], s=140, marker="*", color="black", zorder=5, label="This patient")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False, markerscale=2)
            ax.set_title("PCA view of clusters"); st.pyplot(fig); plt.close(fig)
        st.caption("✅ Saved to the prediction log (see the Prediction log page).")


# ---------------------------------------------------------------- PERFORMANCE
elif page.startswith("🏆"):
    st.title("🏆 Model performance")
    task = st.radio("Task", list(core.TASKS), horizontal=True)
    art = get_art(task); tab = art["results"]
    st.success(f"Best model: **{art['best_name']}**")
    show = tab.drop(columns=[c for c in ["Best params"] if c in tab])
    st.dataframe(show.round(4), use_container_width=True, hide_index=True)
    if task == "Classification":
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        fig, ax = plt.subplots(figsize=(10, 4)); w = .16; x = np.arange(len(tab))
        for i, m in enumerate(metrics): ax.bar(x + i * w, tab[m], w, label=m)
        ax.set_xticks(x + 2 * w); ax.set_xticklabels(tab["Model"], rotation=30, ha="right"); ax.set_ylim(.5, .85)
        ax.legend(ncol=5, frameon=False); ax.set_title("Test-set metrics"); st.pyplot(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        for n, (f, t) in art["roc"].items(): ax.plot(f, t, lw=2 if n == art["best_name"] else 1, label=n)
        ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, frameon=False); ax.set_title("ROC curves"); st.pyplot(fig); plt.close(fig)
    elif task == "Regression":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].barh(tab["Model"][::-1], tab["R2"][::-1], color=BLUE); axes[0].set_title("Test R² (higher is better)")
        a, p = art["actual_vs_pred"]; axes[1].scatter(a, p, s=6, alpha=.3, color=BLUE)
        lo, hi = min(a), max(a); axes[1].plot([lo, hi], [lo, hi], "r--")
        axes[1].set_xlabel("Actual ap_hi"); axes[1].set_ylabel("Predicted"); axes[1].set_title(f"Actual vs predicted – {art['best_name']}")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        st.write("Linkage comparison (silhouette):", {k: round(v, 3) for k, v in art["linkage_scores"]})
        st.caption("New patients are assigned to the nearest cluster centroid of the winning algorithm.")
    with st.expander("Best hyper-parameters per model"):
        if "Best params" in tab: st.dataframe(tab[["Model", "Best params"]], hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------ LOG
else:
    st.title("🗂️ Prediction log")
    log = core.read_log()
    if log.empty:
        st.info("No predictions yet. Use the Predict page.")
    else:
        t = st.multiselect("Filter by task type", sorted(log.task_type.unique()), default=sorted(log.task_type.unique()))
        v = log[log.task_type.isin(t)].iloc[::-1]
        st.metric("Predictions stored", len(v))
        st.dataframe(v, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download log (CSV)", log.to_csv(index=False).encode(), "prediction_log.csv", "text/csv")
