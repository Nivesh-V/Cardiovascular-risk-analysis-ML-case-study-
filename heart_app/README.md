# Cardio ML Studio — Streamlit app

Interactive dashboard + predictor built from three notebooks on `HeartFailureDataset.csv` (70k patients):

| Task | Target | Algorithms compared | Best one is used for predictions |
|---|---|---|---|
| Classification | `cardio` | 10 classifiers | highest test ROC-AUC |
| Regression | `ap_hi` | 10 regressors | highest test R² |
| Clustering | – (unsupervised) | K-Means, Agglomerative | highest silhouette |

Pages: **Dashboard** (filters, KPIs, charts) · **Predict** (choose target → enter the other features → result)
· **Model performance** · **Prediction log** (every prediction: inputs, result, algorithm type, algorithm → `prediction_log.csv`).

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Retrain (optional, ~30-60 min): `python train.py all`  (pre-trained models are already in `models/`).

## Deploy on Streamlit Community Cloud (free, public URL)
1. Create a GitHub repo and upload everything in this folder (keep `models/`, `data/`, `app.py`, `core.py`, `requirements.txt`).
2. Go to https://share.streamlit.io → sign in with GitHub → **New app**.
3. Pick the repo, branch `main`, main file `app.py` → **Deploy**. You get a URL like `https://<name>.streamlit.app`.

Notes
* The prediction log is a CSV on the server's disk. Streamlit Cloud resets the disk when the app restarts/redeploys, so use the **Download log** button to keep a permanent copy.
* If deployment complains about a scikit-learn version mismatch, run `python train.py all` with the same version and commit the new `models/` files.
* Clustering assigns a new patient to the nearest cluster centroid (Agglomerative clustering has no `predict`).
