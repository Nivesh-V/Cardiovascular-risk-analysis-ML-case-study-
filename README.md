# Cardiovascular Disease Analysis — Classification, Regression & Clustering

Machine-learning case study on a cardiovascular patient dataset, applying all three algorithm families
(10 classifiers, 10 regressors, 2 clustering methods) and deploying the results as an interactive Streamlit app.

**Live app:** `<paste your public URL here>`

## Dataset description
`HeartFailureDataset.csv` — 70,000 patients, 13 columns, no missing values or duplicates.

* **Features:** age (days), gender, height, weight, systolic / diastolic blood pressure (`ap_hi`, `ap_lo`),
  cholesterol and glucose level (1–3), smoking, alcohol intake, physical activity.
* **Targets:** `cardio` (cardiovascular disease, 1/0, ≈ 50 % balanced) for classification and `ap_hi` for regression.
  Clustering is unsupervised.
* **Preprocessing:** `id` dropped, age converted to years, BMI added, outliers clipped with the 1.5 × IQR rule,
  features scaled; PCA (95 % variance) for the supervised models.

## Problem statement
Using simple clinical measurements and lifestyle habits, (1) classify whether a patient has cardiovascular disease,
(2) predict systolic blood pressure, and (3) discover natural patient groups with different risk profiles.

## Results summary
Each algorithm was tuned with 3-fold `GridSearchCV` and evaluated on a held-out 20 % test set
(clustering: 5,000-patient sample, k = 3).

| Task | Target | Best algorithm | Result | Runner-up |
|---|---|---|---|---|
| Classification | `cardio` | MLP Neural Network | ROC-AUC 0.789, accuracy 72.4 % | Gradient Boosting (AUC 0.788) |
| Regression | `ap_hi` | Random Forest | R² 0.533, RMSE 11.2 mmHg | Gradient Boosting (R² 0.532) |
| Clustering | – | Agglomerative (average linkage) | Silhouette 0.307 | K-Means (0.215) |

Clustering found 3 groups: a large low-risk majority (48 % disease), a small lifestyle-risk group
(drinkers/smokers, 56 %) and a small metabolic-risk group (high cholesterol and glucose, 70 %).

## Environment setup
Python 3.10+ (tested on 3.13).
```bash
pip install -r requirements.txt     # streamlit, pandas, numpy, matplotlib, scikit-learn, joblib
```
To open the notebooks, also install `jupyter` and `seaborn`.

## How to run
1. **Notebooks:** put `HeartFailureDataset.csv` beside the notebooks and run all cells of
   `Cardiovascular_Combined_10_Classifiers.ipynb`, `Cardiovascular_Combined_10_Regressors.ipynb` and
   `HeartFailure_Clustering_Review2.ipynb`.
2. **Web app:** `streamlit run app.py` (opens at http://localhost:8501) — dashboard, predictions with the best
   model of each task, model comparison, and a prediction log.
3. **Retrain app models (optional):** `python train.py all`.

*Educational project — not medical advice.*
