# PRD: Airport Delay Predictor & Station Health Score

**Goal:** one small, easy-to-explain project that shows every skill in United's Data Science Intern JD: Python, SQL, Random Forest / boosting / KMeans, probability & statistics, metric design, visualization, and plain-English communication.

**Hard rule: keep it simple.** One notebook, one SQL file, one README. No web app, no API, no deep learning, no Docker, no extra frameworks. The owner must be able to explain every line in an interview.

---

## 0. Rules for the coding agent
1. Build exactly what is listed here. Do not add features.
2. Every code section is preceded by a short markdown cell in plain English: what we are doing and why.
3. Short, readable code. Comments only where the reason is not obvious.
4. **Never invent numbers.** All results in the README must come from cells that actually ran.
5. Fixed random seed (42) everywhere.
6. If a dataset column differs from what is written here, adapt to the real column names and mention it in the README.
7. The reference code in Section 4 is the starting point. Copy it into the notebook cell by cell (each `# CELL` block = one code cell). Fix small bugs if something does not run, but keep the logic and structure.
8. Build the notebook with `nbformat` (a small throwaway script) or write it directly, then execute it with `jupyter nbconvert --to notebook --execute --inplace flight_ops.ipynb` to confirm "Restart and run all" works.
9. Do not hard-code any result in markdown cells or the README. Read them from the executed notebook outputs.

---

## 1. Dataset
- Kaggle "2015 Flight Delays and Cancellations" (usdot/flight-delays): `flights.csv`, `airlines.csv`, `airports.csv`.
- If unavailable, use any public US BTS on-time-performance sample that has airline, origin airport, scheduled departure, departure delay and cancellation fields.
- Take a random sample of about 200,000 flights (seed 42) to keep everything fast. Do not commit raw data; add a `data/README.md` with the download link and put `data/` in `.gitignore`.
- Expected `flights.csv` columns used: `MONTH, DAY, DAY_OF_WEEK, AIRLINE, ORIGIN_AIRPORT, SCHEDULED_DEPARTURE (HHMM), DEPARTURE_DELAY, DISTANCE, CANCELLED`. In this dataset `ORIGIN_AIRPORT` sometimes holds numeric codes instead of 3-letter IATA codes; keep only rows with 3-letter codes.

---

## 2. Notebook sections (`flight_ops.ipynb`)

**A. Load and clean (Python/pandas)**
- Load the three tables, sample, check missing values, drop cancelled flights for the delay model (keep them for the cancellation rate metric).
- Define target `delayed = 1` if departure delay > 15 minutes.
- Report the class balance.

**B. SQL analysis (SQL)**
- Load the tables into SQLite (`sqlite3` + `pandas.to_sql`).
- Put the queries in `analysis.sql` and run them from the notebook. Exactly 5 queries:
  1. Delay rate by airline (join flights to airlines).
  2. Delay rate by scheduled departure hour.
  3. Top 5 worst origin airports (minimum flight count filter, `HAVING`).
  4. `RANK()` window function: rank airlines within each of the top 5 airports by delay rate.
  5. A CTE plus `LAG()`: daily delay rate and day-over-day change.
- Under each result, one sentence on what an ops team would learn.

**C. Statistics**
- Plot the delay rate with a 95% confidence interval for two groups (for example evening vs morning departures).
- Run one hypothesis test (chi-square test of independence or two-proportion z-test) on whether the difference is real. State the null hypothesis, the p-value and the conclusion in plain English.
- One sentence on correlation vs causation for this finding.

**D. Prediction model (ML)**
- **Features known before departure only:** month, day of week, scheduled departure hour, airline, origin airport, distance. One-hot encode categoricals (limit origin airports to the top 30, rest = "OTHER").
- **Do not use** arrival delay, delay-reason columns or anything only known after departure. Add a markdown cell explaining this is data leakage.
- **Split by time**: train on earlier months, test on the last 2-3 months. Explain why in one line.
- Models: Logistic Regression (baseline), Random Forest, XGBoost (use sklearn `GradientBoostingClassifier` if XGBoost will not install).
- Handle imbalance with `class_weight` / `scale_pos_weight`. Explain why accuracy alone is misleading here.
- Evaluate each model: precision, recall, F1, ROC-AUC, confusion matrix. Present one comparison table.
- Feature importance chart for the best model, with 3 sentences of interpretation.
- Light tuning only: a small `GridSearchCV` (at most about 6 combinations, 3-fold) on the best model.
- In a markdown cell: why Random Forest (bagging, lower variance) vs XGBoost (boosting, lower bias), in 2 sentences each.

**E. KMeans clustering**
- Aggregate per origin airport (minimum flight count): average departure delay, delay rate, cancellation rate, flight volume.
- Scale the features, choose k with the elbow method and silhouette score, fit KMeans.
- Describe each cluster in one line (for example "high-volume, moderately delayed hubs") and list 3 example airports per cluster.
- One markdown line on KMeans limitations (needs scaling, sensitive to outliers, assumes round clusters).

**F. Operational health metric**
- Define a simple, transparent **Station Health Score** per airport, for example: `100 - (w1 * delay_rate% + w2 * cancel_rate% + w3 * avg_delay_min_scaled)`, with weights stated and justified in text. Keep it to 3 inputs.
- Show the 10 healthiest and 10 least healthy airports (volume filtered).
- Short markdown note: what this metric misses and how an ops team might adjust the weights.

**G. Visuals**
- 4 clean charts total (matplotlib or seaborn): delay rate by hour, model comparison, feature importance, airport clusters. Titles, axis labels, no clutter. Power BI is optional and out of scope.
- (The CI plot in section C is a statistics plot and the elbow/silhouette plot in section E is a diagnostic; the 4 "final" charts are the ones listed here.)

**H. Summary for an ops manager (last markdown cell)**
- Under 200 words, no jargon: what drives delays, what the model can and cannot do, one concrete recommendation (for example, where extra staffing would help), and one limitation.

---

## 3. Repo layout
```
flight-ops-project/
├── flight_ops.ipynb
├── analysis.sql
├── data/README.md
├── requirements.txt   (pandas, numpy, scikit-learn, xgboost, scipy, matplotlib, seaborn, jupyter)
├── .gitignore
└── README.md
```

---

## 4. Reference code (copy these files; fix only what breaks)

### 4.1 `requirements.txt`
```
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
seaborn
jupyter
```

### 4.2 `.gitignore`
```
data/*
!data/README.md
.ipynb_checkpoints/
__pycache__/
*.db
.venv/
.DS_Store
```

### 4.3 `data/README.md`
```markdown
# Data

Raw data is not committed to this repo.

1. Download "2015 Flight Delays and Cancellations" from Kaggle:
   https://www.kaggle.com/datasets/usdot/flight-delays
2. Unzip and place these three files in this folder:
   - `flights.csv`
   - `airlines.csv`
   - `airports.csv`
```

### 4.4 `analysis.sql`
The notebook splits this file on the `-- name:` markers. Keep that marker format.
The SQLite table `flights` is created in the notebook (non-cancelled and cancelled rows; `delayed` is NULL for cancelled flights). Table `airlines` has `IATA_CODE, AIRLINE`.

```sql
-- name: q1_delay_by_airline
SELECT a.AIRLINE AS airline,
       COUNT(*) AS flights,
       ROUND(100.0 * AVG(f.delayed), 2) AS delay_rate_pct
FROM flights f
JOIN airlines a ON f.airline = a.IATA_CODE
WHERE f.cancelled = 0
GROUP BY a.AIRLINE
ORDER BY delay_rate_pct DESC;

-- name: q2_delay_by_hour
SELECT sched_hour AS hour,
       COUNT(*) AS flights,
       ROUND(100.0 * AVG(delayed), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
GROUP BY sched_hour
ORDER BY sched_hour;

-- name: q3_worst_origin_airports
SELECT origin,
       COUNT(*) AS flights,
       ROUND(100.0 * AVG(delayed), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
GROUP BY origin
HAVING COUNT(*) >= 500
ORDER BY delay_rate_pct DESC
LIMIT 5;

-- name: q4_rank_airlines_in_top5_airports
WITH top5 AS (
    SELECT origin
    FROM flights
    WHERE cancelled = 0
    GROUP BY origin
    HAVING COUNT(*) >= 500
    ORDER BY AVG(delayed) DESC
    LIMIT 5
),
stats AS (
    SELECT f.origin, f.airline,
           COUNT(*) AS flights,
           100.0 * AVG(f.delayed) AS delay_rate_pct
    FROM flights f
    JOIN top5 t ON f.origin = t.origin
    WHERE f.cancelled = 0
    GROUP BY f.origin, f.airline
    HAVING COUNT(*) >= 30
)
SELECT origin, airline, flights,
       ROUND(delay_rate_pct, 2) AS delay_rate_pct,
       RANK() OVER (PARTITION BY origin ORDER BY delay_rate_pct DESC) AS rank_in_airport
FROM stats
ORDER BY origin, rank_in_airport;

-- name: q5_daily_delay_with_change
WITH daily AS (
    SELECT flight_date,
           100.0 * AVG(delayed) AS delay_rate_pct
    FROM flights
    WHERE cancelled = 0
    GROUP BY flight_date
)
SELECT flight_date,
       ROUND(delay_rate_pct, 2) AS delay_rate_pct,
       ROUND(delay_rate_pct - LAG(delay_rate_pct) OVER (ORDER BY flight_date), 2) AS change_vs_prev_day
FROM daily
ORDER BY flight_date;
```

### 4.5 `flight_ops.ipynb` (cell by cell)

Each `# CELL` block is one code cell. Put the markdown text shown in `# MARKDOWN` blocks before it. Where a markdown block says "(write after seeing results)", the agent must write that text based on the real outputs, never guessed numbers.

```python
# MARKDOWN
# # Airport Delay Predictor & Station Health Score
# **Question:** Which flights and airports are most likely to be delayed, and can we predict it before departure?
# We use 2015 US flight data (sampled to ~200k flights). Departure delay > 15 minutes counts as "delayed".

# CELL 1: imports and settings
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
```

```python
# MARKDOWN (A. Load and clean)
# Load the three tables, take a random sample of 200,000 flights so everything runs fast,
# and keep only the columns we need. We keep cancelled flights for now (needed for the cancellation rate).

# CELL 2: load and sample
usecols = ["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT",
           "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "DISTANCE", "CANCELLED"]
flights = pd.read_csv("data/flights.csv", usecols=usecols, low_memory=False)
airlines = pd.read_csv("data/airlines.csv")
airports = pd.read_csv("data/airports.csv")

# Some rows use numeric airport codes; keep only 3-letter IATA codes
flights["ORIGIN_AIRPORT"] = flights["ORIGIN_AIRPORT"].astype(str)
flights = flights[flights["ORIGIN_AIRPORT"].str.len() == 3]

flights = flights.sample(n=200_000, random_state=SEED).reset_index(drop=True)
print(flights.shape)
flights.head()
```

```python
# CELL 3: missing values
flights.isna().sum()
```

```python
# CELL 4: build clean columns and the target
df = flights.copy()
df["flight_date"] = pd.to_datetime(
    dict(year=2015, month=df["MONTH"], day=df["DAY"])).dt.strftime("%Y-%m-%d")
df["sched_hour"] = (df["SCHEDULED_DEPARTURE"] // 100) % 24   # 2400 -> hour 0

# Cancelled flights have no departure delay, so delayed stays missing for them
df["delayed"] = np.where(df["DEPARTURE_DELAY"].isna(), np.nan,
                         (df["DEPARTURE_DELAY"] > 15).astype(float))

all_flights = df.copy()                                   # keeps cancelled flights (for cancel rate)
model_df = df[(df["CANCELLED"] == 0) & df["delayed"].notna()].copy()
model_df["delayed"] = model_df["delayed"].astype(int)

print("All flights:", len(all_flights), "| Non-cancelled with delay info:", len(model_df))
print("Cancellation rate: {:.2%}".format(all_flights["CANCELLED"].mean()))
```

```python
# CELL 5: class balance
balance = model_df["delayed"].value_counts(normalize=True).rename("share")
print(balance)
model_df["delayed"].value_counts().plot(kind="bar", title="Delayed (1) vs on time (0)")
plt.show()
```

```python
# MARKDOWN (B. SQL analysis)
# We load the data into SQLite and answer five business questions with SQL.
# The queries live in `analysis.sql`; the notebook just runs them.

# CELL 6: load into SQLite and define a helper that runs a named query
conn = sqlite3.connect(":memory:")
sql_df = all_flights.rename(columns={
    "AIRLINE": "airline", "ORIGIN_AIRPORT": "origin", "DAY_OF_WEEK": "day_of_week",
    "DEPARTURE_DELAY": "dep_delay", "DISTANCE": "distance", "CANCELLED": "cancelled",
    "MONTH": "month"})[["flight_date", "month", "day_of_week", "airline", "origin",
                        "sched_hour", "dep_delay", "distance", "cancelled", "delayed"]]
sql_df.to_sql("flights", conn, index=False, if_exists="replace")
airlines.to_sql("airlines", conn, index=False, if_exists="replace")

def run_sql(name):
    text = open("analysis.sql").read()
    blocks = {b.split("\n", 1)[0].strip(): b.split("\n", 1)[1]
              for b in text.split("-- name:")[1:]}
    return pd.read_sql(blocks[name], conn)
```

```python
# CELL 7: Query 1 - delay rate by airline
q1 = run_sql("q1_delay_by_airline")
q1
```
```
# MARKDOWN under Q1 (write after seeing results): one sentence on what an ops team learns.
```

```python
# CELL 8: Query 2 - delay rate by scheduled hour
q2 = run_sql("q2_delay_by_hour")
q2
```
```
# MARKDOWN under Q2 (write after seeing results): one sentence on what an ops team learns.
```

```python
# CELL 9: Query 3 - worst origin airports (min 500 flights)
q3 = run_sql("q3_worst_origin_airports")
q3
```
```
# MARKDOWN under Q3 (write after seeing results): one sentence on what an ops team learns.
```

```python
# CELL 10: Query 4 - rank airlines inside each of the top 5 airports (RANK window function)
q4 = run_sql("q4_rank_airlines_in_top5_airports")
q4
```
```
# MARKDOWN under Q4 (write after seeing results): one sentence on what an ops team learns.
```

```python
# CELL 11: Query 5 - daily delay rate and day-over-day change (CTE + LAG)
q5 = run_sql("q5_daily_delay_with_change")
q5.head(10)
```
```
# MARKDOWN under Q5 (write after seeing results): one sentence on what an ops team learns.
```

```python
# MARKDOWN (C. Statistics)
# Are evening flights really delayed more often than morning flights, or is it just noise?
# We compare delay rates with 95% confidence intervals and run a hypothesis test.
# Morning = scheduled hour 5-9, Evening = 17-21.

# CELL 12: delay rate and 95% confidence interval for two groups
morning = model_df[model_df["sched_hour"].between(5, 9)]["delayed"]
evening = model_df[model_df["sched_hour"].between(17, 21)]["delayed"]

def rate_ci(x):
    p, n = x.mean(), len(x)
    se = np.sqrt(p * (1 - p) / n)
    return p, 1.96 * se, n

groups = {"Morning (5-9h)": morning, "Evening (17-21h)": evening}
rows = []
for name, x in groups.items():
    p, err, n = rate_ci(x)
    rows.append({"group": name, "flights": n, "delay_rate": p, "ci_low": p - err, "ci_high": p + err})
ci_table = pd.DataFrame(rows)
print(ci_table)

plt.figure(figsize=(5, 4))
plt.bar(ci_table["group"], ci_table["delay_rate"] * 100,
        yerr=(ci_table["ci_high"] - ci_table["delay_rate"]) * 100, capsize=8,
        color=["#4C78A8", "#F58518"])
plt.ylabel("Delay rate (%)")
plt.title("Delay rate with 95% confidence interval")
plt.show()
```

```python
# CELL 13: chi-square test of independence
table = pd.crosstab(
    np.r_[np.zeros(len(morning)), np.ones(len(evening))],
    np.r_[morning.values, evening.values])
chi2, p_value, dof, _ = stats.chi2_contingency(table)
print("chi-square =", round(chi2, 2), "| p-value =", p_value)
```
```
# MARKDOWN (write after seeing results, in plain English)
# - Null hypothesis: the delay rate is the same for morning and evening departures.
# - State the real p-value from the cell above and whether we reject the null (p < 0.05).
# - One sentence on correlation vs causation: evening flights are delayed more often, but the time of day itself
#   is probably not the cause; delays build up through the day as earlier delays knock on to later flights.
#   (Keep this sentence only if the result actually shows evening > morning.)
```

```python
# MARKDOWN (D. Prediction model)
# **Data leakage warning.** We only use information known *before* the plane departs:
# month, day of week, scheduled hour, airline, origin airport, distance.
# We must NOT use arrival delay, delay-reason columns, or actual departure times, because they are only known after the flight
# and would make the model look perfect while being useless in real life.
#
# **Time-based split.** We train on January-September and test on October-December. In real life we predict the future from
# the past, so a random split would let the model "peek" at the future.

# CELL 14: features and time split
top_airports = model_df[model_df["MONTH"] <= 9]["ORIGIN_AIRPORT"].value_counts().head(30).index
X = pd.DataFrame({
    "month": model_df["MONTH"],
    "day_of_week": model_df["DAY_OF_WEEK"],
    "sched_hour": model_df["sched_hour"],
    "distance": model_df["DISTANCE"],
    "airline": model_df["AIRLINE"],
    "origin": np.where(model_df["ORIGIN_AIRPORT"].isin(top_airports), model_df["ORIGIN_AIRPORT"], "OTHER"),
})
X = pd.get_dummies(X, columns=["airline", "origin"], drop_first=False)
y = model_df["delayed"]

train_mask = model_df["MONTH"] <= 9
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Delay rate train: {:.3f} | test: {:.3f}".format(y_train.mean(), y_test.mean()))
```

```python
# MARKDOWN
# **Why accuracy alone is misleading.** Most flights are on time, so a model that says "never delayed" would score high accuracy
# and still be useless. We look at precision (when we warn, how often are we right), recall (how many real delays we catch),
# F1 and ROC-AUC. For operations, recall matters: a missed delay is more costly than a false alarm.
# We handle the imbalance with class weights.

# CELL 15: define the three models
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

logreg = make_pipeline(StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)

try:
    from xgboost import XGBClassifier
    boost = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          scale_pos_weight=neg / pos, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1)
    boost_name, use_sample_weight = "XGBoost", False
except ImportError:
    boost = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=SEED)
    boost_name, use_sample_weight = "GradientBoosting", True

models = {"Logistic Regression": logreg, "Random Forest": rf, boost_name: boost}
```

```python
# CELL 16: train and evaluate all models
results, matrices = [], {}
for name, model in models.items():
    if name == "GradientBoosting":
        w = np.where(y_train == 1, neg / pos, 1.0)
        model.fit(X_train, y_train, sample_weight=w)
    else:
        model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results.append({"model": name,
                    "precision": precision_score(y_test, pred),
                    "recall": recall_score(y_test, pred),
                    "f1": f1_score(y_test, pred),
                    "roc_auc": roc_auc_score(y_test, proba)})
    matrices[name] = confusion_matrix(y_test, pred)

results_df = pd.DataFrame(results).set_index("model").round(3)
results_df
```

```python
# CELL 17: confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, cm) in zip(axes, matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["On time", "Delayed"], yticklabels=["On time", "Delayed"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.show()
```

```python
# CELL 18: chart - model comparison  (FINAL CHART 2)
results_df.plot(kind="bar", figsize=(8, 4.5), rot=0)
plt.title("Model comparison on held-out test months")
plt.ylabel("Score"); plt.ylim(0, 1); plt.legend(loc="upper right")
plt.tight_layout(); plt.show()
```

```python
# MARKDOWN (D. cont.)
# **Light tuning.** We tune only the best model (by ROC-AUC) with a small grid (6 combinations, 3-fold).

# CELL 19: pick best model by ROC-AUC and tune it lightly
best_name = results_df["roc_auc"].idxmax()
print("Best model before tuning:", best_name)

grids = {
    "Random Forest": {"max_depth": [8, 12, 16], "min_samples_leaf": [20, 50]},
    "XGBoost": {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
    "GradientBoosting": {"max_depth": [3, 4, 5], "learning_rate": [0.05, 0.1]},
    "Logistic Regression": {"logisticregression__C": [0.1, 1, 10]},
}
search = GridSearchCV(models[best_name], grids[best_name], scoring="roc_auc", cv=3, n_jobs=-1)
search.fit(X_train, y_train)
best_model = search.best_estimator_
proba = best_model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("Best params:", search.best_params_)
print("Tuned test ROC-AUC: {:.3f} | recall: {:.3f} | precision: {:.3f}".format(
    roc_auc_score(y_test, proba), recall_score(y_test, pred), precision_score(y_test, pred)))
```

```python
# CELL 20: chart - feature importance  (FINAL CHART 3)
est = best_model[-1] if hasattr(best_model, "steps") else best_model
if hasattr(est, "feature_importances_"):
    imp = pd.Series(est.feature_importances_, index=X_train.columns)
else:                                   # logistic regression: use absolute coefficients
    imp = pd.Series(np.abs(est.coef_[0]), index=X_train.columns)
imp = imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 5))
imp[::-1].plot(kind="barh", color="#4C78A8")
plt.title("Top 15 features - " + best_name)
plt.xlabel("Importance"); plt.tight_layout(); plt.show()
```
```
# MARKDOWN (write after seeing the chart): exactly 3 sentences interpreting the top features.
```

```python
# MARKDOWN
# **Random Forest vs XGBoost.**
# - Random Forest builds many deep trees independently on random samples and averages them (bagging), which mainly lowers variance
#   and makes it hard to overfit.
# - XGBoost builds trees one after another, each fixing the mistakes of the previous ones (boosting), which mainly lowers bias and
#   often gives a more accurate model if tuned carefully.
```

```python
# MARKDOWN (E. KMeans clustering)
# We group airports with similar behaviour using four numbers per airport: average departure delay, delay rate, cancellation rate
# and flight volume. KMeans uses distances, so we scale the features first.

# CELL 21: airport-level table (min 500 flights)
ok = all_flights[all_flights["CANCELLED"] == 0].dropna(subset=["DEPARTURE_DELAY"])
by_airport = pd.DataFrame({
    "flights": all_flights.groupby("ORIGIN_AIRPORT").size(),
    "cancel_rate": all_flights.groupby("ORIGIN_AIRPORT")["CANCELLED"].mean(),
    "avg_delay_min": ok.groupby("ORIGIN_AIRPORT")["DEPARTURE_DELAY"].mean(),
    "delay_rate": ok.groupby("ORIGIN_AIRPORT")["delayed"].mean(),
}).dropna()
by_airport = by_airport[by_airport["flights"] >= 500]
print(len(by_airport), "airports with at least 500 flights")
by_airport.head()
```

```python
# CELL 22: scale and choose k (elbow + silhouette)
feat = by_airport[["avg_delay_min", "delay_rate", "cancel_rate", "flights"]].copy()
feat["flights"] = np.log1p(feat["flights"])        # volume is very skewed
Xs = StandardScaler().fit_transform(feat)

ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(list(ks), inertia, marker="o"); ax[0].set_title("Elbow method")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, marker="o", color="#F58518"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
plt.tight_layout(); plt.show()

best_k = list(ks)[int(np.argmax(sil))]
print("Chosen k (highest silhouette):", best_k)
```

```python
# CELL 23: fit KMeans and describe clusters
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xs)
by_airport["cluster"] = km.labels_
by_airport = by_airport.merge(airports[["IATA_CODE", "AIRPORT"]], left_index=True,
                              right_on="IATA_CODE", how="left").set_index("IATA_CODE")

summary = by_airport.groupby("cluster").agg(
    airports=("flights", "size"),
    avg_flights=("flights", "mean"),
    avg_delay_min=("avg_delay_min", "mean"),
    delay_rate=("delay_rate", "mean"),
    cancel_rate=("cancel_rate", "mean")).round(3)
print(summary)

for c in sorted(by_airport["cluster"].unique()):
    top3 = by_airport[by_airport["cluster"] == c].nlargest(3, "flights").index.tolist()
    print("Cluster", c, "- example airports:", top3)
```
```
# MARKDOWN (write after seeing results): one line per cluster, e.g. "high-volume, moderately delayed hubs", plus the example airports.
# Then one line on KMeans limitations: needs scaling, sensitive to outliers, assumes round clusters of similar size.
```

```python
# CELL 24: chart - airport clusters  (FINAL CHART 4)
plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=by_airport, x="flights", y="delay_rate", hue="cluster",
                palette="tab10", s=70)
plt.xscale("log")
plt.title("Airport clusters: volume vs delay rate")
plt.xlabel("Flights in sample (log scale)"); plt.ylabel("Delay rate")
plt.tight_layout(); plt.show()
```

```python
# MARKDOWN (F. Station Health Score)
# **Station Health Score (0-100, higher = healthier)**
# `score = 100 - (0.5 * delay_rate% + 2.0 * cancel_rate% + 0.3 * avg_delay_scaled)`
# - `delay_rate%`: share of flights delayed more than 15 minutes (0-100). Weight 0.5: the main signal of how reliable a station is.
# - `cancel_rate%`: share of flights cancelled. Weight 2.0: cancellations are rare, but each one hurts customers far more than a delay,
#   so one percentage point counts four times as much as a delay point.
# - `avg_delay_scaled`: average delay in minutes, rescaled 0-100 across airports. Weight 0.3: captures how *long* delays are, not just how often.
# Weights are judgement calls and are easy to change. The score is clipped to 0-100.

# CELL 25: compute the score
h = by_airport.copy()
h["delay_pct"] = h["delay_rate"] * 100
h["cancel_pct"] = h["cancel_rate"] * 100
h["avg_delay_scaled"] = 100 * (h["avg_delay_min"] - h["avg_delay_min"].min()) / (
    h["avg_delay_min"].max() - h["avg_delay_min"].min())

W_DELAY, W_CANCEL, W_AVG = 0.5, 2.0, 0.3
h["health_score"] = (100 - (W_DELAY * h["delay_pct"] + W_CANCEL * h["cancel_pct"]
                            + W_AVG * h["avg_delay_scaled"])).clip(0, 100).round(1)

cols = ["AIRPORT", "flights", "delay_pct", "cancel_pct", "avg_delay_min", "health_score"]
print("10 healthiest airports"); display(h.nlargest(10, "health_score")[cols].round(2))
print("10 least healthy airports"); display(h.nsmallest(10, "health_score")[cols].round(2))
```
```
# MARKDOWN (write after seeing results): short note on what the metric misses (weather, airport size, passengers affected,
# arrival delays, one-off events) and how ops could change weights (e.g. raise cancel weight before a holiday period);
# also one line on how it could be gamed (e.g. cancelling a flight early instead of delaying it, or padding the schedule).
```

```python
# MARKDOWN (G. Visuals)
# Chart 1 below: delay rate by scheduled hour. Charts 2-4 appear above (model comparison, feature importance, airport clusters).

# CELL 26: chart - delay rate by hour  (FINAL CHART 1)
plt.figure(figsize=(9, 4.5))
plt.plot(q2["hour"], q2["delay_rate_pct"], marker="o", color="#4C78A8")
plt.title("Delay rate by scheduled departure hour")
plt.xlabel("Scheduled departure hour"); plt.ylabel("Delay rate (%)")
plt.xticks(range(0, 24, 2)); plt.tight_layout(); plt.show()
```

```
# MARKDOWN (H. Summary for an ops manager) - LAST CELL
# Under 200 words, no jargon. Write it from the REAL results above. Must cover:
# 1. What drives delays (from hour chart, SQL results and feature importance)
# 2. What the model can and cannot do (use the real recall / precision / ROC-AUC numbers)
# 3. One concrete recommendation (e.g. where or when extra staffing would help)
# 4. One limitation (e.g. no weather or previous-flight information)
```

### 4.6 `README.md` template
The agent fills every `{{...}}` from executed notebook outputs. No placeholder may remain and no number may be invented.

```markdown
# Airport Delay Predictor & Station Health Score

{{2 sentences: which flights/airports get delayed, and can we predict it before departure?}}

**Dataset:** [2015 Flight Delays and Cancellations (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays), random sample of {{n}} flights (seed 42).

## How to run
1. `pip install -r requirements.txt`
2. Download the data into `data/` (see `data/README.md`).
3. `jupyter notebook flight_ops.ipynb` then "Restart and run all".

## Results
Model comparison (time-based split: train Jan-Sep, test Oct-Dec; delayed = departure delay > 15 min):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| {{filled from results_df}} | | | | |

### Key findings
1. {{finding from SQL / stats}}
2. {{finding from model / feature importance}}
3. {{finding from clustering or health score}}

### Station Health Score
`score = 100 - (0.5 * delay_rate% + 2.0 * cancel_rate% + 0.3 * avg_delay_scaled)`, clipped to 0-100.
{{one line justifying weights; list the real top and bottom 3 airports}}

## Limitations
- 2015 data only; patterns may have changed.
- No weather or previous-flight (knock-on delay) information.
- Sampled data (~200k flights), not the full year.
- A "delay" is defined as more than 15 minutes of departure delay.

## Next steps
- Add weather data.
- Add previous-flight delay as a feature.
- Try real-time scoring.
```

---

## 5. Acceptance criteria
- [ ] Notebook runs top to bottom with "Restart and run all" and no errors.
- [ ] The 5 SQL queries are in `analysis.sql` and run from the notebook.
- [ ] No leakage columns used as features; split is by time.
- [ ] Three models compared with precision, recall, F1, ROC-AUC and a confusion matrix.
- [ ] One hypothesis test with the null hypothesis and p-value stated in words.
- [ ] KMeans with k justified by silhouette.
- [ ] Station Health Score defined, justified and shown.
- [ ] Every number in the README matches a notebook output.
- [ ] Total notebook length stays reasonable (about 25-35 cells).
- [ ] All "write after seeing results" markdown cells are filled in from real outputs.

---

## 6. After the agent finishes (owner checklist)
- Re-run the notebook yourself and read every markdown cell.
- Be able to answer: why this target definition, why a time split, why recall matters here, what bagging vs boosting means, what the p-value says, why k was chosen, how the health score could be gamed.
- Only list on the resume what the notebook really does.
