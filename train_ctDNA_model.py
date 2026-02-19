import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored


# =========================
# CONFIG
# =========================
DATA_PATH = "breast_cancer_ctnda_clinical_raw.csv"

# Time is in MONTHS
TIME_COL = "OS at baseline"
EVENT_COL = "Status OS"

# Display timepoints in months
DISPLAY_TIMEPOINTS = [6, 12, 24, 36, 60]

# Risk bands by risk-score quantiles
RISK_Q_LOW = 0.33
RISK_Q_HIGH = 0.66

# Slider bounds (robust to outliers)
BOUNDS_Q_LOW = 0.01
BOUNDS_Q_HIGH = 0.99

# Model save path
ARTIFACT_PATH = "clinical_rsf_artifact.joblib"


# =========================
# 1) LOAD + CLEAN
# =========================
df = pd.read_csv(DATA_PATH, sep=",", na_values=["NA", "NaN", ""])
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

cols_keep = [
    "Patient",
    "Mutation",
    "MAF of gene used at baseline",
    "CTCs counts at baseline",
    "AGE",
    TIME_COL,
    EVENT_COL,
]
df = df[cols_keep].copy()
df = df.dropna(subset=cols_keep).copy()

# Ensure numeric types
df["MAF of gene used at baseline"] = pd.to_numeric(df["MAF of gene used at baseline"], errors="coerce")
df["CTCs counts at baseline"] = pd.to_numeric(df["CTCs counts at baseline"], errors="coerce")
df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce")

df = df.dropna(subset=["MAF of gene used at baseline", "CTCs counts at baseline", "AGE", TIME_COL, EVENT_COL]).copy()
df = df[df[TIME_COL] >= 0].copy()


# =========================
# 2) MUTATION GROUPING
# =========================
keep_as_is = {"TP53", "PIK3CA", "GATA3", "ESR1", "PTEN", "ARID1A", "KMT2C", "BRCA1", "BRCA2"}

def group_mutation(m):
    m = str(m).strip()
    if m == "No mutated":
        return "No_mutation"
    if m in {"BRCA1", "BRCA2"}:
        return "BRCA_HRD"
    if m in keep_as_is:
        return m
    return "Other"

df["Mutation_grouped"] = df["Mutation"].apply(group_mutation)
df = df.drop(columns=["Mutation"])


# =========================
# 3) BUILD X, y (MONTHS)
# =========================
# IMPORTANT: Do NOT include Patient ID as a model feature
X = df.drop(columns=["Patient", TIME_COL, EVENT_COL]).copy()

y_time = df[TIME_COL].astype(float).values
y_event = df[EVENT_COL].astype(int).astype(bool).values
y = np.array(list(zip(y_event, y_time)), dtype=[("event", "?"), ("time", "<f8")])


numeric_features = [
    "MAF of gene used at baseline",
    "CTCs counts at baseline",
    "AGE",
]
categorical_features = ["Mutation_grouped"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ],
    remainder="drop"
)

rsf = RandomSurvivalForest(
    n_estimators=3000,
    min_samples_split=6,
    min_samples_leaf=2,
    max_features=0.5,
    max_depth=12,
    max_samples=0.8,
    n_jobs=-1,
    random_state=42
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("rsf", rsf),
    ]
)


# =========================
# 4) TRAIN / TEST EVAL
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

risk_test = model.predict(X_test)
cindex = concordance_index_censored(y_test["event"], y_test["time"], risk_test)[0]
print(f"RSF Test C-index (months): {cindex:.3f}")


# =========================
# 5) RISK BANDS (fit on full cohort)
# =========================
risk_full = model.predict(X)
q_low = float(np.quantile(risk_full, RISK_Q_LOW))
q_high = float(np.quantile(risk_full, RISK_Q_HIGH))

def risk_band(r):
    if r <= q_low:
        return "Low"
    elif r <= q_high:
        return "Intermediate"
    else:
        return "High"

risk_band_full = np.array([risk_band(r) for r in risk_full])


# =========================
# 6) PRECOMPUTE SURVIVAL CURVES (cohort + mutation-group means)
# =========================
rsf_fitted = model.named_steps["rsf"]
X_trans_full = model.named_steps["preprocess"].transform(X)
surv_fns = rsf_fitted.predict_survival_function(X_trans_full, return_array=False)

# Shared time grid within domain of ALL curves
t_max_all = min([float(fn.domain[1]) for fn in surv_fns])
times_grid = np.arange(0.0, float(t_max_all) + 1.0, 1.0, dtype=float)

surv_matrix = np.vstack([fn(times_grid) for fn in surv_fns])
cohort_surv_mean = surv_matrix.mean(axis=0)

mutation_groups = df["Mutation_grouped"].values
group_surv_means = {}
for g in sorted(pd.unique(mutation_groups)):
    idx = np.where(mutation_groups == g)[0]
    if len(idx) < 3:
        continue
    group_surv_means[g] = surv_matrix[idx].mean(axis=0)


# =========================
# 7) PERMUTATION IMPORTANCE (ON PROCESSED FEATURES)
# =========================
def cindex_scorer(estimator, X_array, y_struct):
    risk = estimator.predict(X_array)
    return concordance_index_censored(y_struct["event"], y_struct["time"], risk)[0]

perm_importances = None
try:
    X_test_trans = model.named_steps["preprocess"].transform(X_test)

    cat_names = (
        model.named_steps["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )
    feat_names = numeric_features + list(cat_names)

    perm = permutation_importance(
        model.named_steps["rsf"],
        X_test_trans,
        y_test,
        scoring=cindex_scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    perm_importances = (
        pd.Series(perm.importances_mean, index=feat_names)
        .sort_values(ascending=False)
        .head(15)
        .to_dict()
    )
except Exception as e:
    print("Permutation importance skipped due to error:", str(e))


# =========================
# 8) DATA-DRIVEN SLIDER BOUNDS (robust)
# =========================
slider_bounds = {
    "AGE": {
        "q_low": float(X["AGE"].quantile(BOUNDS_Q_LOW)),
        "q_high": float(X["AGE"].quantile(BOUNDS_Q_HIGH)),
        "median": float(X["AGE"].median()),
        "min": float(X["AGE"].min()),
        "max": float(X["AGE"].max()),
    },
    "MAF": {
        "q_low": float(X["MAF of gene used at baseline"].quantile(BOUNDS_Q_LOW)),
        "q_high": float(X["MAF of gene used at baseline"].quantile(BOUNDS_Q_HIGH)),
        "median": float(X["MAF of gene used at baseline"].median()),
        "min": float(X["MAF of gene used at baseline"].min()),
        "max": float(X["MAF of gene used at baseline"].max()),
    },
    "CTCs": {
        "q_low": float(X["CTCs counts at baseline"].quantile(BOUNDS_Q_LOW)),
        "q_high": float(X["CTCs counts at baseline"].quantile(BOUNDS_Q_HIGH)),
        "median": float(X["CTCs counts at baseline"].median()),
        "min": float(X["CTCs counts at baseline"].min()),
        "max": float(X["CTCs counts at baseline"].max()),
    },
}
allowed_mutation_groups = sorted(pd.unique(df["Mutation_grouped"]).tolist())


# =========================
# 9) SAVE ARTIFACT (for Streamlit)
# =========================
artifact = {
    "model": model,
    "cindex_test": float(cindex),
    "time_unit": "months",
    "times_grid": times_grid,
    "cohort_surv_mean": cohort_surv_mean,
    "group_surv_means": group_surv_means,
    "risk_thresholds": {"q_low": q_low, "q_high": q_high},
    "display_timepoints": DISPLAY_TIMEPOINTS,
    "perm_importances_top15": perm_importances,

    # Guardrails / UI realism
    "slider_bounds": slider_bounds,
    "allowed_mutation_groups": allowed_mutation_groups,

    # For nearest-neighbor display
    "train_rows_raw": X.copy(),                 # raw features (no Patient/time/event)
    "train_rows_trans": X_trans_full,           # transformed features
    "train_patient_ids": df["Patient"].values,  # kept separately for display
    "train_time": df[TIME_COL].values,
    "train_event": df[EVENT_COL].values,
    "train_mut_group": df["Mutation_grouped"].values,
    "train_risk": risk_full,
    "train_risk_band": risk_band_full,
}

joblib.dump(artifact, ARTIFACT_PATH)
print(f"Saved artifact: {ARTIFACT_PATH}")
