import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ============================================================
# Config
# ============================================================
ARTIFACT_PATH = "clinical_rsf_artifact.joblib"
st.set_page_config(
    page_title="Breast Cancer Patient Monitoring",
    layout="wide",
    page_icon="🧬",
)

# ============================================================
# Global Styles (LIGHT clinical theme + soft oncology accents)
#   - smoother, more spacing, rounder corners
#   - mostly light background with gentle pink/violet hints
#   - UPDATED: tabs padding/centering + chart container breathing room
# ============================================================
st.markdown(
    """
<style>
/* ---------- Theme tokens ---------- */
:root{
  --bg0: #fafbff;
  --bg1: #f6f7ff;
  --panel: rgba(255,255,255,0.82);
  --panel2: rgba(255,255,255,0.92);
  --border: rgba(15,23,42,0.08);
  --border2: rgba(15,23,42,0.10);
  --text: rgba(15,23,42,0.92);
  --muted: rgba(15,23,42,0.66);
  --muted2: rgba(15,23,42,0.52);

  /* Oncology accent (soft rose/violet) */
  --accent: #ff4da6;
  --accentSoft: rgba(255,77,166,0.12);
  --accent2: #7c3aed;
  --accent2Soft: rgba(124,58,237,0.10);

  /* Risk colors */
  --low: #10b981;
  --mid: #f59e0b;
  --high:#ef4444;

  --shadow: 0 18px 38px rgba(2,6,23,0.08);
  --shadow2: 0 12px 24px rgba(2,6,23,0.06);

  --radius: 22px;     /* rounder */
  --radius2: 18px;
}

/* ---------- Background ---------- */
.stApp{
  background:
    radial-gradient(1200px 700px at 8% 0%, rgba(255,77,166,0.20), transparent 55%),
    radial-gradient(900px 600px at 96% 8%, rgba(124,58,237,0.14), transparent 58%),
    radial-gradient(800px 520px at 40% 96%, rgba(16,185,129,0.10), transparent 60%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
  color: var(--text);
}

/* ---------- Layout spacing ---------- */
.main .block-container{
  padding-top: 1.6rem;
  padding-bottom: 2.6rem;
  max-width: 1220px;
}

/* Typography */
h1, h2, h3 { letter-spacing: -0.02em; color: var(--text); }
p, li, span { color: var(--muted); }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.72);
  border-right: 1px solid var(--border);
  box-shadow: 0 12px 34px rgba(2,6,23,0.05);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span{
  color: rgba(15,23,42,0.88) !important;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div{
  background: rgba(255,255,255,0.95) !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  border-radius: 16px !important;
  box-shadow: 0 10px 18px rgba(2,6,23,0.04);
}
.stSlider [data-baseweb="slider"]{
  background: rgba(255,255,255,0.65);
  border-radius: 16px;
  border: 1px solid rgba(15,23,42,0.10);
}
.stSlider > div { padding-top: 0.35rem; }

/* Alerts */
div[data-testid="stAlert"]{
  border-radius: 18px !important;
  border: 1px solid rgba(15,23,42,0.10) !important;
  box-shadow: var(--shadow2);
}

/* DataFrames */
div[data-testid="stDataFrame"]{
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow2);
}

/* Charts (UPDATED to be more pillowy, esp. y-axis breathing room) */
div[data-testid="stLineChart"]{
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 28px !important;
  padding: 18px 18px 18px 20px !important; /* extra left padding helps y-axis */
  box-shadow: var(--shadow2);
  overflow: hidden !important;             /* hide sharp inner corners */
}
div[data-testid="stLineChart"] > div{
  border-radius: 22px !important;
  overflow: hidden !important;
}

/* ---------- Tabs (UPDATED padding + centering) ---------- */
button[data-baseweb="tab"]{
  background: rgba(255,255,255,0.70) !important;
  border-radius: 18px 18px 0 0 !important;
  border: 1px solid rgba(15,23,42,0.10) !important;

  padding: 12px 18px !important;   /* breathing room */
  margin-right: 8px !important;    /* separation between tabs */
  min-height: 44px !important;     /* better touch target */
}
button[data-baseweb="tab"] > div{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 10px !important;            /* space between emoji and text */
  padding: 0px 6px !important;     /* label away from edges */
  line-height: 1 !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  background: linear-gradient(180deg, rgba(255,77,166,0.16), rgba(124,58,237,0.10)) !important;
  border-color: rgba(255,77,166,0.26) !important;
}

/* ---------- Components ---------- */
.hr{
  height: 1px;
  background: rgba(15,23,42,0.08);
  margin: 18px 0px 10px 0px;
}

/* pill */
.pill{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.75);
  font-size: 13px;
  color: rgba(15,23,42,0.82);
  box-shadow: 0 10px 18px rgba(2,6,23,0.04);
}
.dot{ width: 8px; height: 8px; border-radius: 50%; display:inline-block; }

/* Cards */
.card{
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 26px; /* extra round */
  padding: 18px 20px;
  box-shadow: var(--shadow);
}
.cardHeader{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap: 14px;
  margin-bottom: 12px;
}
.cardTitle{
  font-size: 16px;
  font-weight: 850;
  color: rgba(15,23,42,0.92);
  letter-spacing: -0.01em;
}
.cardNote{
  font-size: 13px;
  color: rgba(15,23,42,0.62);
  margin-top: 3px;
  line-height: 1.35;
}

/* Metric cards */
.metricCard{
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.84));
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 24px;
  padding: 16px 18px;
  box-shadow: var(--shadow2);
}
.metricLabel{
  font-size: 12px;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: rgba(15,23,42,0.52);
  margin-bottom: 7px;
}
.metricValue{
  font-size: 30px;
  font-weight: 900;
  color: rgba(15,23,42,0.92);
  line-height: 1.12;
}
.metricSub{
  margin-top: 7px;
  font-size: 13px;
  color: rgba(15,23,42,0.62);
  line-height: 1.35;
}

/* Soft accent badge */
.badgeAccent{
  border: 1px solid rgba(255,77,166,0.20);
  background: linear-gradient(180deg, rgba(255,77,166,0.14), rgba(124,58,237,0.10));
  color: rgba(15,23,42,0.86);
}

/* Footer */
.footer{
  margin-top: 20px;
  text-align: center;
  color: rgba(15,23,42,0.46);
  font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Load artifact
# ============================================================
@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_PATH)

artifact = load_artifact()
model = artifact["model"]

TIME_UNIT = artifact.get("time_unit", "months")
times_grid = artifact["times_grid"]
cohort_surv_mean = artifact["cohort_surv_mean"]
group_surv_means = artifact["group_surv_means"]
risk_thresholds = artifact["risk_thresholds"]
display_timepoints = artifact["display_timepoints"]
perm_top = artifact.get("perm_importances_top15")

slider_bounds = artifact.get("slider_bounds", {})
allowed_groups = artifact.get(
    "allowed_mutation_groups",
    ["No_mutation", "TP53", "PIK3CA", "GATA3", "ESR1", "PTEN", "ARID1A", "KMT2C", "BRCA_HRD", "Other"],
)

train_raw = artifact["train_rows_raw"]
train_trans = artifact["train_rows_trans"]
train_patient_ids = artifact["train_patient_ids"]
train_time = artifact["train_time"]
train_event = artifact["train_event"]
train_risk = artifact["train_risk"]
train_risk_band = artifact["train_risk_band"]

q_low = risk_thresholds["q_low"]
q_high = risk_thresholds["q_high"]

# ============================================================
# Helpers
# ============================================================
def assign_risk_band(r):
    if r <= q_low:
        return "Low"
    elif r <= q_high:
        return "Intermediate"
    else:
        return "High"

def band_color(band):
    return {"Low": "🟢", "Intermediate": "🟠", "High": "🔴"}.get(band, "⚪")

def band_hex(band):
    return {"Low": "#10b981", "Intermediate": "#f59e0b", "High": "#ef4444"}.get(band, "#94a3b8")

def predict_survival_for_input(input_df):
    rsf = model.named_steps["rsf"]
    X_trans = model.named_steps["preprocess"].transform(input_df)
    fn = rsf.predict_survival_function(X_trans, return_array=False)[0]

    _, t_max = fn.domain
    t_max = min(float(t_max), float(times_grid.max()))
    times = np.arange(0.0, t_max + 1.0, 1.0, dtype=float)
    surv = fn(times)

    idx = np.where(surv <= 0.5)[0]
    median = None if len(idx) == 0 else float(times[idx[0]])

    risk = float(model.predict(input_df)[0])
    band = assign_risk_band(risk)

    return times, surv, median, risk, band

def format_median(median, max_followup):
    if median is None:
        return f"> {int(max_followup)} {TIME_UNIT} (median not reached)"
    yrs = median / 12.0
    return f"{median:.0f} {TIME_UNIT} (~{yrs:.1f} years)"

def survival_at_time(times, surv, t):
    if t > times.max():
        return None
    return float(np.interp(t, times, surv))

def nearest_neighbors(input_df, k=5):
    x = model.named_steps["preprocess"].transform(input_df)
    d = np.linalg.norm(train_trans - x, axis=1)
    idx = np.argsort(d)[:k]

    out = train_raw.iloc[idx].copy()
    out.insert(0, "Patient", train_patient_ids[idx])
    out["OS_months"] = train_time[idx]
    out["Event(Status OS)"] = train_event[idx]
    out["RiskScore"] = train_risk[idx]
    out["RiskBand"] = train_risk_band[idx]
    out["Distance"] = d[idx]
    return out

def risk_percentile(risk_value):
    return float((train_risk <= risk_value).mean() * 100.0)

def flag_ood(value, qlo, qhi, name):
    if value < qlo or value > qhi:
        st.warning(
            f"{name} looks outside the model’s typical training range "
            f"(~{qlo:.2f} to ~{qhi:.2f}). Predictions may be less reliable."
        )

def metric_card(label, value, sub=None, pill_text=None, pill_color=None, pill_class=""):
    pill_html = ""
    if pill_text is not None:
        dot = f"<span class='dot' style='background:{pill_color or '#94a3b8'};'></span>"
        pill_html = f"<span class='pill {pill_class}'>{dot}{pill_text}</span>"

    sub_html = f"<div class='metricSub'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
        <div class="metricCard">
          <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:12px;">
            <div style="flex:1;">
              <div class="metricLabel">{label}</div>
              <div class="metricValue">{value}</div>
              {sub_html}
            </div>
            <div>{pill_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# UI Header
# ============================================================
st.markdown(
    """
<div style="display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:14px;">
  <div>
    <div style="
      font-size:36px; font-weight:950; color:rgba(15,23,42,0.94);
      letter-spacing:-0.03em; line-height:1.08;">
      Breast Cancer Patient Monitoring
    </div>
    <div style="margin-top:8px; font-size:14px; color:rgba(15,23,42,0.64); max-width:920px; line-height:1.45;">
      Predicts patient survival time using liquid biopsy markers (ctDNA mutation group, MAF, CTCs) and patient age.
    </div>
  </div>

  <div style="display:flex; flex-direction:column; gap:10px; align-items:flex-end; margin-top:2px;">
    <span class="pill badgeAccent">
      <span class="dot" style="background:rgba(255,77,166,0.95);"></span>
      Prototype • Research-use only
    </span>
    <div style="font-size:12px; color:rgba(15,23,42,0.55); text-align:right;">
      Not validated for clinical decision-making
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar inputs
# ============================================================
age_qlo = slider_bounds.get("AGE", {}).get("q_low", float(train_raw["AGE"].quantile(0.01)))
age_qhi = slider_bounds.get("AGE", {}).get("q_high", float(train_raw["AGE"].quantile(0.99)))
age_med = slider_bounds.get("AGE", {}).get("median", float(train_raw["AGE"].median()))

maf_qlo = slider_bounds.get("MAF", {}).get("q_low", float(train_raw["MAF of gene used at baseline"].quantile(0.01)))
maf_qhi = slider_bounds.get("MAF", {}).get("q_high", float(train_raw["MAF of gene used at baseline"].quantile(0.99)))
maf_med = slider_bounds.get("MAF", {}).get("median", float(train_raw["MAF of gene used at baseline"].median()))

ctc_qlo = slider_bounds.get("CTCs", {}).get("q_low", float(train_raw["CTCs counts at baseline"].quantile(0.01)))
ctc_qhi = slider_bounds.get("CTCs", {}).get("q_high", float(train_raw["CTCs counts at baseline"].quantile(0.99)))
ctc_med = slider_bounds.get("CTCs", {}).get("median", float(train_raw["CTCs counts at baseline"].median()))

age_min = int(np.floor(age_qlo))
age_max = int(np.ceil(age_qhi))
age_default = int(np.clip(round(age_med), age_min, age_max))

maf_max = float(max(1.0, round(maf_qhi, 2)))
maf_default = float(np.clip(round(maf_med, 2), 0.0, maf_max))

ctc_max = int(max(50, np.ceil(ctc_qhi)))
ctc_default = int(np.clip(round(ctc_med), 0, ctc_max))

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 4px 8px 4px;">
          <div style="font-size:14px; font-weight:900; color:rgba(15,23,42,0.92); letter-spacing:-0.01em;">
            Patient Inputs
          </div>
          <div style="margin-top:6px; font-size:12px; color:rgba(15,23,42,0.60); line-height:1.35;">
            Set baseline blood markers and age to estimate survival risk and life expectancy.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    mutation = st.selectbox("ctDNA mutation group", options=allowed_groups, index=0)

    maf = st.number_input(
        "Mutant Allele Frequency (MAF)",
        min_value=0.0,
        max_value=float(maf_max),
        value=float(0.0),
        step=0.1,
        help="Baseline ctDNA mutant allele frequency.",
    )

    ctcs = st.slider(
        "Circulating Tumour Cells (CTCs)",
        min_value=0,
        max_value=int(ctc_max),
        value=int(0),
        step=1,
        help="Baseline CTC count.",
    )

    age = st.slider(
        "Age",
        min_value=int(age_min),
        max_value=int(age_max),
        value=int(age_default),
        step=1,
    )

# OOD checks
flag_ood(age, age_qlo, age_qhi, "Age")
flag_ood(maf, maf_qlo, maf_qhi, "MAF")
flag_ood(ctcs, ctc_qlo, ctc_qhi, "CTCs")

# ============================================================
# Predict
# ============================================================
input_df = pd.DataFrame(
    [
        {
            "MAF of gene used at baseline": float(maf),
            "CTCs counts at baseline": int(ctcs),
            "AGE": int(age),
            "Mutation_grouped": mutation,
        }
    ]
)

times, surv, median, risk, band = predict_survival_for_input(input_df)

max_followup = float(times_grid.max())
risk_pct = risk_percentile(risk)

# ============================================================
# Top Summary
# ============================================================
st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

summary_left, summary_right = st.columns([1.15, 0.85], gap="large")

with summary_left:
    st.markdown(
        """
        <div class="card">
          <div class="cardHeader">
            <div>
              <div class="cardTitle">Clinical Summary</div>
              <div class="cardNote">Risk stratification and survival estimate for the current inputs.</div>
            </div>
            <span class="pill badgeAccent">
              <span class="dot" style="background:rgba(255,77,166,0.95);"></span>
              Liquid biopsy
            </span>
          </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns([1, 1], gap="large")
    with m1:
        metric_card(
            label="Risk band",
            value=f"{band_color(band)} {band}",
            sub=f"Risk percentile: {risk_pct:.0f}th (vs training cohort)",
            pill_text="Cohort-relative",
            pill_color="rgba(15,23,42,0.45)",
        )
    with m2:
        metric_card(
            label="Estimated Survival Time",
            value=format_median(median, max_followup),
            sub="Median OS estimate from the cohort survival model",
            pill_text=f"Max follow-up ~{int(max_followup)} {TIME_UNIT}",
            pill_color="rgba(255,77,166,0.85)",
            pill_class="badgeAccent",
        )

    st.markdown("</div>", unsafe_allow_html=True)

with summary_right:
    st.markdown(
        f"""
        <div class="card">
          <div class="cardHeader">
            <div>
              <div class="cardTitle">Model context</div>
              <div class="cardNote">How does the model make decisions?.</div>
            </div>
            <span class="pill">
              <span class="dot" style="background:{band_hex(band)};"></span>
              Current: {band} risk
            </span>
          </div>

          <div style="font-size:13px; color:rgba(15,23,42,0.74); margin-top:2px; line-height:1.35;">
            Outputs are determined using <b>liquid biopsy data only</b> (blood test):
          </div>

          <ul style="margin:12px 0 12px 0; padding-left:18px; color:rgba(15,23,42,0.62); font-size:13px; line-height:1.5;">
            <li>ctDNA mutation group (e.g., TP53, PIK3CA)</li>
            <li>Circulating Tumour Cells (CTCs) count</li>
            <li>Mutant Allele Frequency (MAF)</li>
            <li>Patient age</li>
          </ul>

          <div style="font-size:12.5px; color:rgba(15,23,42,0.56); line-height:1.45;">
            Backed by survival model (RSF) with C-index score of 0.795
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Main Content (Tabs)
# ============================================================
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Survival", "🧪 Drivers", "👥 Similar patients"])

with tab1:
    st.markdown(
        """
        <div class="card">
          <div class="cardHeader">
            <div>
              <div class="cardTitle">Survival probabilities</div>
              <div class="cardNote">Survival probability at key timepoints for the inputs.</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []
    for t in display_timepoints:
        p = survival_at_time(times, surv, t)
        rows.append({f"Time ({TIME_UNIT})": t, "Survival probability": None if p is None else round(p, 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card" style="margin-top:18px;">
          <div class="cardHeader">
            <div>
              <div class="cardTitle">Survival curve overlays</div>
              <div class="cardNote">Patient curve compared to cohort and (if available) mutation-group average.</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    plot_df = pd.DataFrame({"month": times, "Patient (baseline)": surv})
    plot_df["Cohort average"] = np.interp(times, times_grid, cohort_surv_mean)

    if mutation in group_surv_means:
        plot_df[f"{mutation} group avg"] = np.interp(times, times_grid, group_surv_means[mutation])

    st.line_chart(plot_df.set_index("month"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown(
        """
        <div class="card">
          <div class="cardHeader">
            <div>
              <div class="cardTitle">Global drivers (permutation importance)</div>
              <div class="cardNote">Feature importance computed on the training cohort (if available).</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    if perm_top is None:
        st.info("Permutation importance not available (it may have been skipped during training).")
    else:
        imp_df = pd.DataFrame({"feature": list(perm_top.keys()), "importance": list(perm_top.values())})
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown(
        """
        <div class="card">
          <div class="cardHeader">
            <div>
              <div class="cardTitle">Similar patients (nearest neighbors)</div>
              <div class="cardNote">Nearest neighbors in the model’s transformed feature space (scaled numeric + one-hot mutation group).</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    nn = nearest_neighbors(input_df, k=5)
    st.dataframe(nn, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div style="margin-top:12px; font-size:12.5px; color:rgba(15,23,42,0.56); line-height:1.45;">
          This helps contextualize predictions with real cohort examples rather than presenting a single number in isolation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown(
    """
<div class="footer">
  Built for research exploration • Liquid biopsy/ctDNA model • UI optimized
</div>
""",
    unsafe_allow_html=True,
)
