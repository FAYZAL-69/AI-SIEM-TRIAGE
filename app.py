import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
from llm_helper import generate_report

st.set_page_config(page_title="AI SIEM Alert Triage", layout="wide")

# ──────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.impact-banner {
    background: linear-gradient(90deg, #1e3a8a, #2563eb);
    padding: 30px 20px;
    border-radius: 16px;
    text-align: center;
    margin: 20px 0 30px 0;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.impact-big {
    font-size: 62px;
    font-weight: 800;
    margin: 8px 0;
    color: #10b981;
    line-height: 1;
}
.impact-sub {
    font-size: 26px;
    margin: 0;
    opacity: 0.95;
}
.metric-box {
    background: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
}
.status-live  { background:#065f46; color:#6ee7b7; }
.status-wait  { background:#1e3a5f; color:#93c5fd; }
.status-error { background:#7f1d1d; color:#fca5a5; }
.explain-box {
    background: #111827;
    border-left: 4px solid #f59e0b;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 14px;
}
.explain-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid #1f2937;
}
.explain-bar {
    height: 8px;
    border-radius: 4px;
    background: #f59e0b;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered SIEM Alert Triage Assistant")

# ──────────────────────────────────────────────────────────────────
# MODEL LOADING — safe, with specific error messages
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    with open('expected_features.json', 'r') as f:
        expected = json.load(f)
    return model, expected

try:
    model, expected_features = load_resources()
except FileNotFoundError as e:
    st.error(f" Required model file not found: {e}")
    st.info("Make sure `model.pkl` and `expected_features.json` are in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f" Model loading failed: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────
# EXPLAINABILITY HELPER
# Produces per-alert human-readable reasons using model internals.
# Works with RandomForest/GradientBoosting (feature_importances_)
# and Linear models (coef_). Falls back gracefully for others.
# ──────────────────────────────────────────────────────────────────

# Thresholds for contextual labels on numeric features
FEATURE_CONTEXT = {
    'sbytes':  [(0,    1000,   "very low src bytes"),   (1000,  50000,  "moderate src bytes"),  (50000, 1e9,    "high src bytes")],
    'dbytes':  [(0,    1000,   "very low dst bytes"),   (1000,  50000,  "moderate dst bytes"),  (50000, 1e9,    "high dst bytes")],
    'rate':    [(0,    0.1,    "near-zero rate"),        (0.1,   0.85,   "normal rate"),          (0.85,  1e9,    "abnormal rate")],
    'dur':     [(0,    0.001,  "instant connection"),   (0.001, 10,     "normal duration"),      (10,    1e9,    "long connection")],
    'spkts':   [(0,    5,      "few src packets"),       (5,     100,    "moderate src packets"), (100,   1e9,    "high src packets")],
    'dpkts':   [(0,    5,      "few dst packets"),       (5,     100,    "moderate dst packets"), (100,   1e9,    "high dst packets")],
}

HIGH_RISK_SERVICES = {'ftp', 'ftp-data', 'smtp', 'dns', 'irc', 'telnet', 'rpc', 'netbios', '-'}
HIGH_RISK_PROTOS   = {'tcp', 'udp'}

def contextual_label(feature_name, value):
    """Return a human-readable description of a feature value."""
    base = feature_name.split('_')[0]  # handles one-hot like proto_tcp → proto
    if base in FEATURE_CONTEXT:
        for lo, hi, label in FEATURE_CONTEXT[base]:
            if lo <= value < hi:
                return label
    return f"{feature_name}={value:.3g}"

def explain_alert(row: pd.Series, feature_names: list, model) -> list[dict]:
    """
    Return top-N reasons why the model flagged this alert as high risk.
    Each reason is {'feature': str, 'value': float, 'importance': float, 'label': str}
    """
    reasons = []

    # --- Get global feature importances ---
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

    if importances is not None and len(importances) == len(feature_names):
        # Multiply global importance by the actual feature value (scaled 0-1 relative to max)
        row_vals = np.array([row.get(f, 0) for f in feature_names], dtype=float)
        max_vals = row_vals.max()
        if max_vals > 0:
            row_vals_norm = row_vals / max_vals
        else:
            row_vals_norm = row_vals

        # Combined score: how important × how present in THIS alert
        scores = importances * (0.5 + 0.5 * row_vals_norm)
        top_indices = np.argsort(scores)[::-1][:5]

        for idx in top_indices:
            fname  = feature_names[idx]
            fval   = row.get(fname, 0)
            imp    = float(importances[idx])
            label  = contextual_label(fname, fval)
            reasons.append({'feature': fname, 'value': fval, 'importance': imp, 'label': label})

    # --- Always add categorical context if present ---
    svc = str(row.get('service', '')).lower().strip()
    if svc in HIGH_RISK_SERVICES and not any(r['feature'] == 'service' for r in reasons):
        reasons.insert(0, {'feature': 'service', 'value': svc, 'importance': 0.0, 'label': f"high-risk service: {svc}"})

    proto = str(row.get('proto', '')).lower().strip()
    if not any(r['feature'] == 'proto' for r in reasons):
        reasons.insert(0, {'feature': 'proto', 'value': proto, 'importance': 0.0, 'label': f"protocol: {proto}"})

    return reasons[:5]  # cap at 5 reasons


def render_explanation(reasons: list[dict], score: float):
    """Render the explanation box using fully inline styles (no CSS class dependency)."""
    max_imp = max((r['importance'] for r in reasons if r['importance'] > 0), default=1) or 1
    rows_html = ""
    for r in reasons:
        bar_pct  = int((r['importance'] / max_imp) * 100) if r['importance'] > 0 else 10
        bar_html = (
            f'<span style="display:inline-block; height:8px; border-radius:4px; '
            f'background:#f59e0b; width:{bar_pct}px;"></span>'
        )
        rows_html += (
            f'<div style="display:flex; justify-content:space-between; align-items:center; '
            f'padding:6px 0; border-bottom:1px solid #1f2937;">'
            f'<span style="color:#f3f4f6; min-width:200px; font-size:14px;">{r["label"]}</span>'
            f'{bar_html}'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#111827; border-left:4px solid #f59e0b; padding:14px 18px; '
        f'border-radius:8px; margin:10px 0;">'
        f'<strong style="color:#f59e0b; font-size:15px;"> Why HIGH RISK? (score: {score:.2f})</strong>'
        f'<div style="margin-top:10px;">{rows_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────────────────────────
# SIDEBAR CONFIG
# ──────────────────────────────────────────────────────────────────
DATA_FILE  = "live_logs.json"
API_URL    = st.sidebar.text_input(
    "FastAPI endpoint URL",
    value="http://localhost:8000/logs",
    help="Your api_server.py endpoint that returns a JSON array of log records"
)
API_KEY    = st.sidebar.text_input(
    "API Key",
    value="your-secret-key",
    type="password",
    help="The SIEM_API_KEY you set when starting api_server.py"
)
API_HEADERS = {"X-API-Key": API_KEY}
live_mode  = st.sidebar.checkbox("📡 Enable Live Streaming (from API)", value=False)

st.sidebar.markdown("---")

# Sidebar connection status
if live_mode:
    # Try a lightweight HEAD/GET to show true API health
    try:
        probe = requests.get(API_URL, headers=API_HEADERS, timeout=2)
        if probe.status_code == 200:
            st.sidebar.markdown('<span class="status-pill status-live">🟢 API: CONNECTED</span>', unsafe_allow_html=True)
            st.sidebar.caption(f"Endpoint: {API_URL}")
        else:
            st.sidebar.markdown('<span class="status-pill status-wait">🟡 API: HTTP {probe.status_code}</span>', unsafe_allow_html=True)
    except requests.exceptions.ConnectionError:
        st.sidebar.markdown('<span class="status-pill status-error">🔴 API: UNREACHABLE</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Cannot connect to {API_URL}")
    except requests.exceptions.Timeout:
        st.sidebar.markdown('<span class="status-pill status-wait">🟡 API: TIMEOUT</span>', unsafe_allow_html=True)
    except Exception:
        st.sidebar.markdown('<span class="status-pill status-error">🔴 API: ERROR</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-pill status-wait">⬆️ Upload Mode</span>', unsafe_allow_html=True)
    st.sidebar.caption("Enable Live Streaming to connect the FastAPI feed")


# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────
df = pd.DataFrame()

if live_mode:
    st.markdown("**Live Streaming Mode** — Fetching logs directly from FastAPI")

    api_ok    = False
    live_data = []

    try:
        resp = requests.get(API_URL, headers=API_HEADERS, timeout=5)
        resp.raise_for_status()
        live_data = resp.json()
        api_ok    = True

    except requests.exceptions.ConnectionError:
        st.markdown('<span class="status-pill status-error">🔴 Cannot reach API server</span>', unsafe_allow_html=True)
        st.warning(f"Connection refused at `{API_URL}`. Is `api_server.py` running?")

    except requests.exceptions.Timeout:
        st.markdown('<span class="status-pill status-wait">🟡 API timeout</span>', unsafe_allow_html=True)
        st.warning("API did not respond within 5s. Will retry.")

    except requests.exceptions.HTTPError as e:
        st.markdown('<span class="status-pill status-error">🔴 API returned an error</span>', unsafe_allow_html=True)
        st.error(f"HTTP {resp.status_code}: {e}")

    except (ValueError, json.JSONDecodeError):
        st.error("❌ API returned invalid JSON. Check your FastAPI server output.")

    except Exception as e:
        st.error(f"❌ Unexpected error fetching from API: {e}")

    if not api_ok:
        st.error("❌ API not running — no data to display")
        st.stop()

    if not isinstance(live_data, list):
        st.error("❌ Invalid data format: expected a JSON array.")
        st.stop()

    if live_data:
        df = pd.DataFrame(live_data)
        st.markdown('<span class="status-pill status-live">🟢 Live — data received from API</span>', unsafe_allow_html=True)
        st.success(f"📡 {len(df):,} alerts fetched • {time.strftime('%H:%M:%S')} • Source: FastAPI")
    else:
        st.markdown('<span class="status-pill status-wait">🟡 Connected — no log entries yet</span>', unsafe_allow_html=True)
        st.info("Waiting for log_generator.py to push data…")


# 👇 SEPARATE BLOCK (IMPORTANT)
else:
    uploaded_file = st.file_uploader(
        "Upload CSV log file (UNSW-NB15 format or similar)",
        type="csv",
        help="Best with UNSW_NB15_training-set.csv"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df):,} rows.")
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            st.stop()
    else:
        st.info("Upload your dataset to see results.")


# ──────────────────────────────────────────────────────────────────
# SHARED ML + DISPLAY LOGIC
# ──────────────────────────────────────────────────────────────────
if not df.empty:
    try:
        total_rows = len(df)
        df_calc    = df.copy()

        REQUIRED_FEATURES = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate']
        missing_cols = [f for f in REQUIRED_FEATURES if f not in df_calc.columns]
        if missing_cols:
            st.error(f" Missing required columns: {missing_cols}")
            st.stop()

        # === ML PREDICTION ===
        X_calc = pd.get_dummies(df_calc[REQUIRED_FEATURES])
        X_calc = X_calc.reindex(columns=expected_features, fill_value=0)

        if X_calc.empty or X_calc.shape[1] == 0:
            st.error(" No matching features for the trained model. Check that your data matches training format.")
            st.stop()

        ml_start = time.time()

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_calc)[:, 1]
        else:
            st.warning(" Model does not support probability scores — using binary predictions instead.")
            probs = model.predict(X_calc).astype(float)

        ml_elapsed       = time.time() - ml_start
        df_calc['ml_score'] = probs

        # POINT 4 — division-by-zero guard
        throughput = (total_rows / ml_elapsed) if ml_elapsed > 0 else 0
        st.caption(f" ML inference: {ml_elapsed:.3f}s on {total_rows:,} rows  ({throughput:,.0f} rows/sec)")

        # === THRESHOLD SLIDER ===
        threshold = st.slider(
            "Risk Threshold (Higher = stricter triage)",
            min_value=0.3, max_value=0.9, value=0.70, step=0.01,
            help="Lower → catch more attacks (higher recall)  |  Higher → fewer false positives"
        )

        df_calc['is_high'] = df_calc['ml_score'] > threshold
        df_calc['triage']  = df_calc['is_high'].map({
            True:  'HIGH RISK ',
            False: 'Low / Likely False Positive '
        })

        high_count = int(df_calc['is_high'].sum())
        st.info(f"**{high_count:,} alerts marked HIGH RISK** out of {total_rows:,} (at current threshold)")

        # Data quality check
        if df_calc['rate'].max() > 1_000_000:
            st.warning("⚠️ Input data has unusually high packet rates. Model may need re-calibration for this dataset.")

        # === IMPACT BANNER ===
        has_label = 'label' in df_calc.columns
        if has_label:
            benign_count  = (df_calc['label'] == 0).sum()
            pre_fp_rate   = (benign_count / total_rows) * 100
            post_fp_count = (df_calc['is_high'] & (df_calc['label'] == 0)).sum()
            post_fp_rate  = (post_fp_count / total_rows) * 100
            reduction_pct = ((pre_fp_rate - post_fp_rate) / pre_fp_rate * 100) if pre_fp_rate > 0 else 0

            st.markdown(f"""
            <div class="impact-banner">
                <h3 style="margin:0; font-size:22px; opacity:0.9;"> REAL SOC IMPACT</h3>
                <div class="impact-big">{reduction_pct:.0f}% Reduction</div>
                <p class="impact-sub">
                    False positives reduced from <strong>{pre_fp_rate:.1f}%</strong> → <strong>{post_fp_rate:.1f}%</strong><br>
                    <span style="font-size:22px;">on {total_rows:,} alerts</span><br>
                    <span style="font-size:18px; opacity:0.85;">(Untuned SIEM baseline: all benign alerts were previously flagged high)</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            tp = (df_calc['is_high'] & (df_calc['label'] == 1)).sum()
            fp = (df_calc['is_high'] & (df_calc['label'] == 0)).sum()
            fn = (~df_calc['is_high'] & (df_calc['label'] == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div class="metric-box">
                    <h4>Precision</h4><h2 style="color:#34d399;">{precision:.1%}</h2>
                    <small>% of HIGH RISK alerts that were actually attacks</small>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-box">
                    <h4>Recall</h4><h2 style="color:#34d399;">{recall:.1%}</h2>
                    <small>% of real attacks caught by the model</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning(" No ground truth 'label' column found. Cannot compute false positive reduction.")

        st.caption("**ML Score** = probability of attack (0 = almost certainly benign, 1 = almost certainly malicious)")

        # ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        # UPGRADE 2 — PER-ALERT EXPLAINABILITY
        # ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        st.subheader(" Alert Explainability — Why did the model flag this?")
        st.caption("Select any HIGH RISK alert to see the model's reasoning in plain English.")

        high_df = df_calc[df_calc['is_high']].copy()

        if not high_df.empty:
            # Let user pick which alert to inspect
            high_df['display_label'] = (
                "Alert #" + high_df.index.astype(str) +
                "  |  score=" + high_df['ml_score'].round(3).astype(str) +
                ("  |  " + high_df['attack_cat'] if 'attack_cat' in high_df.columns else "")
            )
            selected_label = st.selectbox(
                "Inspect alert:",
                options=high_df['display_label'].tolist(),
                index=0
            )
            selected_idx   = high_df[high_df['display_label'] == selected_label].index[0]
            selected_alert = df_calc.loc[selected_idx]
            selected_score = float(selected_alert['ml_score'])

            # Build explanation using model internals + row values
            alert_as_series = selected_alert.copy()
            # Map one-hot columns back: X_calc row for this alert
            x_row = X_calc.loc[selected_idx] if selected_idx in X_calc.index else pd.Series(dtype=float)
            reasons = explain_alert(alert_as_series, expected_features, model)
            render_explanation(reasons, selected_score)

            # Show raw values for the selected alert
            with st.expander(" Raw feature values for this alert"):
                show_cols = [c for c in REQUIRED_FEATURES if c in selected_alert.index]
                st.dataframe(pd.DataFrame(selected_alert[show_cols]).T, use_container_width=True)
        else:
            st.info("No HIGH RISK alerts at current threshold — lower the slider to see explanations.")

        # ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        # UPGRADE 3 — ALERT CORRELATION VIEW
        # ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        st.subheader("🕸️ Alert Correlation — Is this part of a bigger attack?")
        st.caption("Groups alerts by protocol + service to surface coordinated attack patterns.")

        group_cols = [c for c in ['proto', 'service'] if c in df_calc.columns]

        if group_cols:
            corr = (
                df_calc.groupby(group_cols)
                .agg(
                    total_alerts  = ('ml_score', 'count'),
                    high_risk     = ('is_high',  'sum'),
                    avg_score     = ('ml_score', 'mean'),
                    max_score     = ('ml_score', 'max'),
                )
                .reset_index()
                .sort_values('high_risk', ascending=False)
            )
            corr['high_risk']     = corr['high_risk'].astype(int)
            corr['risk_density']  = (corr['high_risk'] / corr['total_alerts'] * 100).round(1)
            corr['threat_level']  = corr['risk_density'].apply(
                lambda x: '🔴 CAMPAIGN' if x > 70 else ('🟡 SUSPICIOUS' if x > 30 else '🟢 NORMAL')
            )

            # Highlight rows with campaign-level risk
            def highlight_threat(row):
                if '🔴' in str(row['threat_level']):
                    return ['background-color: #3b0000'] * len(row)
                elif '🟡' in str(row['threat_level']):
                    return ['background-color: #2d2400'] * len(row)
                return [''] * len(row)

            st.dataframe(
                corr.style.apply(highlight_threat, axis=1),
                use_container_width=True
            )

            # Bubble chart: total alerts vs high-risk count, sized by avg score
            if len(corr) > 1:
                corr['group_label'] = corr[group_cols].astype(str).agg(' / '.join, axis=1)
                fig_corr = px.scatter(
                    corr,
                    x='total_alerts',
                    y='high_risk',
                    size='avg_score',
                    color='risk_density',
                    text='group_label',
                    color_continuous_scale='Reds',
                    title="Attack Pattern Clustering — each bubble is a proto/service group",
                    labels={
                        'total_alerts': 'Total Alerts',
                        'high_risk':    'High Risk Alerts',
                        'risk_density': 'Risk Density (%)'
                    }
                )
                fig_corr.update_traces(textposition='top center', textfont_size=10)
                fig_corr.update_layout(coloraxis_colorbar_title="Risk %")
                st.plotly_chart(fig_corr, use_container_width=True)

            # Highlight any campaign-level groups with a clear warning
            campaigns = corr[corr['threat_level'].str.contains('CAMPAIGN')]
            if not campaigns.empty:
                for _, row in campaigns.iterrows():
                    label = " / ".join(str(row[c]) for c in group_cols)
                    st.error(
                        f"🚨 **Potential attack campaign detected:** `{label}` — "
                        f"{int(row['high_risk'])} HIGH RISK alerts out of {int(row['total_alerts'])} "
                        f"({row['risk_density']:.0f}% risk density, max score {row['max_score']:.2f})"
                    )
        else:
            st.info("No groupable columns (proto/service) found for correlation analysis.")

        # === UI TABLE ===
        st.subheader("AI Triage Results")
        df_display     = df_calc.head(50).copy()
        cols_to_show   = ['attack_cat', 'label', 'ml_score', 'triage', 'proto', 'service']
        available_cols = [c for c in cols_to_show if c in df_display.columns]
        st.dataframe(df_display[available_cols], use_container_width=True)

        # === RISK DISTRIBUTION PLOT ===
        st.subheader("Alert Risk Distribution")
        plot_df    = df_calc.sample(min(2000, len(df_calc)), random_state=42)
        hover_cols = [c for c in ['attack_cat', 'label', 'proto', 'service'] if c in plot_df.columns]

        fig = px.scatter(
            plot_df,
            x='dur', y='ml_score', color='triage',
            hover_data=hover_cols,
            title="Duration vs Attack Probability (sampled for speed)",
            color_discrete_map={'HIGH RISK ': 'red', 'Low / Likely False Positive ': 'green'}
        )
        fig.update_layout(xaxis_title="Duration (seconds)", yaxis_title="ML Attack Probability")
        st.plotly_chart(fig, use_container_width=True)

        # === REPORT BUTTON ===
        if st.button("Generate Investigation Report for Highest Risk Alert"):
            high_alerts = df_calc[df_calc['is_high']]
            if not high_alerts.empty:
                top_idx   = high_alerts['ml_score'].idxmax()
                top_alert = high_alerts.loc[top_idx]
            else:
                top_idx   = df_calc['ml_score'].idxmax()
                top_alert = df_calc.loc[top_idx]

            similar = df_calc[
                (df_calc['proto']   == top_alert['proto']) &
                (df_calc['service'] == top_alert['service'])
            ].head(3)
            similar_text = similar.to_string(index=False) if not similar.empty else "No similar alerts."

            with st.spinner("Talking to local LLM (Ollama)..."):
                llm_start   = time.time()
                report      = generate_report(top_alert, similar_text)
                llm_elapsed = time.time() - llm_start

                st.success(f"Report ready in {llm_elapsed:.1f}s")
                st.markdown("### Investigation Report")
                st.markdown(report)

                st.download_button(
                    label="Download Report.txt",
                    data=report,
                    file_name=f"siem_report_{top_idx}.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f" Processing error: {e}")
        st.caption("Check that your data matches the expected UNSW-NB15 format and the model was trained on compatible features.")

# ──────────────────────────────────────────────────────────────────
# AUTO-RERUN (live mode only)
# ──────────────────────────────────────────────────────────────────
if live_mode:
    time.sleep(3)
    st.rerun()

st.caption("Built with Streamlit • scikit-learn • Ollama • FastAPI Live Streaming")