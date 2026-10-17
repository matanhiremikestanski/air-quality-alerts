"""Streamlit dashboard. Run: streamlit run app.py"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, plotly.graph_objects as go, streamlit as st
from plotly.subplots import make_subplots
from pipeline import AlertService, INPUTS, SENSORS

st.set_page_config(page_title="Air-quality alerts", layout="wide")
@st.cache_resource
def get_service(): return AlertService()
svc = get_service()
COL = {"NONE": "#2e7d32", "WATCH": "#f9a825", "ALERT": "#ef6c00", "CRITICAL": "#c62828", "SENSOR_FAULT": "#6a1b9a", "NO_DATA": "#757575"}

st.title("Urban air-quality anomaly alerts")
st.warning("Decision support only. These alerts come from low-cost sensors and an unsupervised model. "
           "They do NOT replace calibrated reference monitoring. Confirm every ALERT/CRITICAL with a calibrated instrument or site visit.")
mode = st.sidebar.radio("Mode", ["Replay an hour from history", "Enter a new hourly record"])
hist = svc.history
if mode.startswith("Replay"):
    ts = st.sidebar.select_slider("Hour", options=list(hist.index[500:]), value=hist.index[-1000])
    vals = hist.loc[ts, INPUTS].to_dict()
else:
    ts = st.sidebar.text_input("Timestamp", str(hist.index[-1] + pd.Timedelta(hours=1)))
    last = hist.iloc[-1]
    vals = {c: st.sidebar.number_input(c, value=float(last[c]) if pd.notna(last[c]) else 0.0) for c in INPUTS}
if mode.startswith("Replay") or st.sidebar.button("Score record", type="primary") or True:
    res = svc.score_record(ts, vals)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"### Severity: <span style='color:{COL[res['severity']]}'>{res['severity']}</span>", unsafe_allow_html=True)
    c2.metric("Percentile IF / LOF", f"{res['percentile_isolation_forest']} / {res['percentile_lof']}")
    c3.metric("Sensors moved together", res["sensors_moved_together"])
    st.write(f"**Likely cause (hypothesis, not a diagnosis):** {res['cause']}")
    st.write("**Why:**"); [st.write("- " + t) for t in res["explanation"]["text"]]
    if res["gap_fraction_24h"] > 0.25: st.info(f"{res['gap_fraction_24h']:.0%} of the last 24 h had missing sensor data; treat with caution.")
    w = pd.DataFrame(res["window"]).set_index("ts")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.6, .4], vertical_spacing=.06,
                        subplot_titles=("Sensors, indexed to window median = 100", "Anomaly percentile (alert thresholds 0.99 / 0.995 / 0.999)"))
    for c in SENSORS:
        s = pd.to_numeric(w[c]); fig.add_trace(go.Scatter(x=w.index, y=100*s/s.median(), name=c, mode="lines"), 1, 1)
    fig.add_trace(go.Scatter(x=w.index, y=w.p_if, name="Isolation Forest"), 2, 1)
    fig.add_trace(go.Scatter(x=w.index, y=w.p_lof, name="LOF"), 2, 1)
    for y in (.99, .995, .999): fig.add_hline(y=y, line_dash="dot", line_color="grey", row=2, col=1)
    fig.add_vline(x=res["timestamp"], line_color="black")
    fig.update_yaxes(range=[.9, 1.001], row=2, col=1); fig.update_layout(height=620, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("Window = previous 72 h up to the scored hour. Missing hours appear as gaps.")
