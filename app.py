
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import requests

# Page config
st.set_page_config(
    page_title="MS Publication Readership Predictor",
    page_icon="📊",
    layout="wide"
)

# Load data and model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "ms_publications_v2.csv"))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "model", "feature_names.pkl"))
    with open(os.path.join(BASE_DIR, "model", "mappings.json"), "r") as f:
        mappings = json.load(f)
    return model, feature_names, mappings

df = load_data()
model, feature_names, mappings = load_model()

# Title
st.title("📊 Morgan Stanley — Publication Readership Predictor")
st.markdown("Predict expected reads for analyst publications before they are sent.")
st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 EDA", "🔍 SHAP Insights", "🎯 Predict Reads"])

# ── TAB 1: EDA ──────────────────────────────────────────────
with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown(f"Dataset: **{len(df):,} publications** across {df['analyst'].nunique()} analysts")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Reads", f"{df['reads'].mean():.0f}")
    col2.metric("Max Reads", f"{df['reads'].max():,}")
    col3.metric("Median Reads", f"{df['reads'].median():.0f}")

    st.subheader("Average Reads by Category")

    cat = st.selectbox("Select category", [
        "analyst_seniority", "channel", "sector",
        "report_type", "region", "client_tier"
    ])

    avg = df.groupby(cat)["reads"].mean().reset_index().sort_values("reads", ascending=False)
    fig = px.bar(avg, x=cat, y="reads", color="reads",
                 color_continuous_scale="Blues",
                 title=f"Average Reads by {cat.replace('_', ' ').title()}")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reads Distribution")
        fig2 = px.histogram(df, x="reads", nbins=40,
                            title="Distribution of Reads",
                            color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Reads by Send Hour")
        hour_avg = df.groupby("send_hour")["reads"].mean().reset_index()
        fig3 = px.line(hour_avg, x="send_hour", y="reads",
                       title="Avg Reads by Hour of Day",
                       markers=True)
        fig3.add_vline(x=9, line_dash="dash", line_color="red",
                       annotation_text="Peak 9am")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Key EDA Findings")
    st.markdown("""
    - **Analyst seniority** is the strongest driver — MDs generate **2.6x** more reads than Junior analysts
    - **Email channel** outperforms Web by **53%** — push vs pull dynamic
    - **Peak send hour is 9am** — reads drop significantly after 2pm
    - **Sector, region and report type** show no statistically significant difference (ANOVA p>0.05)
    """)

# ── TAB 2: SHAP ─────────────────────────────────────────────
with tab2:
    st.header("SHAP Feature Importance")
    st.markdown("SHAP explains **why** the model makes each prediction — not just what it predicts.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global Feature Importance")
        shap_img = os.path.join(BASE_DIR, "shap_global_bar.png")
        if os.path.exists(shap_img):
            st.image(shap_img, use_container_width=True)
        else:
            st.info("Run SHAP analysis in Jupyter to generate this chart")

    with col2:
        st.subheader("Feature Impact Direction")
        beeswarm_img = os.path.join(BASE_DIR, "shap_beeswarm.png")
        if os.path.exists(beeswarm_img):
            st.image(beeswarm_img, use_container_width=True)
        else:
            st.info("Run SHAP analysis in Jupyter to generate this chart")

    st.subheader("Key SHAP Findings")
    st.markdown("""
    - **analyst_seniority** — top feature globally, MD publications receive significantly more reads
    - **analyst_30d_avg_reads** — recent momentum is the strongest engagement signal
    - **subscriber_count** — audience size directly drives readership volume
    - **is_market_open** — publications sent during trading hours outperform after-hours sends
    - **topic_sensitivity** — market-moving notes generate 9% more reads on average
    """)

# ── TAB 3: PREDICT ──────────────────────────────────────────
with tab3:
    st.header("Predict Publication Readership")
    st.markdown("Fill in the publication details below to get a predicted read count.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Analyst")
        analyst_seniority = st.selectbox("Seniority", ["Junior", "Mid", "Senior", "Managing Director"])
        analyst_publish_frequency = st.selectbox("Publish Frequency", ["Daily", "Weekly", "Bi-weekly", "Monthly"])
        analyst_30d_avg_reads = st.slider("30-Day Avg Reads", 50, 1000, 400)
        subscriber_count = st.slider("Subscriber Count", 100, 5000, 1000)

    with col2:
        st.subheader("Publication")
        channel = st.selectbox("Channel", ["Email", "Portal", "Web"])
        word_count = st.slider("Word Count", 300, 3000, 1000)
        num_charts = st.slider("Number of Charts", 0, 10, 2)
        report_page_count = st.slider("Page Count", 1, 40, 8)
        topic_sensitivity = st.selectbox("Market Sensitive Topic?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        st.subheader("Timing & Audience")
        send_hour = st.slider("Send Hour (24h)", 6, 20, 9)
        is_market_open = st.selectbox("Market Open?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        days_to_earnings = st.slider("Days to Earnings", 0, 90, 30)
        num_explicit_recipients = st.slider("Email Recipients", 0, 2000, 500)
        days_since_last_pub = st.slider("Days Since Last Publication", 1, 30, 7)
        subject_line_length = st.slider("Subject Line Length", 20, 100, 60)

    st.divider()

    if st.button("🎯 Predict Reads", type="primary", use_container_width=True):

        # Encode features
        seniority_encoded = mappings["seniority_map"][analyst_seniority]
        freq_encoded = mappings["freq_map"][analyst_publish_frequency]
        channel_encoded = mappings["channel_map"][channel]

        features = {
            "analyst_seniority": seniority_encoded,
            "word_count": word_count,
            "num_charts": num_charts,
            "report_page_count": report_page_count,
            "topic_sensitivity": topic_sensitivity,
            "subject_line_length": subject_line_length,
            "send_hour": send_hour,
            "is_market_open": is_market_open,
            "days_to_earnings": days_to_earnings,
            "subscriber_count": subscriber_count,
            "num_explicit_recipients": num_explicit_recipients,
            "analyst_30d_avg_reads": analyst_30d_avg_reads,
            "analyst_publish_frequency": freq_encoded,
            "days_since_last_pub": days_since_last_pub,
            "channel_Portal": channel_encoded["channel_Portal"],
            "channel_Web": channel_encoded["channel_Web"]
        }

        df_input = pd.DataFrame([features])[feature_names]
        pred_log = model.predict(df_input)[0]
        predicted_reads = int(np.expm1(pred_log))
        mae = 89

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Reads", f"{predicted_reads:,}")
        col2.metric("Lower Bound", f"{max(0, predicted_reads - mae):,}")
        col3.metric("Upper Bound", f"{predicted_reads + mae:,}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_reads,
            title={"text": "Predicted Reads"},
            gauge={
                "axis": {"range": [0, 3000]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 500], "color": "#ffcccc"},
                    {"range": [500, 1000], "color": "#fff0cc"},
                    {"range": [1000, 3000], "color": "#ccffcc"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": predicted_reads
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        if predicted_reads > 1000:
            st.success("🟢 High engagement expected — prioritise broad distribution")
        elif predicted_reads > 500:
            st.warning("🟡 Moderate engagement — consider timing or channel optimisation")
        else:
            st.error("🔴 Low engagement expected — review seniority, channel or timing")
