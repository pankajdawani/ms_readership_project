import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json
import os

BASE_DIR = '/Users/pankajdawani/Desktop/ms_readership_project'

model = joblib.load(os.path.join(BASE_DIR, 'model', 'model.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'model', 'feature_names.pkl'))
with open(os.path.join(BASE_DIR, 'model', 'mappings.json'), 'r') as f:
    mappings = json.load(f)

df = pd.read_csv(os.path.join(BASE_DIR, 'ms_publications_v2.csv'))

def predict_reads(analyst_seniority, analyst_publish_frequency,
                  analyst_30d_avg_reads, subscriber_count,
                  channel, word_count, num_charts,
                  report_page_count, topic_sensitivity,
                  send_hour, is_market_open, days_to_earnings,
                  num_explicit_recipients, days_since_last_pub,
                  subject_line_length):

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

    if predicted_reads > 1000:
        signal = "🟢 High engagement expected — prioritise broad distribution"
    elif predicted_reads > 500:
        signal = "🟡 Moderate engagement — consider timing or channel optimisation"
    else:
        signal = "🔴 Low engagement expected — review seniority, channel or timing"

    result = f"""
## Prediction Results

| Metric | Value |
|--------|-------|
| **Predicted Reads** | {predicted_reads:,} |
| **Lower Bound** | {max(0, predicted_reads - mae):,} |
| **Upper Bound** | {predicted_reads + mae:,} |

{signal}
"""
    return result

def eda_chart(category):
    avg = df.groupby(category)["reads"].mean().reset_index().sort_values("reads", ascending=False)
    fig = px.bar(avg, x=category, y="reads", color="reads",
                 color_continuous_scale="Blues",
                 title=f"Average Reads by {category.replace('_', ' ').title()}")
    fig.update_layout(showlegend=False)
    return fig

def reads_by_hour():
    hour_avg = df.groupby("send_hour")["reads"].mean().reset_index()
    fig = px.line(hour_avg, x="send_hour", y="reads",
                  title="Avg Reads by Hour of Day", markers=True)
    fig.add_vline(x=9, line_dash="dash", line_color="red",
                  annotation_text="Peak 9am")
    return fig

def reads_distribution():
    fig = px.histogram(df, x="reads", nbins=40,
                       title="Reads Distribution",
                       color_discrete_sequence=["#1f77b4"])
    return fig

# Build app
with gr.Blocks(title="MS Publication Readership Predictor") as app:

    gr.Markdown("""
    # 📊 Morgan Stanley — Publication Readership Predictor
    Predict expected reads for analyst publications before they are sent.
    """)

    with gr.Tabs():

        # TAB 1 — EDA
        with gr.Tab("📈 EDA"):
            gr.Markdown("### Exploratory Data Analysis")
            gr.Markdown(f"Dataset: **{len(df):,} publications** across {df['analyst'].nunique()} analysts")

            with gr.Row():
                gr.Markdown(f"**Avg Reads:** {df['reads'].mean():.0f}")
                gr.Markdown(f"**Max Reads:** {df['reads'].max():,}")
                gr.Markdown(f"**Median Reads:** {df['reads'].median():.0f}")

            category = gr.Dropdown(
                choices=["analyst_seniority", "channel", "sector", "report_type", "region", "client_tier"],
                value="analyst_seniority",
                label="Select Category"
            )
            bar_chart = gr.Plot()
            category.change(fn=eda_chart, inputs=category, outputs=bar_chart)
            app.load(fn=eda_chart, inputs=category, outputs=bar_chart)

            with gr.Row():
                hour_chart = gr.Plot()
                dist_chart = gr.Plot()

            app.load(fn=reads_by_hour, outputs=hour_chart)
            app.load(fn=reads_distribution, outputs=dist_chart)

            gr.Markdown("""
            **Key Findings:**
            - Analyst seniority is the strongest driver — MDs generate **2.6x** more reads than Junior analysts
            - Email channel outperforms Web by **53%**
            - Peak send hour is **9am**
            - Sector, region and report type show no statistically significant difference
            """)

        # TAB 2 — SHAP
        with gr.Tab("🔍 SHAP Insights"):
            gr.Markdown("### SHAP explains WHY the model makes each prediction")
            with gr.Row():
                gr.Image(os.path.join(BASE_DIR, "shap_global_bar.png"), label="Global Feature Importance")
                gr.Image(os.path.join(BASE_DIR, "shap_beeswarm.png"), label="Feature Impact Direction")

            gr.Markdown("""
            **Key SHAP Findings:**
            - **analyst_seniority** — top feature globally
            - **analyst_30d_avg_reads** — recent momentum is strongest signal
            - **subscriber_count** — audience size drives volume
            - **is_market_open** — trading hours outperform after-hours
            - **topic_sensitivity** — market-moving notes get 9% more reads
            """)

        # TAB 3 — PREDICT
        with gr.Tab("🎯 Predict Reads"):
            gr.Markdown("### Fill in publication details to predict readership")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Analyst**")
                    analyst_seniority = gr.Dropdown(["Junior", "Mid", "Senior", "Managing Director"], value="Senior", label="Seniority")
                    analyst_publish_frequency = gr.Dropdown(["Daily", "Weekly", "Bi-weekly", "Monthly"], value="Weekly", label="Publish Frequency")
                    analyst_30d_avg_reads = gr.Slider(50, 1000, value=400, label="30-Day Avg Reads")
                    subscriber_count = gr.Slider(100, 5000, value=1000, label="Subscriber Count")

                with gr.Column():
                    gr.Markdown("**Publication**")
                    channel = gr.Dropdown(["Email", "Portal", "Web"], value="Email", label="Channel")
                    word_count = gr.Slider(300, 3000, value=1000, label="Word Count")
                    num_charts = gr.Slider(0, 10, value=2, label="Number of Charts")
                    report_page_count = gr.Slider(1, 40, value=8, label="Page Count")
                    topic_sensitivity = gr.Radio([0, 1], value=0, label="Market Sensitive Topic? (0=No, 1=Yes)")

                with gr.Column():
                    gr.Markdown("**Timing & Audience**")
                    send_hour = gr.Slider(6, 20, value=9, label="Send Hour (24h)")
                    is_market_open = gr.Radio([0, 1], value=1, label="Market Open? (0=No, 1=Yes)")
                    days_to_earnings = gr.Slider(0, 90, value=30, label="Days to Earnings")
                    num_explicit_recipients = gr.Slider(0, 2000, value=500, label="Email Recipients")
                    days_since_last_pub = gr.Slider(1, 30, value=7, label="Days Since Last Publication")
                    subject_line_length = gr.Slider(20, 100, value=60, label="Subject Line Length")

            predict_btn = gr.Button("🎯 Predict Reads", variant="primary")
            output = gr.Markdown()

            predict_btn.click(
                fn=predict_reads,
                inputs=[analyst_seniority, analyst_publish_frequency,
                        analyst_30d_avg_reads, subscriber_count,
                        channel, word_count, num_charts,
                        report_page_count, topic_sensitivity,
                        send_hour, is_market_open, days_to_earnings,
                        num_explicit_recipients, days_since_last_pub,
                        subject_line_length],
                outputs=output
            )

app.launch()
