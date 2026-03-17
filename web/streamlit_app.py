from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from malware_detector.inference.predictor import load_predictor


MODEL_PATH = Path("artifacts/model.joblib")
METADATA_PATH = Path("artifacts/metadata.json")
TARGET_COL = "Label"


st.set_page_config(
    page_title="AutoPlay Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.10), transparent 28%),
                    radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 25%),
                    linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
                color: #e5e7eb;
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }

            .hero {
                position: relative;
                overflow: hidden;
                padding: 34px 36px;
                border-radius: 24px;
                background:
                    linear-gradient(135deg, rgba(15,23,42,0.92), rgba(17,24,39,0.84)),
                    linear-gradient(135deg, #0f172a 0%, #111827 55%, #0f766e 100%);
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 20px 50px rgba(0,0,0,0.35);
                margin-bottom: 1rem;
            }

            .hero:before {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    radial-gradient(circle at 20% 20%, rgba(45,212,191,0.18), transparent 24%),
                    radial-gradient(circle at 82% 22%, rgba(59,130,246,0.18), transparent 20%);
                pointer-events: none;
            }

            .hero h1 {
                margin: 0;
                font-size: 2.25rem;
                line-height: 1.1;
                font-weight: 800;
                color: #f8fafc;
                letter-spacing: -0.02em;
            }

            .hero p {
                margin: 0.8rem 0 0 0;
                max-width: 780px;
                color: #cbd5e1;
                font-size: 1rem;
                line-height: 1.55;
            }

            .hero-strip {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 1rem;
            }

            .hero-pill {
                padding: 0.45rem 0.8rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                color: #e2e8f0;
                font-size: 0.86rem;
                font-weight: 600;
            }

            .glass-card {
                background: rgba(15, 23, 42, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
            }

            .metric-card {
                background: linear-gradient(180deg, rgba(15,23,42,0.86), rgba(30,41,59,0.74));
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
                min-height: 105px;
            }

            .metric-label {
                color: #94a3b8;
                font-size: 0.86rem;
                margin-bottom: 0.35rem;
                font-weight: 600;
            }

            .metric-value {
                color: #f8fafc;
                font-size: 1.75rem;
                font-weight: 800;
                letter-spacing: -0.03em;
            }

            .metric-sub {
                color: #cbd5e1;
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }

            .section-title {
                color: #f8fafc;
                font-size: 1.08rem;
                font-weight: 700;
                margin-bottom: 0.55rem;
            }

            .muted {
                color: #94a3b8;
                font-size: 0.92rem;
            }

            .risk-high, .risk-medium, .risk-low {
                display: inline-block;
                padding: 0.3rem 0.65rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 700;
            }

            .risk-high {
                background: rgba(239, 68, 68, 0.18);
                color: #fecaca;
                border: 1px solid rgba(239, 68, 68, 0.28);
            }

            .risk-medium {
                background: rgba(245, 158, 11, 0.16);
                color: #fde68a;
                border: 1px solid rgba(245, 158, 11, 0.26);
            }

            .risk-low {
                background: rgba(16, 185, 129, 0.18);
                color: #bbf7d0;
                border: 1px solid rgba(16, 185, 129, 0.28);
            }

            .threat-panel {
                background:
                    linear-gradient(180deg, rgba(127,29,29,0.20), rgba(15,23,42,0.90));
                border: 1px solid rgba(248, 113, 113, 0.20);
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
            }

            .threat-title {
                color: #fecaca;
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            }

            .threat-item {
                padding: 10px 0;
                border-top: 1px solid rgba(255,255,255,0.08);
            }

            .threat-item:first-child {
                border-top: none;
                padding-top: 0;
            }

            .threat-score {
                color: #fca5a5;
                font-weight: 700;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 16px;
                overflow: hidden;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(148,163,184,0.14);
                border-radius: 12px;
                color: #cbd5e1;
                padding: 10px 14px;
            }

            .stTabs [aria-selected="true"] {
                background: rgba(20, 184, 166, 0.12) !important;
                color: #f8fafc !important;
                border-color: rgba(45,212,191,0.28) !important;
            }

            section[data-testid="stSidebar"] {
                background: rgba(2, 6, 23, 0.92);
                border-right: 1px solid rgba(148,163,184,0.12);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_predictor():
    return load_predictor(MODEL_PATH)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df[TARGET_COL]
    y_proba = df["malware_probability"]
    y_pred = df["predicted_label"]

    return {
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
    }


def render_metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_bucket(prob: float) -> str:
    if prob >= 0.80:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def risk_badge(level: str) -> str:
    mapping = {
        "High": '<span class="risk-high">High Risk</span>',
        "Medium": '<span class="risk-medium">Medium Risk</span>',
        "Low": '<span class="risk-low">Low Risk</span>',
    }
    return mapping[level]


def enrich_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prediction_label_text"] = out["predicted_label"].map({0: "Goodware", 1: "Malware"})
    out["risk_tier"] = out["malware_probability"].apply(risk_bucket)
    out["risk_badge"] = out["risk_tier"].apply(risk_badge)
    return out


def plot_prediction_distribution(df: pd.DataFrame):
    counts = df["predicted_label"].value_counts().sort_index()
    labels = ["Goodware", "Malware"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.bar(labels, values)
    ax.set_title("Prediction Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_risk_tiers(df: pd.DataFrame):
    order = ["Low", "Medium", "High"]
    counts = df["risk_tier"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.bar(order, counts.values)
    ax.set_title("Threat Tier Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_probability_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.hist(df["malware_probability"], bins=20)
    ax.set_title("Malware Probability Histogram")
    ax.set_xlabel("Predicted Malware Probability")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Pred Goodware", "Pred Malware"])
    ax.set_yticks([0, 1], labels=["True Goodware", "True Malware"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def render_header(metadata: dict) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>AutoPlay Shield</h1>
            <p>
                A polished malware triage workspace for static PE-feature analysis.
                Upload datasets, inspect model outcomes, surface high-risk samples,
                and validate performance through an analyst-grade dashboard.
            </p>
            <div class="hero-strip">
                <div class="hero-pill">Random Forest Pipeline</div>
                <div class="hero-pill">Threat Scoring</div>
                <div class="hero-pill">Analyst Review</div>
                <div class="hero-pill">FastAPI + Streamlit</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Model", metadata.get("model_name", "N/A"), "Production artifact")
    with c2:
        render_metric_card("Threshold", str(metadata.get("threshold", "N/A")), "Adjustable in sidebar")
    with c3:
        render_metric_card("Train Rows", str(metadata.get("train_rows", "N/A")), "Training footprint")
    with c4:
        render_metric_card("Test Rows", str(metadata.get("test_rows", "N/A")), "Hold-out benchmark")


def render_sidebar(metadata: dict) -> tuple[float, bool, int]:
    st.sidebar.title("Control Center")
    st.sidebar.caption("Tune analyst view and triage output.")

    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(metadata.get("threshold", 0.5)),
        step=0.01,
    )
    show_all_columns = st.sidebar.checkbox("Show all feature columns", value=False)
    top_n_alerts = st.sidebar.slider("Threat panel sample size", 5, 25, 10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Input expectations")
    st.sidebar.caption(
        "Upload a CSV of PE features. Include `Label` to unlock evaluation metrics."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Suggested use")
    st.sidebar.caption(
        "Use this workspace for demos, analyst review, test-set validation, and rapid model QA."
    )

    return threshold, show_all_columns, top_n_alerts


def render_threat_panel(df: pd.DataFrame, top_n: int) -> None:
    ranked = df.sort_values("malware_probability", ascending=False).head(top_n)

    st.markdown('<div class="threat-title">Threat Analysis Panel</div>', unsafe_allow_html=True)

    if ranked.empty:
        st.caption("No predictions available yet.")
        return

    for idx, row in ranked.iterrows():
        st.markdown(
            f"""
            <div class="threat-item">
                <div><strong>Row {idx}</strong> · {risk_badge(row["risk_tier"])}</div>
                <div class="muted">Prediction: {row["prediction_label_text"]}</div>
                <div class="threat-score">Malware score: {row["malware_probability"]:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_css()
    metadata = load_metadata()
    render_header(metadata)

    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Train the model first.")
        st.code("python -m malware_detector.modeling.final_train_and_test")
        return

    threshold, show_all_columns, top_n_alerts = render_sidebar(metadata)

    intro_left, intro_right = st.columns([1.7, 1])
    with intro_left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Workspace Overview</div>
                <div class="muted">
                    This interface is designed like a product dashboard: upload a dataset,
                    inspect predictions, review threat tiers, and validate model quality.
                    It supports both unlabeled inference datasets and labeled test uploads.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_right:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Workflow</div>
                <div class="muted">
                    1. Upload CSV<br>
                    2. Score threats<br>
                    3. Review top alerts<br>
                    4. Validate metrics if labels exist<br>
                    5. Export enriched results
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader("Upload PE feature dataset", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to activate the dashboard.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if df.empty:
        st.warning("The uploaded CSV is empty.")
        return

    predictor = get_predictor()
    predictor.threshold = threshold

    has_labels = TARGET_COL in df.columns
    X = df.drop(columns=[TARGET_COL]) if has_labels else df.copy()

    try:
        result_df = predictor.predict_dataframe(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    if has_labels:
        result_df[TARGET_COL] = df[TARGET_COL].values

    result_df = enrich_predictions(result_df)

    total_rows = len(result_df)
    malware_count = int((result_df["predicted_label"] == 1).sum())
    goodware_count = int((result_df["predicted_label"] == 0).sum())
    avg_risk = float(result_df["malware_probability"].mean())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Rows Processed", str(total_rows), "Current upload")
    with k2:
        render_metric_card("Predicted Malware", str(malware_count), "Alert-classified rows")
    with k3:
        render_metric_card("Predicted Goodware", str(goodware_count), "Benign-classified rows")
    with k4:
        render_metric_card("Average Malware Score", f"{avg_risk:.3f}", "Mean risk signal")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Executive Overview", "Threat Analysis", "Predictions", "Validation", "Model Center"]
    )

    with tab1:
        top_left, top_right = st.columns([1.25, 1])
        with top_left:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Uploaded Dataset Preview</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(20), use_container_width=True)

        with top_right:
            st.markdown(
                '<div class="threat-panel">',
                unsafe_allow_html=True,
            )
            render_threat_panel(result_df, top_n_alerts)
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_prediction_distribution(result_df), clear_figure=True)
        with c2:
            st.pyplot(plot_risk_tiers(result_df), clear_figure=True)

    with tab2:
        a, b = st.columns([1.1, 1])
        with a:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Probability Distribution</div>
                    <div class="muted">Inspect how confidently the model separates benign and malicious samples.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.pyplot(plot_probability_histogram(result_df), clear_figure=True)

        with b:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Threat Tier Counts</div>
                    <div class="muted">High-risk rows are ideal for manual triage or immediate downstream action.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            counts = result_df["risk_tier"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
            for tier, value in counts.items():
                badge = risk_badge(tier)
                st.markdown(
                    f"""
                    <div style="padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.08);">
                        <div>{badge}</div>
                        <div class="metric-value" style="font-size:1.35rem; margin-top:0.3rem;">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Top suspicious rows")
        threat_view = result_df.sort_values("malware_probability", ascending=False).head(top_n_alerts).copy()
        display_cols = ["prediction_label_text", "risk_tier", "malware_probability"]
        if has_labels:
            display_cols.append(TARGET_COL)
        st.dataframe(threat_view[display_cols + [c for c in threat_view.columns if c not in display_cols][:8]], use_container_width=True)

    with tab3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Prediction Explorer</div>
                <div class="muted">Sort, inspect, and export enriched records for reporting or downstream action.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        base_cols = ["prediction_label_text", "predicted_label", "malware_probability", "risk_tier"]
        if has_labels:
            base_cols.append(TARGET_COL)

        if show_all_columns:
            prediction_view = result_df
        else:
            remaining = [c for c in result_df.columns if c not in base_cols + ["risk_badge"]]
            prediction_view = result_df[base_cols + remaining[:10]]

        st.dataframe(
            prediction_view.sort_values("malware_probability", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            label="Download enriched predictions CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    with tab4:
        if not has_labels:
            st.info("Upload a CSV with a `Label` column to unlock validation metrics.")
        else:
            metrics = compute_metrics(result_df)

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                render_metric_card("AUC", f"{metrics['AUC']:.4f}")
            with m2:
                render_metric_card("Accuracy", f"{metrics['Accuracy']:.4f}")
            with m3:
                render_metric_card("Precision", f"{metrics['Precision']:.4f}")
            with m4:
                render_metric_card("Recall", f"{metrics['Recall']:.4f}")
            with m5:
                render_metric_card("F1", f"{metrics['F1']:.4f}")

            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.pyplot(plot_confusion_matrix(metrics["Confusion Matrix"]), clear_figure=True)
            with c2:
                st.markdown(
                    """
                    <div class="glass-card">
                        <div class="section-title">Validation Interpretation</div>
                        <div class="muted">
                            Use these metrics to compare uploaded labeled datasets against your hold-out benchmark.
                            AUC reflects ranking quality, recall emphasizes malware capture, and precision
                            indicates how reliable positive alerts are.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab5:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Model Metadata</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if metadata:
                st.json(metadata)
            else:
                st.warning("No metadata.json found.")
        with c2:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Product Notes</div>
                    <div class="muted">
                        AutoPlay Shield is designed as a startup-style analyst console:
                        a premium demo surface for stakeholders, security review, and model QA.
                        Pair it with the FastAPI service for production APIs and CI/CD-backed deployment.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="glass-card" style="margin-top: 12px;">
                    <div class="section-title">Suggested Next Enhancements</div>
                    <div class="muted">
                        • add user authentication<br>
                        • log prediction requests<br>
                        • export analyst notes<br>
                        • connect to deployed API<br>
                        • support drift monitoring
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()