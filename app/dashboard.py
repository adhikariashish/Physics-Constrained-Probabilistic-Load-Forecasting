from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# This is the forecast JSON produced by:
#   python scripts/forecast_next_24h.py --run runs/exp_transformer_physics_v0
#
# The dashboard reads that file and visualizes:
#   - next 24h mean forecast
#   - 80% / 95% prediction intervals
#   - peak hour summary
#   - risk level
#   - downloadable forecast table
# -----------------------------------------------------------------------------
FORECAST_JSON_PATH = Path("reports/latest_forecast/next_24h_forecast.json")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_forecast(path: Path) -> dict:
    """
    Load forecast JSON from disk.

    Streamlit cache is used so the file is not repeatedly re-read on every UI
    interaction unless the file changes.

    Expected top-level keys in the JSON:
        - run_name
        - source
        - target_col
        - feature_cols
        - context_length
        - horizon
        - last_context_timestamp
        - peak_summary
        - forecast

    Returns:
        Parsed JSON payload as a Python dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")
    return json.loads(path.read_text())


def build_forecast_df(payload: dict) -> pd.DataFrame:
    """
    Convert forecast list into a pandas DataFrame.

    Expected payload["forecast"] row fields:
        - timestamp
        - mean_mw
        - sigma_mw
        - lower_80_mw
        - upper_80_mw
        - lower_95_mw
        - upper_95_mw
        - risk_level
        - is_peak_mean
        - is_peak_upper95

    Returns:
        DataFrame with parsed timestamps.
    """
    df = pd.DataFrame(payload["forecast"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# -----------------------------------------------------------------------------
# Chart construction
# -----------------------------------------------------------------------------
def make_forecast_chart(df: pd.DataFrame) -> go.Figure:
    """
    Build Plotly figure for the next-24h probabilistic forecast.

    Visual layers:
        1. 95% prediction interval
        2. 80% prediction interval
        3. Mean forecast line

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # -------------------------------------------------------------------------
    # 95% interval
    # We add upper and lower boundaries separately, then fill between them.
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["upper_95_mw"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name="Upper 95%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["lower_95_mw"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="95% Interval",
            hovertemplate="95% Interval<br>%{x}<br>%{y:.0f} MW<extra></extra>",
        )
    )

    # -------------------------------------------------------------------------
    # 80% interval
    # Inner uncertainty band, narrower than the 95% band.
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["upper_80_mw"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name="Upper 80%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["lower_80_mw"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="80% Interval",
            hovertemplate="80% Interval<br>%{x}<br>%{y:.0f} MW<extra></extra>",
        )
    )

    # -------------------------------------------------------------------------
    # Mean forecast line
    # This is the expected load trajectory.
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["mean_mw"],
            mode="lines+markers",
            name="Mean Forecast",
            hovertemplate="%{x}<br>Mean: %{y:.0f} MW<extra></extra>",
        )
    )

    fig.update_layout(
        title="Next 24-Hour Load Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# -----------------------------------------------------------------------------
# Risk formatting helpers
# -----------------------------------------------------------------------------
def style_risk(value: str) -> str:
    """
    Convert raw risk label into a UI-friendly label.

    Input:
        'high', 'moderate', 'normal'

    Output:
        decorated string with visual indicator
    """
    value = str(value).lower()
    if value == "high":
        return "🔴 HIGH"
    if value == "moderate":
        return "🟠 MODERATE"
    return "🟢 NORMAL"


def render_risk_banner(peak_risk: str) -> None:
    """
    Show top-level operational alert banner based on the detected peak-risk level.
    """
    risk_label = str(peak_risk).lower()
    if risk_label == "high":
        st.error("⚠️ High peak-load risk detected in the next 24 hours.")
    elif risk_label == "moderate":
        st.warning("⚠️ Moderate peak-load risk detected in the next 24 hours.")
    else:
        st.success("Peak-load risk is within normal range.")


# -----------------------------------------------------------------------------
# Main dashboard
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Streamlit entrypoint.

    Dashboard flow:
        1. Read forecast JSON
        2. Extract peak summary + forecast rows
        3. Show KPI cards
        4. Show risk alert banner
        5. Plot probabilistic forecast chart
        6. Show metadata panel
        7. Show forecast table + CSV download
    """
    st.set_page_config(
        page_title="Probabilistic Load Forecast Dashboard",
        layout="wide",
    )

    st.title("⚡ Physics-Constrained Probabilistic Load Forecast")
    st.caption("24-hour ahead system load forecast with uncertainty bands and peak-risk alerting")

    # -------------------------------------------------------------------------
    # Load forecast file
    # -------------------------------------------------------------------------
    try:
        payload = load_forecast(FORECAST_JSON_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Run the inference script first:\n\n"
            "`python scripts/forecast_next_24h.py --run runs/exp_transformer_physics_v0`"
        )
        return

    df = build_forecast_df(payload)

    # -------------------------------------------------------------------------
    # Peak summary
    # If present in JSON, use it directly.
    # If missing, fall back safely to a value computed from the forecast rows.
    # -------------------------------------------------------------------------
    peak = payload.get("peak_summary", {})
    peak_mean = peak.get("peak_mean_mw", float(df["mean_mw"].max()))
    peak_ts = peak.get("peak_timestamp", str(df.loc[df["mean_mw"].idxmax(), "timestamp"]))
    peak_risk = peak.get("peak_risk_level", "normal")
    peak_threshold = peak.get("peak_threshold_mw", 0.0)
    peak_upper95 = peak.get("peak_upper_95_mw", 0.0)

    # -------------------------------------------------------------------------
    # Aggregate summary metrics for top cards
    # -------------------------------------------------------------------------
    avg_load = float(df["mean_mw"].mean())
    max_sigma = float(df["sigma_mw"].max())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Forecast Load", f"{peak_mean:,.0f} MW")
    c2.metric("Peak Hour", str(peak_ts))
    c3.metric("Average Forecast Load", f"{avg_load:,.0f} MW")
    c4.metric("Max Uncertainty (σ)", f"{max_sigma:,.0f} MW")

    # -------------------------------------------------------------------------
    # Peak-risk banner
    # This is the operator-style alerting layer.
    # -------------------------------------------------------------------------
    render_risk_banner(peak_risk)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Main layout:
    # left  -> forecast chart
    # right -> metadata + peak summary
    # -------------------------------------------------------------------------
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Probabilistic Forecast")
        fig = make_forecast_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model Metadata")
        st.write(f"**Run Name:** {payload.get('run_name', 'N/A')}")
        st.write(f"**Source:** {payload.get('source', 'N/A')}")
        st.write(f"**Target Column:** {payload.get('target_col', 'N/A')}")
        st.write(f"**Context Length:** {payload.get('context_length', 'N/A')} hours")
        st.write(f"**Forecast Horizon:** {payload.get('horizon', 'N/A')} hours")
        st.write(f"**Last Context Timestamp:** {payload.get('last_context_timestamp', 'N/A')}")

        st.subheader("Peak Summary")
        st.write(f"**Peak Threshold:** {peak_threshold:,.0f} MW")
        st.write(f"**Peak Mean Forecast:** {peak_mean:,.0f} MW")
        st.write(f"**Peak Upper 95%:** {peak_upper95:,.0f} MW")
        st.write(f"**Peak Risk Level:** {style_risk(peak_risk)}")

        with st.expander("Feature Columns"):
            st.write(payload.get("feature_cols", []))

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Forecast table
    # This table is useful for operations-style review and CSV export.
    # -------------------------------------------------------------------------
    st.subheader("Forecast Table")

    display_df = df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    if "risk_level" in display_df.columns:
        display_df["risk_level"] = display_df["risk_level"].map(style_risk)

    display_df = display_df.rename(
        columns={
            "timestamp": "Timestamp",
            "mean_mw": "Mean (MW)",
            "sigma_mw": "Sigma (MW)",
            "lower_80_mw": "Lower 80% (MW)",
            "upper_80_mw": "Upper 80% (MW)",
            "lower_95_mw": "Lower 95% (MW)",
            "upper_95_mw": "Upper 95% (MW)",
            "risk_level": "Risk Level",
            "is_peak_mean": "Peak Mean Flag",
            "is_peak_upper95": "Peak Upper95 Flag",
        }
    )

    st.dataframe(display_df, width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # Download button
    # Allow dashboard users to export the displayed forecast as CSV.
    # -------------------------------------------------------------------------
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecast CSV",
        data=csv_bytes,
        file_name="next_24h_forecast.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()