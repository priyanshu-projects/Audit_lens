"""
src/dashboard/components/shap_chart.py
========================================
SHAP waterfall chart component for Streamlit.
Renders token-level attributions as a horizontal bar chart.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.classification.shap_explainer import ShapResult


def render_shap_chart(shap_result: ShapResult) -> None:
    """
    Render a SHAP waterfall/bar chart for token attributions.

    Risk-raising tokens → red bars (positive SHAP values)
    Risk-lowering tokens → green bars (negative SHAP values)
    """
    if not shap_result.token_attributions:
        st.warning("No SHAP token attributions available.")
        return

    # Filter to meaningful tokens (non-zero, non-punctuation)
    meaningful = [
        t for t in shap_result.token_attributions
        if len(t.token.strip(".,;:\"'()[]")) > 1
    ]

    # Sort by absolute value, take top 15
    top_tokens = sorted(meaningful, key=lambda t: abs(t.value), reverse=True)[:15]
    top_tokens_sorted = sorted(top_tokens, key=lambda t: t.value)

    tokens = [t.token for t in top_tokens_sorted]
    values = [t.value for t in top_tokens_sorted]
    bar_colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=tokens,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))

    fig.update_layout(
        title=dict(
            text="SHAP Token Attributions",
            font=dict(color="#f1f5f9", size=14),
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        xaxis=dict(
            title="SHAP Value (+ = raises risk | − = lowers risk)",
            zeroline=True,
            zerolinecolor="#475569",
            tickfont=dict(color="#e2e8f0"),
        ),
        yaxis=dict(tickfont=dict(color="#e2e8f0")),
        height=400,
        margin=dict(l=100, r=80, t=60, b=60),
    )

    # Add zero reference line
    fig.add_vline(x=0, line_width=1, line_color="#475569")

    st.plotly_chart(fig, use_container_width=True)

    # Narrative
    st.markdown(f"**Explanation:** {shap_result.narrative}")

    # Token table
    with st.expander("📊 All token attributions"):
        token_data = [
            {"Token": t.token, "SHAP Value": round(t.value, 4),
             "Direction": "🔴 Risk" if t.value > 0 else "🟢 Safe"}
            for t in sorted(shap_result.token_attributions, key=lambda x: x.value, reverse=True)
            if abs(t.value) > 0.001
        ]
        if token_data:
            import pandas as pd
            st.dataframe(pd.DataFrame(token_data), use_container_width=True, hide_index=True)
