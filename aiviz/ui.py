"""Altair chart definitions and page CSS: given prepared data, render a styled chart.

Data assembly and Streamlit wiring stay in main.py; this module only knows
how a DataFrame should look once it's handed a chart to build, and how the
page itself should be styled.
"""

from typing import no_type_check

import altair as alt
import pandas as pd

CHART_WIDTH = 600
CHART_HEIGHT = 600
BAR_CHART_HEIGHT = CHART_HEIGHT - 100

# CSS breakpoints
STACK_BREAKPOINT_PX = 1400
STACK_BREAKPOINT_CSS = f"""
<style>
@media (max-width: {STACK_BREAKPOINT_PX}px) {{
    .st-key-main_layout [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap;
    }}
    .st-key-main_layout [data-testid="stColumn"] {{
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
    }}
}}
</style>
"""

ATTENTION_SELECT_CSS = f"""
<style>
.st-key-attention_layer {{
    margin-left: auto;
    margin-right: {CHART_WIDTH // 10}px;
}}
</style>
"""

TAB_CENTER_CSS = """
<style>
.st-key-chart_box [role="tablist"] {
    justify-content: center;
}
</style>
"""


def softmax_chart(data: pd.DataFrame) -> alt.Chart:
    """Vertical bars of the most likely next tokens, most likely on the left."""
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                shorthand="token:N",
                title=None,
                sort=alt.SortField("rank"),
                axis=alt.Axis(labelLimit=200, labelAngle=-45, labelOverlap=False),
            ),
            y=alt.Y(
                shorthand="probability:Q",
                title="Probability",
                scale=alt.Scale(zero=True, nice=True),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("token:N", title="Token"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(width="container", height=BAR_CHART_HEIGHT)
    )


def attention_chart(data: pd.DataFrame, top_n: int) -> alt.Chart:
    """Vertical bars of how much attention the current token pays to each context token."""
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                shorthand="token:N",
                title=f"Top {top_n} attended tokens (position)",
                sort=alt.SortField(field="rank"),
                axis=alt.Axis(labelLimit=200, labelAngle=-45, labelOverlap=False),
            ),
            y=alt.Y(
                shorthand="attention:Q",
                title="Attention",
                scale=alt.Scale(zero=True, nice=True),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip(shorthand="token:N", title="Token (position)"),
                alt.Tooltip(shorthand="attention:Q", title="Attention", format=".1%"),
            ],
        )
        .properties(width="container", height=BAR_CHART_HEIGHT)
    )


@no_type_check
def embedding_chart(data: pd.DataFrame) -> alt.LayerChart:
    """Scatter of tokens by 2D-projected final hidden state; the selected token is labeled."""
    base: alt.Chart = alt.Chart(data).encode(
        x=alt.X("x:Q").title(None).scale(zero=False).axis(labels=False, ticks=False),
        y=alt.Y("y:Q").title(None).scale(zero=False).axis(labels=False, ticks=False),
    )
    is_selected = alt.datum.status == "Selected"
    base_circled: alt.Chart = base.mark_circle()
    points: alt.Chart = base_circled.encode(
        color=alt.Color("status:N").scale(domain=["Selected", "Other"]).legend(None),
        size=alt.when(is_selected).then(alt.value(400)).otherwise(alt.value(80)),
        opacity=alt.when(is_selected).then(alt.value(1.0)).otherwise(alt.value(0.6)),
        tooltip=[
            alt.Tooltip("token:N").title("Token"),
            alt.Tooltip("position:O").title("Position"),
        ],
    )
    label: alt.Chart = (
        base.transform_filter(is_selected)
        .mark_text(dy=-18, fontSize=14, fontWeight="bold")
        .encode(text="token:N")
    )
    return (points + label).properties(width="container", height=BAR_CHART_HEIGHT)


def activation_chart(
    data: pd.DataFrame, n_layers: int, n_scaler_groups: int
) -> alt.Chart:
    """Render a token's per-layer activation grid as a heatmap."""
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "layer:O",
                title="Tensor",
                # label every other tensor (0, 2, 4, ...)
                axis=alt.Axis(labelAngle=0, values=list(range(0, n_layers, 2))),
            ),
            y=alt.Y(
                "scaler_group:O",
                title="Scaler group",
                sort="descending",
                # label every other group (0, 2, 4, ...)
                axis=alt.Axis(values=list(range(0, n_scaler_groups, 2))),
            ),
            color=alt.Color(
                "activation:Q",
                title="Activation",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("layer:O", title="Tensor"),
                alt.Tooltip("scaler_group:O", title="Scaler group"),
                alt.Tooltip("activation:Q", title="Activation", format=".2f"),
            ],
        )
        .properties(width="container", height=CHART_HEIGHT)
    )
