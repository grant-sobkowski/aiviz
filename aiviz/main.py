import os
import time
from collections.abc import Iterator, Sequence

import fixtures
import numpy as np
import pandas as pd
import streamlit as st
import ui
from llm import N_SCALER_GROUPS, ProfiledSmolLM, ProfiledToken
from streamlit.elements.lib.mutable_tab_container import TabContainer

USE_MOCK_LLM: bool = os.environ.get(key="USE_MOCK_LLM", default="").lower() in (
    "1",
    "true",
    "yes",
)

defaults = {
    "prompt": "",
    "timings": None,
    "tokens": [],  # list[ProfiledToken]
    "token_idx": 0,
    "model_id": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

N_LAYERS = 30
INPUT_PLACEHOLDER = "Enter a prompt"
STREAM_DELAY_SECONDS = 0.10
LOAD_STEP_LABEL = f"Loading {st.session_state.model_id}"
GENERATE_STEP_LABEL = "Generating response"
DONE_STEP_LABEL = "Generated response"
ATTENTION_TOP_N = 8  # context tokens shown in the attention chart
ALL_LAYERS = "Tensor Avg."
ATTENTION_SELECT_WIDTH_PX = 130


def main() -> None:
    """Renders chat window, token picker, and chart vizualizations"""
    st.set_page_config(page_title="aiviz", layout="wide")
    st.html(ui.STACK_BREAKPOINT_CSS + ui.TAB_CENTER_CSS + ui.ATTENTION_SELECT_CSS)

    with st.container(key="main_layout"):
        controls_col, chart_col = st.columns([1, 1])

        # render llm chat and token selector
        with controls_col:
            output_box = st.container(border=True, height=300)
            pills_slot = st.empty()

            prompt: str | None = st.chat_input(INPUT_PLACEHOLDER)

            if st.session_state.tokens:
                with pills_slot.container(border=True):
                    token_picker(disabled=bool(prompt))
            with output_box:
                if prompt:
                    run_llm(prompt)
                else:
                    output_field()

        # render visualizations
        with chart_col, st.container(border=True, key="chart_box"):
            tabs: Sequence[TabContainer] = st.tabs(
                tabs=[
                    "Tensor Activations",
                    "Attention Weights",
                    "Output Softmax",
                    "Embedding Space",
                ]
            )
            activations: TabContainer = tabs[0]
            attentions: TabContainer = tabs[1]
            softmax: TabContainer = tabs[2]
            embedding: TabContainer = tabs[3]

            with activations, st.container(horizontal_alignment="center"):
                render_activations()
            with softmax, st.container(horizontal_alignment="center"):
                render_softmax_chart()
            with attentions:
                render_attention_chart()
            with embedding, st.container(horizontal_alignment="center"):
                render_embedding_chart()


def output_field() -> None:
    """Renders llm output box"""
    tokens: list[ProfiledToken] = st.session_state.tokens
    if not tokens:
        st.caption("Response will appear here.")
        return

    with st.chat_message("user"):
        st.write(st.session_state.prompt)

    if st.session_state.timings:
        with st.chat_message("assistant"):
            render_timeline(*st.session_state.timings)

    with st.chat_message("assistant"):
        st.write("".join(t.text for t in tokens))


def token_picker(disabled: bool = False) -> None:
    """Renders outputted tokens as pill inputs for control of vizualizations"""
    tokens: list[ProfiledToken] = st.session_state.tokens
    last_idx = len(tokens) - 1

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(":material/search: **Token Selector**")
        st.badge(f"{st.session_state.token_idx} / {last_idx}", color="gray")

    st.session_state.token_pill = st.session_state.token_idx
    labels = pill_labels([t.text for t in tokens])
    st.pills(
        "Tokens",
        options=range(len(tokens)),
        format_func=labels.__getitem__,
        selection_mode="single",
        key="token_pill",
        on_change=select_token,
        disabled=disabled,
        label_visibility="collapsed",
    )


def select_token() -> None:
    """Pill callback: show the clicked token's activations (ignores deselecting a pill)."""
    idx = st.session_state.token_pill
    if idx is not None:
        st.session_state.token_idx = idx


def pill_labels(tokens: list[str]) -> list[str]:
    """
    Create display label per token.

    Duplicate tokens made unique with invisible characters.
    """
    seen: dict[str, int] = {}
    labels = []
    for token in tokens:
        text = token.strip() and token or repr(token)
        n = seen.get(text.strip(), 0)
        seen[text.strip()] = n + 1
        labels.append(text + "\u200b" * n)
    return labels


def run_llm(prompt: str) -> None:
    """Generate response for prompt and render output status / text"""
    with st.chat_message("user"):
        st.write(prompt)

    model = None
    model_id = ""

    with st.chat_message("assistant"):
        with st.status(LOAD_STEP_LABEL, type="step"):
            start = time.perf_counter()
            if not USE_MOCK_LLM:
                model = load_llm()
                model_id = model.MODEL_ID
            load_seconds = time.perf_counter() - start
            st.write(load_step_detail(load_seconds))

        with st.status(GENERATE_STEP_LABEL, type="step"):
            start = time.perf_counter()
            if model is None:
                tokens = fixtures.mock_tokens()
            else:
                tokens = model.run(prompt)
            generate_seconds = time.perf_counter() - start
            st.write(generate_step_detail(generate_seconds))

        st.status(DONE_STEP_LABEL, state="complete", type="step")

    with st.chat_message("assistant"):
        st.write_stream(stream_tokens(tokens))

    st.session_state.model_id = model_id
    st.session_state.prompt = prompt
    st.session_state.timings = (load_seconds, generate_seconds)
    st.session_state.tokens = tokens
    st.session_state.token_idx = 0
    st.rerun("app")


def stream_tokens(tokens: list[ProfiledToken]) -> Iterator[str]:
    """Yield the generated tokens' text one at a time, paced so the stream is visible."""
    for token in tokens:
        yield token.text
        time.sleep(STREAM_DELAY_SECONDS)


def load_step_detail(seconds: float) -> str:
    return f"Loaded model: {st.session_state.model_id} in {seconds:.1f} seconds."


def generate_step_detail(seconds: float) -> str:
    return f"Generated output in {seconds:.1f} seconds."


def render_timeline(load_seconds: float, generate_seconds: float) -> None:
    """Re-render a finished run's steps (same as the live ones, but already complete)."""
    with st.status(LOAD_STEP_LABEL, state="complete", type="step"):
        st.write(load_step_detail(load_seconds))
    with st.status(GENERATE_STEP_LABEL, state="complete", type="step"):
        st.write(generate_step_detail(generate_seconds))
    st.status(DONE_STEP_LABEL, state="complete", type="step")


def load_llm() -> ProfiledSmolLM:
    """Download (if needed) and load the local model once per server process."""
    return ProfiledSmolLM()


#          ╭──────────────────────────────────────────────────────────╮
#          │                     CHART RENDERING                      │
#          ╰──────────────────────────────────────────────────────────╯


def render_activations() -> None:
    """Right-hand column: current token's per-layer activation heatmap."""
    tokens: list[ProfiledToken] = st.session_state.tokens
    if tokens:
        data: pd.DataFrame = activation_frame(
            tokens[st.session_state.token_idx].tensors
        )
    else:
        data: pd.DataFrame = empty_grid()
    st.altair_chart(
        altair_chart=ui.activation_chart(
            data, n_layers=N_LAYERS, n_scaler_groups=N_SCALER_GROUPS
        ),
        width="stretch",
    )


def render_softmax_chart() -> None:
    """Current token's top candidate tokens by softmax probability, as a bar chart."""
    tokens: list[ProfiledToken] = st.session_state.tokens
    probs = tokens[st.session_state.token_idx].top_probabilities if tokens else {}
    data = pd.DataFrame(
        [
            {"token": repr(token), "probability": p, "rank": rank}
            for rank, (token, p) in enumerate(probs.items())
        ],
        columns=["token", "probability", "rank"],
    )
    st.altair_chart(altair_chart=ui.softmax_chart(data), width="stretch")


def render_attention_chart() -> None:
    """Current token's attention over the tokens before it (and itself), as a bar chart."""
    tokens: list[ProfiledToken] = st.session_state.tokens
    layer_choice: str | int = st.selectbox(
        "Tensor",
        [ALL_LAYERS, *range(N_LAYERS)],
        format_func=lambda layer: layer if layer == ALL_LAYERS else f"Tensor {layer}",
        key="attention_layer",
        width=ATTENTION_SELECT_WIDTH_PX,
        label_visibility="collapsed",
    )

    rows: list[dict] = []
    if tokens:
        token = tokens[st.session_state.token_idx]
        attention: np.ndarray = token.attention
        context: list[str] = token.context_tokens
        per_token = (
            attention.mean(axis=0)
            if layer_choice == ALL_LAYERS
            else attention[layer_choice]
        )
        top = np.argsort(per_token)[::-1][:ATTENTION_TOP_N]
        rows = [
            {
                "token": f"{context[i]!r} ({i})",
                "attention": float(per_token[i]),
                "rank": rank,
            }
            for rank, i in enumerate(top)
        ]
    data = pd.DataFrame(rows, columns=["token", "attention", "rank"])
    with st.container(horizontal_alignment="center"):
        st.altair_chart(
            altair_chart=ui.attention_chart(data, top_n=ATTENTION_TOP_N),
            width="stretch",
        )


def render_embedding_chart() -> None:
    """All output tokens in the 2D embedding space, with the selected token highlighted."""
    tokens: list[ProfiledToken] = st.session_state.tokens
    selected = st.session_state.token_idx
    data = pd.DataFrame(
        data=[
            {
                "token": repr(token.text),
                "position": i,
                "x": token.embedding_2d[0],
                "y": token.embedding_2d[1],
                "status": "Selected" if i == selected else "Other",
            }
            for i, token in enumerate(tokens)
        ],
        columns=["token", "position", "x", "y", "status"],
    )
    st.altair_chart(altair_chart=ui.embedding_chart(data), width="stretch")


def empty_grid() -> pd.DataFrame:
    """Zero-valued grid shown before any tokens have been generated."""
    layers, groups = range(N_LAYERS), range(N_SCALER_GROUPS)
    return pd.DataFrame(
        data=[
            {"layer": layer, "scaler_group": group, "activation": 0.0}
            for layer in layers
            for group in groups
        ]
    )


def activation_frame(tensors: list[list[float]]) -> pd.DataFrame:
    """Build the per-layer/scaler-group DataFrame for the activation heatmap from a
    token's already mean-pooled tensors."""
    return pd.DataFrame(
        [
            {"layer": layer, "scaler_group": group, "activation": value}
            for layer, tensor in enumerate(tensors)
            for group, value in enumerate(tensor)
        ]
    )


if __name__ == "__main__":
    main()
