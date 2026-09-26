from typing import cast
from unittest.mock import MagicMock

import llm
import numpy as np
import pytest
import torch


@pytest.fixture
def profiled() -> llm.ProfiledSmolLM:
    """ProfiledSmolLM with a mock tokenizer, skipping the model download."""
    profiled = object.__new__(llm.ProfiledSmolLM)
    profiled.tokenizer = MagicMock()
    profiled.tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [
        f"Ġtok{i}" for i in ids
    ]
    return profiled


def test_clean_token() -> None:
    """Test clean token removes weird llm standin chars"""
    assert llm.ProfiledSmolLM._clean_token("ĠParisĊ") == " Paris\n"
    assert llm.ProfiledSmolLM._clean_token("Paris") == "Paris"


def test_template_prompt(profiled: llm.ProfiledSmolLM) -> None:
    tokenizer = cast(MagicMock, profiled.tokenizer)
    tokenizer.apply_chat_template.return_value = "<templated>"

    assert profiled._template_prompt("Hello, world!") == "<templated>"

    kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["conversation"][-1] == {"role": "user", "content": "Hello, world!"}
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is False


def test_template_prompt_non_str(profiled: llm.ProfiledSmolLM) -> None:
    tokenizer = cast(MagicMock, profiled.tokenizer)
    tokenizer.apply_chat_template.return_value = [1, 2, 3]

    with pytest.raises(TypeError):
        profiled._template_prompt("Hello, world!")


def test_parse_output_ids(profiled: llm.ProfiledSmolLM) -> None:
    """
    Generation is only profiled for n-1 output tokens.
    This test ensures only profiled tokens are returned.
    """

    profiled.layers = {0: llm._LayerActivations(llm_layer=0, tokens=[[0.0], [0.0]])}

    # ids past the profiled tokens (e.g. a trailing eos) are dropped
    assert profiled._parse_output_ids(torch.tensor([5, 6, 7])) == [" tok5", " tok6"]


def test_parse_output_ids_too_few(profiled: llm.ProfiledSmolLM) -> None:
    profiled.layers = {0: llm._LayerActivations(llm_layer=0, tokens=[[0.0], [0.0]])}

    with pytest.raises(AssertionError):
        profiled._parse_output_ids(torch.tensor([5]))


def test_mean_pool_tensor() -> None:
    assert llm.mean_pool_tensor([1.0, 3.0, 5.0, 7.0], n_groups=2) == [2.0, 6.0]


def test_mean_pool_tensor_default_groups() -> None:
    """
    Check that mean pool function evenly distributes our input into
    N_SCALER_GROUPS buckets
    """

    pooled = llm.mean_pool_tensor([1.0] * llm.N_SCALER_GROUPS * 4)
    assert pooled == [1.0] * llm.N_SCALER_GROUPS


def test_mean_pool_tensor_uneven_groups() -> None:
    """Test mean pool function errors when input % N_SCALER_GROUPS != 0"""
    with pytest.raises(ValueError):
        llm.mean_pool_tensor([1.0, 2.0, 3.0, 4.0, 5.0], n_groups=2)


def test_parse_mean_pooled_tensors(profiled: llm.ProfiledSmolLM) -> None:
    width = llm.N_SCALER_GROUPS * 2
    profiled.layers = {
        0: llm._LayerActivations(llm_layer=0, tokens=[[0.0] * width, [1.0] * width]),
        1: llm._LayerActivations(llm_layer=1, tokens=[[2.0] * width, [3.0] * width]),
    }

    tensors = profiled._parse_mean_pooled_tensors()

    # indexed [token][layer][scaler_group]
    assert tensors == [
        [[0.0] * llm.N_SCALER_GROUPS, [2.0] * llm.N_SCALER_GROUPS],
        [[1.0] * llm.N_SCALER_GROUPS, [3.0] * llm.N_SCALER_GROUPS],
    ]


def test_parse_top_k_logits(profiled: llm.ProfiledSmolLM) -> None:
    """test top k logits implements softmax and returns k elements"""

    vocab = llm.TOP_K + 2

    # simulate scores per 'logit' or output token as a simple range
    step = torch.FloatTensor([list(range(vocab))])

    top_probabilities = profiled._parse_top_k_logits((step,))

    assert top_probabilities == [
        pytest.approx(
            {
                " tok9": 0.6321,
                " tok8": 0.2326,
                " tok7": 0.0856,
                " tok6": 0.0315,
                " tok5": 0.0116,
                " tok4": 0.0043,
                " tok3": 0.0016,
                " tok2": 0.0006,
            },
            abs=1e-4,
        )
    ]
    assert list(top_probabilities[0]) == [f" tok{i}" for i in range(9, 1, -1)]


def test_project_2d() -> None:
    vectors = [torch.rand(8) for _ in range(5)]

    coords = llm.ProfiledSmolLM._project_2d(vectors)

    # one (x, y) float pair per input vector
    assert isinstance(coords, list)
    assert len(coords) == len(vectors)
    for coord in coords:
        assert isinstance(coord, tuple)
        assert len(coord) == 2
        assert all(isinstance(value, float) for value in coord)


def test_parse_token_embeddings(profiled: llm.ProfiledSmolLM) -> None:
    """Test that given a list of hidden states for model layers,
    only last layer is grabbed as token's embedding"""

    def pass_(
        last_layer: list[float], query_len: int = 1
    ) -> tuple[torch.FloatTensor, ...]:
        """One forward pass's hidden states, (batch, query_len, hidden) per layer."""
        first_layer = torch.FloatTensor([[[99.0] * 3] * query_len])
        return (first_layer, torch.FloatTensor([[last_layer] * query_len]))

    hidden_states = (
        pass_([99.0, 99.0, 99.0], query_len=4),  # prefill, skipped
        pass_([0.0, 0.0, 0.0]),
        pass_([3.0, 4.0, 0.0]),
    )

    coords = profiled._parse_token_embeddings(hidden_states)

    # only the last layer of each generated token is projected
    assert coords == pytest.approx(
        llm.ProfiledSmolLM._project_2d(
            [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([3.0, 4.0, 0.0])]
        )
    )


def test_parse_average_layer_attentions(profiled: llm.ProfiledSmolLM) -> None:
    """LLM's generate attention scores using multiple 'heads'.
    Test _parse_average_layer_attentions grabs the average attention score
    across all these heads"""

    # (batch, heads, generated_length, sequence_length)
    layer_0 = torch.FloatTensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    layer_1 = torch.FloatTensor([[[[0.5, 0.5]], [[1.0, 0.0]]]])

    (attention,) = profiled._parse_average_layer_attentions(((layer_0, layer_1),))

    # one (layers, context) array per token, averaged over heads
    np.testing.assert_allclose(attention, [[0.5, 0.5], [0.75, 0.25]])
