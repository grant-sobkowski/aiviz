import random

import numpy as np
from llm import ProfiledToken, mean_pool_tensor


def _make_tensors(seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    tensors = []
    for layer in range(30):  # matches ProfiledSmolLM's 30 hidden layers
        tensor = []
        # 768 values per tensor (common LLM hidden dim); 768 / 32 (scaler groups) = 24 evenly
        for _ in range(768):
            # Skewed toward 0 like real post-ReLU activations: most near zero,
            # sparse high-magnitude spikes
            v = rng.expovariate(3.5)
            tensor.append(min(v, 1.0))

        # Simulate attention sink: a handful of heads fire strongly each layer
        num_spikes = rng.randint(1, 4)
        for _ in range(num_spikes):
            idx = rng.randint(0, 767)
            tensor[idx] = rng.uniform(0.75, 1.0)

        tensors.append(tensor)
    return tensors


TOKENS: list[tuple[str, list[list[float]]]] = [
    ("The", _make_tensors(0)),
    (" capital", _make_tensors(1)),
    (" of", _make_tensors(2)),
    (" France", _make_tensors(3)),
    (" is", _make_tensors(4)),
    (" Paris", _make_tensors(5)),
    (".", _make_tensors(6)),
]


def mock_tokens() -> list[ProfiledToken]:
    """Fixture tokens with made-up probabilities and attention, for running without the LLM."""
    prompt_tokens = ["<prompt>"] * 8
    tokens = []
    for i, (text, tensors) in enumerate(TOKENS):
        p = random.random()
        pooled_tensors = [mean_pool_tensor(layer) for layer in tensors]
        context = prompt_tokens + [t for t, _ in TOKENS[: i + 1]]
        attention = np.random.dirichlet(np.ones(len(context)), size=len(pooled_tensors))
        embedding = (random.gauss(0, 10), random.gauss(0, 10))
        tokens.append(
            ProfiledToken(
                text, pooled_tensors, {text: p}, attention, context, embedding
            )
        )
    return tokens
