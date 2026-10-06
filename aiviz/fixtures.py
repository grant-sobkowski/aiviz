import random

import numpy as np
from llm import ProfiledToken, mean_pool_tensor


def _make_tensors(seed: int, layers: int = 30, scalars: int = 768) -> list[list[float]]:
    rng = random.Random(x=seed)
    tensors: list[list[float]] = []
    for _ in range(layers):
        tensor: list[float] = []
        for _ in range(scalars):
            # Skewed toward 0 like real post-ReLU activations: most near zero,
            # sparse high-magnitude spikes
            v: float = rng.expovariate(lambd=3.5)
            tensor.append(min(v, 1.0))

        # Simulate attention sink: a handful of heads fire strongly each layer
        num_spikes: int = rng.randint(a=1, b=4)
        for _ in range(num_spikes):
            idx: int = rng.randint(a=0, b=767)
            tensor[idx] = rng.uniform(a=0.75, b=1.0)

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
    prompt_tokens: list[str] = ["<prompt>"] * 8
    tokens: list[ProfiledToken] = []
    for i, (text, tensors) in enumerate(iterable=TOKENS):
        p: float = random.random()
        pooled_tensors: list[list[float]] = [
            mean_pool_tensor(tensor=layer) for layer in tensors
        ]
        context: list[str] = prompt_tokens + [t for t, _ in TOKENS[: i + 1]]
        attention = np.random.dirichlet(np.ones(len(context)), size=len(pooled_tensors))
        embedding: tuple[float, float] = (random.gauss(0, 10), random.gauss(0, 10))
        tokens.append(
            ProfiledToken(
                text, pooled_tensors, {text: p}, attention, context, embedding
            )
        )
    return tokens
