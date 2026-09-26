import itertools
from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from typing import Any, cast, no_type_check

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.hooks import RemovableHandle
from transformers import (
    GPT2Tokenizer,
    LlamaForCausalLM,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

TOP_K = 8  # number of most likely candidate tokens recorded per output token
MAX_NEW_TOKENS = 300  # cap on llm output
N_SCALER_GROUPS = 32  # each layer's activation vector is mean-pooled down to this width


def mean_pool_tensor(
    tensor: list[float], n_groups: int = N_SCALER_GROUPS
) -> list[float]:
    """Mean-pool a single layer's raw activation vector down to n_groups batches."""
    batch_size = len(tensor) // n_groups
    return [
        sum(batch) / len(batch)
        for batch in itertools.batched(tensor, batch_size, strict=True)
    ]


logger = getLogger(__name__)


@dataclass
class ProfiledToken:
    """Represents a single output token.
    Properties:
        text: textual representation of the output token.
        tensors: per-layer activations, each mean-pooled down to N_SCALER_GROUPS weights
            (0.00 - 1.00).
        top_probabilities: the TOP_K most likely tokens at this position, mapped to their
            probabilities, most likely first.
        attention: (layers, context) array of how much this token attended to each token in
            its context (prompt, previous output tokens, and itself), averaged over heads.
            Rows sum to 1.
        context_tokens: text of the tokens `attention` columns refer to (shared with the
            other tokens' lists where possible, do not mutate).
        embedding_2d: (x, y) of this token's final-layer hidden state (the vector fed to the
            output softmax) in a 2D PCA projection shared by all tokens of the response.
    """

    text: str
    tensors: list[list[float]]
    top_probabilities: dict[str, float]
    attention: np.ndarray
    context_tokens: list[str]
    embedding_2d: tuple[float, float]


@dataclass
class _LayerActivations:
    """Layer activation data for all forward passes ran
    Properties:
        llm_layer: index of this object's layer.
        tokens: hidden layer scalar activation values, per token.
    """

    llm_layer: int
    tokens: list[list[float]]


@dataclass
class ProfiledModelOutput:
    sequences: torch.LongTensor
    logits: tuple[torch.FloatTensor]
    attentions: list[np.ndarray]
    embeddings_2d: list[tuple[float, float]]
    top_probabilities: list[dict[str, float]]
    tensors: list[list[list[float]]]
    input_tokens: list[str]
    output_tokens: list[str]


class ProfiledSmolLM:
    MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"

    def __init__(self) -> None:
        self._download()
        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_ID,
            device_map="auto",
            local_files_only=True,  # no network calls to HF; we do these in _download
            attn_implementation="eager",  # sdpa can't return attention weights
        )
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_ID,
            local_files_only=True,  # tokenizer also downloaded in _download
        )

    def _download(self) -> None:
        """Download model and tokenizer froom HF"""
        snapshot_download(repo_id=self.MODEL_ID)

    def run(self, user_input: str) -> list[ProfiledToken]:
        """Generate a response to user input and return each output token with its layer activations."""

        prompt: str = self._template_prompt(user_input)
        output: ProfiledModelOutput = self._generate_single_sequence(prompt)

        self.embeddings_2d: list[tuple[float, float]] = output.embeddings_2d
        self.top_probabilities: list[dict[str, float]] = output.top_probabilities

        profiled_tokens: list[ProfiledToken] = []
        for i, token in enumerate(output.output_tokens):
            profiled_tokens.append(
                ProfiledToken(
                    token,
                    output.tensors[i],
                    self.top_probabilities[i],
                    output.attentions[i],
                    output.input_tokens + output.output_tokens[: i + 1],
                    self.embeddings_2d[i],
                )
            )

        return profiled_tokens

    def _template_prompt(self, user_input: str) -> str:
        template: list[dict] = [
            {
                "role": "system",
                "content": "Answer concisely, in no more than one paragraph.",
            },
            {"role": "user", "content": user_input},
        ]
        prompt: Any = self.tokenizer.apply_chat_template(
            conversation=template, add_generation_prompt=True, tokenize=False
        )
        if isinstance(prompt, str) is False:
            raise TypeError(
                f"Expected str from tokenizer.apply_chat_template, got: {type(prompt)}"
            )
        else:
            prompt: str = cast(str, prompt)

        return prompt

    def _parse_output_ids(self, generated_ids: torch.Tensor) -> list[str]:
        """Decode the generated token ids into the text tokens that were profiled."""

        num_profiled_tokens = len(self.layers[0].tokens)
        assert len(generated_ids) >= num_profiled_tokens, (
            f"Expected at least {num_profiled_tokens} output tokens, got: {len(generated_ids)}"
        )

        tokens = self.tokenizer.convert_ids_to_tokens(
            generated_ids[:num_profiled_tokens].tolist()
        )
        assert isinstance(tokens, list)
        return [self._clean_token(token) for token in tokens]

    @staticmethod
    def _clean_token(token: str) -> str:
        """Replace the tokenizer's byte-level mappings for spaces and newlines."""
        return token.replace("Ġ", " ").replace("Ċ", "\n")

    @no_type_check
    def _generate_single_sequence(self, prompt: str) -> ProfiledModelOutput:

        input_tokens: list[str] = self.tokenizer.tokenize(text=prompt)
        input_token_ids: int | list[int] = self.tokenizer.convert_tokens_to_ids(
            input_tokens
        )

        # generate expects input tensor shape (batch, token_id)
        input_ids_batched: torch.LongTensor = torch.LongTensor(
            input_token_ids
        ).unsqueeze(0)

        # reset profiling hooks, if any
        hooks: list[RemovableHandle] = []
        self.layers: dict[int, _LayerActivations] = {
            i: _LayerActivations(llm_layer=i, tokens=[])
            for i in range(len(self.model.model.layers))
        }

        # register profiling callbacks on hidden layer of each ML block in model
        for i, layer in enumerate(self.model.model.layers):
            hook: RemovableHandle = layer.mlp.down_proj.register_forward_hook(
                self._create_layer_hook(layer=i)
            )
            hooks.append(hook)

        result: GenerateDecoderOnlyOutput = self.model.generate(
            inputs=input_ids_batched,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_logits=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        # cleanup profiling hooks
        for hook in hooks:
            hook.remove()

        sequences: torch.LongTensor = result.sequences

        # len(sequences) is >1 for batched generation only
        if len(result.sequences) != 1:
            raise ValueError(
                f"Expected results.sequences to be length 1, got: {len(sequences)}"
            )

        generated_tokens: int = len(result.logits)
        generated_ids: torch.Tensor = sequences[0][-generated_tokens:]
        profiled_output = ProfiledModelOutput(
            sequences=sequences[0],
            logits=result.logits,
            attentions=self._parse_average_layer_attentions(
                attentions=result.attentions
            ),
            embeddings_2d=self._parse_token_embeddings(
                hidden_states=result.hidden_states
            ),
            top_probabilities=self._parse_top_k_logits(result.logits),
            tensors=self._parse_mean_pooled_tensors(),
            input_tokens=input_tokens,
            output_tokens=self._parse_output_ids(generated_ids),
        )
        return profiled_output

    def _create_layer_hook(
        self, layer: int
    ) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
        """Build a forward hook that records the given layer's normalized activations."""

        def hook(_module: torch.nn.Module, _input: tuple, output: torch.Tensor) -> None:
            """Record the layer's normalized output activations for a single generated token."""
            del _module, _input  # unused

            # skip adding hooks for prefill
            if output.shape[1] > 1:
                return

            normalized = self.model.model.norm(output.detach())
            self.layers[layer].tokens.append(normalized[0, 0].tolist())

        return hook

    def _parse_mean_pooled_tensors(self) -> list[list[list[float]]]:
        """Group the per-layer activations recorded by the profiling hooks back out per
        generated token, mean-pooling each layer's activation vector down to
        N_SCALER_GROUPS batches for charting."""
        num_tokens = len(self.layers[0].tokens)
        return [
            [
                mean_pool_tensor(layer_tensors.tokens[i])
                for layer_tensors in self.layers.values()
            ]
            for i in range(num_tokens)
        ]

    def _parse_top_k_logits(
        self, logits: tuple[torch.FloatTensor]
    ) -> list[dict[str, float]]:
        """From each generated token's logits, compute the TOP_K most likely candidate
        tokens at that position (text -> probability, most likely first)."""
        top_probabilities: list[dict[str, float]] = []
        for step in logits:
            probs = torch.softmax(step[0].float(), dim=-1)

            top = torch.topk(probs, TOP_K)
            top_tokens = self.tokenizer.convert_ids_to_tokens(top.indices.tolist())
            top_probabilities.append(
                {
                    self._clean_token(t): p
                    for t, p in zip(top_tokens, top.values.tolist())
                }
            )
        return top_probabilities

    def _parse_token_embeddings(
        self, hidden_states: tuple[tuple[torch.FloatTensor, ...], ...]
    ) -> list[tuple[float, float]]:
        """Project each generated token's final-layer hidden state (after the final norm,
        i.e. the vector fed to the output softmax) into a 2D PCA space shared by all
        tokens of the response.

        Where hidden_states[token_idx][layer_idx] -> torch.FloatTensor of shape
        (batch, query_len, hidden), and hidden_states[0] is the prefill pass (skipped).
        """

        vectors: list[torch.Tensor] = []
        for token in hidden_states[1:]:
            last_layer: torch.FloatTensor = token[-1]
            last_layer_vector: torch.Tensor = last_layer[0, 0, :].float().cpu()
            vectors.append(last_layer_vector)

        coords_by_token: list[tuple[float, float]] = self._project_2d(vectors)
        return coords_by_token

    @staticmethod
    def _project_2d(vectors: list[torch.Tensor]) -> list[tuple[float, float]]:
        """PCA-project list of vectors onto their top two principal components, as (n, 2)."""

        nested_vectors: torch.Tensor = torch.stack(vectors)
        centered = nested_vectors - nested_vectors.mean(dim=0)

        _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
        k = min(2, len(singular_values))
        coords = (centered @ components[:k].T).numpy()

        # fewer than two tokens (or dimensions): pad the missing components with zeros
        padded_coords = np.pad(coords, ((0, 0), (0, 2 - k)))

        # format our output as (x, y) per input vector
        coords_by_token: list[tuple[float, float]] = [
            (float(x), float(y)) for x, y in padded_coords
        ]
        return coords_by_token

    def _parse_average_layer_attentions(
        self, attentions: tuple[tuple[torch.FloatTensor]]
    ) -> list[np.ndarray]:
        """Average attentions per layer across attention block heads

        Where GenerateDecoderOnlyOutput.attentions[token_idx][layer_idx] -> torch.FloatTensor
        And where tensor shape is (batch, heads, generated_length, sequence_length)
        - heads: model specific; e.g. SmolLM has 9 attention heads
        - generated_length: tokens to get attentions for (1 for all non-prefill passes)
        - sequence_length: tokens to attend to (including current)
        """

        parsed: list[np.ndarray] = []

        for i in range(len(attentions)):
            layers: tuple[torch.FloatTensor] = attentions[i]
            token_layers: list[np.ndarray] = []

            for j in range(len(layers)):
                heads: torch.FloatTensor = layers[j]
                # TODO: validate tensor shape here is (1, any, 1, any)
                layer_average: torch.Tensor = (
                    heads[
                        0, :, 0, :
                    ].mean(  # convert tensor to just be (head, sequence)
                        dim=0
                    )  # attentions averaged over heads
                )
                chartable: np.ndarray = layer_average.float().cpu().numpy()
                token_layers.append(chartable)

            # stack this token's per-layer vectors into a single (layers, context) array
            parsed.append(np.array(token_layers))

        return parsed
