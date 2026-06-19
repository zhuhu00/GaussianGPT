from __future__ import annotations

from typing import Callable, Optional, Tuple

SamplingParams = Tuple[float, Optional[int], Optional[float]]
SamplingParamsResolver = Callable[[int], SamplingParams]


def build_token_type_sampling_params_resolver(
    *,
    prompt_token_count: int,
    tokens_per_latent: int,
    num_position_tokens: int,
    default_temperature: float,
    default_top_k: Optional[int],
    default_top_p: Optional[float],
    feature_temperature: Optional[float],
    feature_top_k: Optional[int],
    feature_top_p: Optional[float],
) -> Optional[SamplingParamsResolver]:
    if feature_temperature is None and feature_top_k is None and feature_top_p is None:
        return None

    if prompt_token_count < 0:
        raise ValueError("prompt_token_count must be >= 0.")
    if tokens_per_latent < 1:
        raise ValueError("tokens_per_latent must be >= 1.")
    if num_position_tokens < 0 or num_position_tokens > tokens_per_latent:
        raise ValueError("num_position_tokens must be in [0, tokens_per_latent].")

    feature_params: SamplingParams = (
        (
            float(feature_temperature)
            if feature_temperature is not None
            else float(default_temperature)
        ),
        int(feature_top_k) if feature_top_k is not None else default_top_k,
        float(feature_top_p) if feature_top_p is not None else default_top_p,
    )
    default_params: SamplingParams = (
        float(default_temperature),
        default_top_k,
        default_top_p,
    )

    def _resolver(num_generated: int) -> SamplingParams:
        if num_generated < 0:
            raise ValueError("num_generated must be >= 0.")
        slot_idx = int((prompt_token_count + num_generated) % tokens_per_latent)
        if slot_idx < num_position_tokens:
            return default_params
        return feature_params

    return _resolver
