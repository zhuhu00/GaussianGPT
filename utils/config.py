from typing import Any, Dict, List

from hydra.utils import instantiate

from conf.dataclasses import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_LAYER_CONFIGS,
    GaussianFeatures,
    LayerType,
)


def instantiate_feature_config(
    cfg: Dict[GaussianFeatures, Dict[str, Any]],
) -> Dict[GaussianFeatures, Dict[str, Any]]:
    """
    Walk every Mapping or Sequence in cfg, and whenever you find a dict with '_target_':
      - call hydra.utils.instantiate on it,
      - replace it with the resulting object.
    Returns a new data structure; original cfg is untouched.
    """

    out = {}
    # Add default values to the feature config
    for gaussian_feature, feature_config in cfg.items():
        full_feature_config = DEFAULT_FEATURE_CONFIG.copy()
        full_feature_config.update(feature_config)  # Update with provided values

        # Instantiate the internal representation if it exists
        if "internal_representation" in full_feature_config:
            # If internal_representation is provided, instantiate it
            full_feature_config["internal_representation"] = instantiate(
                full_feature_config["internal_representation"]
            )

        # Complete the layer config for the feature head
        full_feature_config["feature_head"] = complete_layer_config(
            full_feature_config["feature_head"]
        )

        out[gaussian_feature] = full_feature_config

    return out


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten a dict of dicts into a single‐level dict."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):
                v = str(v)
            items.append((new_key, v))
    return dict(items)


def complete_layer_config(
    untyped_config: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Adds unused values to the layer config dicts and asserts that they are valid.
    """

    out = []
    for layer_config in untyped_config:
        assert "kind" in layer_config, "Each layer config must have a 'kind' key."
        kind: str = layer_config["kind"]  # eg. "residual_block"

        assert kind in set(
            item.value for item in LayerType
        ), f"Layer kind {kind} is not a valid LayerType."
        assert (
            kind in DEFAULT_LAYER_CONFIGS
        ), f"Missing default config for layer kind: {kind}"

        full_config = DEFAULT_LAYER_CONFIGS[kind].copy()
        full_config.update(layer_config)  # Update with provided values

        out.append(full_config)

    return out
