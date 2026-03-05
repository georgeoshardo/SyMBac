from __future__ import annotations

import yaml


_DEFAULT_VARIANTS = [
    {"cell_max_length": 6.2, "cell_width": 1.0},
    {"cell_max_length": 6.8, "cell_width": 1.1},
]


def default_variants_yaml() -> str:
    return yaml.safe_dump(_DEFAULT_VARIANTS, sort_keys=False)


def load_variants_yaml_text(text: str) -> list[dict[str, float]]:
    data = yaml.safe_load(text) or []
    if not isinstance(data, list):
        raise ValueError("Variants must be a YAML list of mapping objects.")

    variants: list[dict[str, float]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Variant #{idx} must be a mapping.")
        parsed: dict[str, float] = {}
        for key, value in item.items():
            if key not in {"cell_max_length", "cell_width", "max_length_std", "width_std", "lysis_p"}:
                raise ValueError(
                    f"Variant #{idx} has unsupported key {key!r}. "
                    "Supported: cell_max_length, cell_width, max_length_std, width_std, lysis_p."
                )
            parsed[key] = float(value)
        variants.append(parsed)
    return variants
