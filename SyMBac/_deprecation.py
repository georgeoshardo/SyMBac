"""Shared helpers for handling deprecated API aliases."""

from __future__ import annotations

import warnings

_DEPRECATED_ALIAS_REMOVAL_DATE = "2026-09-01"
_UNSET = object()


def _resolve_deprecated_parameter(
    api_name,
    new_name,
    new_value,
    legacy_name,
    legacy_value,
    compatibility_note=None,
):
    """Resolve a new/legacy parameter pair and emit a single warning when needed."""
    new_provided = new_value is not _UNSET
    legacy_provided = legacy_value is not _UNSET

    if legacy_provided:
        if new_provided and new_value != legacy_value:
            raise ValueError(
                f"{api_name}: `{new_name}` and deprecated `{legacy_name}` were both provided with different values."
            )

        warning_message = (
            f"`{legacy_name}` is deprecated and will be removed on "
            f"{_DEPRECATED_ALIAS_REMOVAL_DATE}. Use `{new_name}` instead."
        )
        if compatibility_note:
            warning_message = f"{warning_message} {compatibility_note}"
        warnings.warn(warning_message, FutureWarning, stacklevel=3)
        return legacy_value, True

    if not new_provided:
        raise TypeError(f"{api_name} missing required argument: '{new_name}'")
    return new_value, False


def _require_provided(api_name, arg_name, value):
    """Raise a TypeError for required arguments left unset."""
    if value is _UNSET:
        raise TypeError(f"{api_name} missing required argument: '{arg_name}'")
