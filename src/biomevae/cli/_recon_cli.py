"""Shared utilities for reconstruction-related CLI entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from biomevae.reconstruction import CrossValResult


def _coerce_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_assignments(items: Iterable[str]) -> Dict[str, object]:
    """Parse ``key=value`` pairs provided on the command line."""

    result: Dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid assignment '{item}'. Use key=value format.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("Assignment keys cannot be empty.")
        result[key] = _coerce_value(value.strip())
    return result


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list or range like "2-5" into integers."""

    items: list[int] = []
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        # Detect ranges (e.g. "2-5") but not negative numbers (e.g. "-5").
        # A range has a '-' that is not the first character.
        dash_pos = token.find("-", 1)
        if dash_pos > 0:
            start_str = token[:dash_pos].strip()
            end_str = token[dash_pos + 1:].strip()
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise SystemExit(f"Invalid range '{token}'. Use MIN-MAX format.") from exc
            if start > end:
                raise SystemExit(f"Invalid range '{token}': MIN must be <= MAX.")
            items.extend(range(start, end + 1))
        else:
            try:
                items.append(int(token))
            except ValueError as exc:
                raise SystemExit(f"Invalid integer value '{token}'.") from exc
    if not items:
        raise SystemExit("Rank candidates cannot be empty.")
    return items


def load_json(path: str) -> Mapping[str, object]:
    """Load a JSON file, raising ``SystemExit`` with a helpful message on error."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:  # pragma: no cover - runtime safety
        raise SystemExit(f"Could not read '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - runtime safety
        raise SystemExit(f"Invalid JSON in '{path}': {exc}") from exc


def result_to_dict(result: CrossValResult) -> Dict[str, object]:
    """Convert :class:`CrossValResult` instances into JSON-serialisable dicts."""

    return {
        "fold_metrics": [dict(metrics) for metrics in result.fold_metrics],
        "mean_metrics": dict(result.mean_metrics),
        "std_metrics": dict(result.std_metrics),
        "metadata": dict(result.metadata) if result.metadata is not None else None,
    }


def dict_to_result(payload: Mapping[str, Any]) -> CrossValResult:
    """Recreate :class:`CrossValResult` instances from JSON payloads."""

    def _coerce_metric_mapping(
        name: str, items: Mapping[str, Any]
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, value in items.items():
            try:
                result[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Metric '{key}' in '{name}' must be numeric (got {value!r})."
                ) from exc
        return result

    if not isinstance(payload, Mapping):
        raise SystemExit("Results must be a mapping of method names to metrics.")

    try:
        fold_metrics_raw = payload["fold_metrics"]
        mean_metrics_raw = payload["mean_metrics"]
        std_metrics_raw = payload["std_metrics"]
    except KeyError as exc:  # pragma: no cover - runtime validation
        raise SystemExit(f"Missing field '{exc.args[0]}' in result payload.") from exc

    if not isinstance(fold_metrics_raw, Sequence):
        raise SystemExit("'fold_metrics' must be a sequence of metric mappings.")

    fold_metrics: list[Dict[str, float]] = []
    for index, metrics in enumerate(fold_metrics_raw):
        if not isinstance(metrics, Mapping):
            raise SystemExit(
                f"Fold metrics at position {index} must be a mapping of metric names to values."
            )
        fold_metrics.append(_coerce_metric_mapping("fold_metrics", metrics))

    if not isinstance(mean_metrics_raw, Mapping):
        raise SystemExit("'mean_metrics' must be a mapping of metric names to values.")
    if not isinstance(std_metrics_raw, Mapping):
        raise SystemExit("'std_metrics' must be a mapping of metric names to values.")

    metadata_raw = payload.get("metadata")
    metadata: Mapping[str, Any] | None
    if metadata_raw is None:
        metadata = None
    elif isinstance(metadata_raw, Mapping):
        metadata = dict(metadata_raw)
    else:
        raise SystemExit("'metadata' must be a mapping when provided.")

    return CrossValResult(
        fold_metrics=fold_metrics,
        mean_metrics=_coerce_metric_mapping("mean_metrics", mean_metrics_raw),
        std_metrics=_coerce_metric_mapping("std_metrics", std_metrics_raw),
        metadata=metadata,
    )


def dump_result(payload: Mapping[str, object], output: str | None) -> None:
    """Serialise results to JSON, optionally writing them to disk."""

    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    print(text)
