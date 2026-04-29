from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import orjson
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support


TARGET_KINDS = ("inside", "start", "end", "left_open", "right_open")
BOUNDARY_TARGET_KINDS = ("start", "end", "left_open", "right_open")
DEFAULT_LABELS = ("sentence", "paragraph", "section", "dialogue", "list_item", "metadata")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CPU multi-label per-position model from streaming span JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--model-output")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--calibration-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-kind", choices=("sgd", "extra-trees"), default="extra-trees")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--class-balance-cap", type=float, default=50.0)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--threshold-steps", type=int, default=99)
    parser.add_argument(
        "--max-fit-inside-positions-per-label",
        type=int,
        default=60_000,
        help="Reservoir cap for non-boundary positive inside positions per label in the fit split. Use -1 for all.",
    )
    parser.add_argument(
        "--max-fit-negative-positions",
        type=int,
        default=200_000,
        help="Reservoir cap for fully negative/background positions in the fit split. Use -1 for all.",
    )
    parser.add_argument(
        "--max-calibration-inside-positions-per-label",
        type=int,
        default=20_000,
        help="Reservoir cap for non-boundary positive inside positions per label in the threshold-calibration split. Use -1 for all.",
    )
    parser.add_argument(
        "--max-calibration-negative-positions",
        type=int,
        default=60_000,
        help="Reservoir cap for fully negative/background positions in the threshold-calibration split. Use -1 for all.",
    )
    parser.add_argument(
        "--max-valid-inside-positions-per-label",
        type=int,
        default=30_000,
        help="Reservoir cap for non-boundary positive inside positions per label in the validation split. Use -1 for all.",
    )
    parser.add_argument(
        "--max-valid-negative-positions",
        type=int,
        default=80_000,
        help="Reservoir cap for fully negative/background positions in the validation split. Use -1 for all.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    rows = load_rows(Path(args.input))
    labels = list(dict.fromkeys(args.labels))
    train_indices, valid_indices = split_indices(len(rows), args.valid_frac, args.seed)
    fit_indices, calibration_indices = split_train_calibration(
        train_indices, args.calibration_frac, args.seed + 1
    )

    print(
        f"loaded rows={len(rows)} fit_rows={len(fit_indices)} "
        f"calibration_rows={len(calibration_indices)} valid_rows={len(valid_indices)}",
        flush=True,
    )
    fit_examples, fit_sampling = select_positions(
        rows,
        labels,
        fit_indices,
        max_inside_positions_per_label=args.max_fit_inside_positions_per_label,
        max_negative_positions=args.max_fit_negative_positions,
        seed=args.seed + 101,
    )
    calibration_examples, calibration_sampling = select_positions(
        rows,
        labels,
        calibration_indices,
        max_inside_positions_per_label=args.max_calibration_inside_positions_per_label,
        max_negative_positions=args.max_calibration_negative_positions,
        seed=args.seed + 151,
    )
    valid_examples, valid_sampling = select_positions(
        rows,
        labels,
        valid_indices,
        max_inside_positions_per_label=args.max_valid_inside_positions_per_label,
        max_negative_positions=args.max_valid_negative_positions,
        seed=args.seed + 202,
    )
    print(
        f"selected positions fit={len(fit_examples)} "
        f"calibration={len(calibration_examples)} valid={len(valid_examples)}",
        flush=True,
    )
    fit_feature_dicts, fit_targets, fit_row_ids, fit_positions = build_position_dataset(
        rows, labels, fit_examples
    )
    calibration_feature_dicts, calibration_targets, calibration_row_ids, calibration_positions = (
        build_position_dataset(rows, labels, calibration_examples)
    )
    valid_feature_dicts, valid_targets, valid_row_ids, valid_positions = build_position_dataset(
        rows, labels, valid_examples
    )

    vectorizer = DictVectorizer(sparse=True)
    x_fit = vectorizer.fit_transform(fit_feature_dicts)
    x_calibration = vectorizer.transform(calibration_feature_dicts)
    x_valid = vectorizer.transform(valid_feature_dicts)
    print(
        f"vectorized fit_shape={x_fit.shape} calibration_shape={x_calibration.shape} "
        f"valid_shape={x_valid.shape}",
        flush=True,
    )

    head_names = list(fit_targets)
    fit_target_arrays = {
        name: np.asarray(values, dtype=np.uint8) for name, values in fit_targets.items()
    }
    calibration_target_arrays = {
        name: np.asarray(values, dtype=np.uint8) for name, values in calibration_targets.items()
    }
    valid_target_arrays = {
        name: np.asarray(values, dtype=np.uint8) for name, values in valid_targets.items()
    }
    fit_matrix = np.column_stack([fit_target_arrays[name] for name in head_names])
    calibration_matrix = np.column_stack([calibration_target_arrays[name] for name in head_names])
    valid_matrix = np.column_stack([valid_target_arrays[name] for name in head_names])

    model_started = time.perf_counter()
    if args.model_kind == "extra-trees":
        print("fitting extra-trees multi-output model", flush=True)
        model_bundle = fit_extra_trees_multi_output(x_fit, fit_matrix, args=args)
        calibration_scores_matrix = predict_multi_output_scores(model_bundle, x_calibration)
        valid_scores_matrix = predict_multi_output_scores(model_bundle, x_valid)
    else:
        print("fitting independent sgd heads", flush=True)
        model_bundle, calibration_scores_matrix, valid_scores_matrix = fit_sgd_heads(
            x_fit,
            x_calibration,
            x_valid,
            fit_target_arrays,
            head_names,
            args=args,
        )
    model_seconds = time.perf_counter() - model_started
    print(f"model fit/predict seconds={model_seconds:.3f}", flush=True)

    thresholds: dict[str, float] = {}
    head_reports: dict[str, dict[str, Any]] = {}
    for column, name in enumerate(head_names):
        y_fit = fit_matrix[:, column]
        y_calibration = calibration_matrix[:, column]
        y_valid = valid_matrix[:, column]
        calibration_scores = calibration_scores_matrix[:, column]
        valid_scores = valid_scores_matrix[:, column]
        threshold = best_threshold(
            calibration_scores,
            y_calibration,
            threshold_steps=args.threshold_steps,
        )
        valid_predictions = (valid_scores >= threshold).astype(np.uint8)
        precision, recall, f1, _support = precision_recall_fscore_support(
            y_valid,
            valid_predictions,
            average="binary",
            zero_division=0,
        )
        thresholds[name] = threshold
        head_reports[name] = {
            "kind": name.split(":", 1)[0],
            "label": name.split(":", 1)[1],
            "threshold": threshold,
            "fit_positive": int(y_fit.sum()),
            "fit_negative": int(len(y_fit) - y_fit.sum()),
            "calibration_positive": int(y_calibration.sum()),
            "calibration_negative": int(len(y_calibration) - y_calibration.sum()),
            "valid_positive": int(y_valid.sum()),
            "valid_negative": int(len(y_valid) - y_valid.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "model": args.model_kind,
        }

    report = {
        "format": "charstreamer.streaming-training-report.v2",
        "classification_type": "multi_label_per_position",
        "model_kind": args.model_kind,
        "model_config": model_config(args),
        "target_semantics": {
            "inside": "position is inside a visible span for this label",
            "start": "position is the first visible character of a closed-left span",
            "end": "position is the final visible character of a closed-right span",
            "left_open": "position 0 is inside a span that started before the target",
            "right_open": "final position is inside a span that continues after the target",
        },
        "input": str(Path(args.input)),
        "labels": labels,
        "rows": len(rows),
        "train_rows": len(train_indices),
        "fit_rows": len(fit_indices),
        "calibration_rows": len(calibration_indices),
        "valid_rows": len(valid_indices),
        "positions": len(fit_row_ids) + len(calibration_row_ids) + len(valid_row_ids),
        "fit_positions": len(fit_row_ids),
        "calibration_positions": len(calibration_row_ids),
        "valid_positions": len(valid_row_ids),
        "feature_dim": int(x_fit.shape[1]),
        "sampling": {
            "strategy": "row_split_all_boundary_open_edges_label_stratified_inside_reservoir_global_negative_reservoir",
            "fit": fit_sampling,
            "calibration": calibration_sampling,
            "valid": valid_sampling,
        },
        "data_stats": dataset_stats(rows, labels),
        "head_metrics": head_reports,
        "group_metrics": group_metrics(head_reports),
        "model_seconds": model_seconds,
        "training_seconds": time.perf_counter() - started,
    }

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(orjson.dumps(json_safe(report), option=orjson.OPT_INDENT_2))

    if args.model_output:
        model_path = Path(args.model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": "charstreamer.streaming-model.v2",
                "labels": labels,
                "target_kinds": TARGET_KINDS,
                "head_names": head_names,
                "model_kind": args.model_kind,
                "model_config": model_config(args),
                "vectorizer": vectorizer,
                "model": model_bundle,
                "thresholds": thresholds,
                "report": json_safe(report),
            },
            model_path,
            compress=3,
        )

    print_summary(report)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    return rows


def split_indices(row_count: int, valid_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < valid_frac < 1.0:
        raise SystemExit("--valid-frac must be between 0 and 1")
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    valid_count = max(1, round(row_count * valid_frac))
    valid = sorted(indices[:valid_count])
    train = sorted(indices[valid_count:])
    if not train:
        raise SystemExit("training split is empty")
    return train, valid


def split_train_calibration(
    train_indices: list[int],
    calibration_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < calibration_frac < 1.0:
        raise SystemExit("--calibration-frac must be between 0 and 1")
    indices = list(train_indices)
    random.Random(seed).shuffle(indices)
    calibration_count = max(1, round(len(indices) * calibration_frac))
    calibration = sorted(indices[:calibration_count])
    fit = sorted(indices[calibration_count:])
    if not fit:
        raise SystemExit("fit split is empty")
    return fit, calibration


def select_positions(
    rows: list[dict[str, Any]],
    labels: list[str],
    row_indices: Iterable[int],
    *,
    max_inside_positions_per_label: int,
    max_negative_positions: int,
    seed: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    rng = random.Random(seed)
    boundary_positions: list[tuple[int, int]] = []
    inside_samples = {label: [] for label in labels}
    inside_seen = Counter()
    negative_sample: list[tuple[int, int]] = []
    negative_seen = 0
    total_positions = 0

    for row_index in row_indices:
        row = rows[row_index]
        text = row["text"]
        row_targets = build_row_targets(row, labels)
        for position in range(len(text)):
            total_positions += 1
            boundary_labels = positive_labels(row_targets, labels, BOUNDARY_TARGET_KINDS, position)
            if boundary_labels:
                boundary_positions.append((row_index, position))
                continue

            inside_labels = positive_labels(row_targets, labels, ("inside",), position)
            if inside_labels:
                for label in inside_labels:
                    inside_seen[label] += 1
                    reservoir_add(
                        inside_samples[label],
                        (row_index, position),
                        inside_seen[label],
                        max_inside_positions_per_label,
                        rng,
                    )
                continue

            negative_seen += 1
            reservoir_add(
                negative_sample,
                (row_index, position),
                negative_seen,
                max_negative_positions,
                rng,
            )

    selected = sorted(set(boundary_positions).union(*inside_samples.values(), negative_sample))
    return selected, {
        "total_positions": total_positions,
        "selected_positions": len(selected),
        "boundary_positions": len(set(boundary_positions)),
        "inside_seen_by_label": dict(inside_seen),
        "inside_kept_by_label": {label: len(set(sample)) for label, sample in inside_samples.items()},
        "negative_seen": negative_seen,
        "negative_kept": len(set(negative_sample)),
        "max_inside_positions_per_label": max_inside_positions_per_label,
        "max_negative_positions": max_negative_positions,
    }


def reservoir_add(
    sample: list[tuple[int, int]],
    item: tuple[int, int],
    seen_count: int,
    cap: int,
    rng: random.Random,
) -> None:
    if cap < 0:
        sample.append(item)
        return
    if cap == 0:
        return
    if len(sample) < cap:
        sample.append(item)
        return
    replacement_index = rng.randrange(seen_count)
    if replacement_index < cap:
        sample[replacement_index] = item


def positive_labels(
    row_targets: dict[str, list[int]],
    labels: list[str],
    kinds: tuple[str, ...],
    position: int,
) -> list[str]:
    positive = []
    for label in labels:
        for kind in kinds:
            if row_targets[target_name(kind, label)][position]:
                positive.append(label)
                break
    return positive


def build_position_dataset(
    rows: list[dict[str, Any]],
    labels: list[str],
    examples: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]], list[int], list[int]]:
    feature_dicts: list[dict[str, Any]] = []
    targets: dict[str, list[int]] = {
        target_name(kind, label): [] for label in labels for kind in TARGET_KINDS
    }
    row_ids: list[int] = []
    positions: list[int] = []
    positions_by_row: dict[int, list[int]] = defaultdict(list)
    for row_index, position in examples:
        positions_by_row[row_index].append(position)

    for row_index in sorted(positions_by_row):
        row = rows[row_index]
        text = row["text"]
        row_targets = build_row_targets(row, labels)
        for position in sorted(positions_by_row[row_index]):
            feature_dicts.append(position_features(text, position))
            row_ids.append(row_index)
            positions.append(position)
            for name, values in row_targets.items():
                targets[name].append(values[position])

    return feature_dicts, targets, row_ids, positions


def target_name(kind: str, label: str) -> str:
    return f"{kind}:{label}"


def build_row_targets(row: dict[str, Any], labels: list[str]) -> dict[str, list[int]]:
    text = row["text"]
    length = len(text)
    targets = {target_name(kind, label): [0] * length for label in labels for kind in TARGET_KINDS}
    for span in row.get("spans", []):
        label = span["label"]
        if label not in labels:
            continue
        start = int(span["char_start"])
        end = int(span["char_end"])
        if start < 0 or end > length or start >= end:
            continue
        for position in range(start, end):
            targets[target_name("inside", label)][position] = 1
        if not span.get("left_open", False):
            targets[target_name("start", label)][start] = 1
        if not span.get("right_open", False):
            targets[target_name("end", label)][end - 1] = 1
        if span.get("left_open", False) and start == 0:
            targets[target_name("left_open", label)][0] = 1
        if span.get("right_open", False) and end == length:
            targets[target_name("right_open", label)][length - 1] = 1
    return targets


def position_features(text: str, position: int) -> dict[str, Any]:
    ch = text[position]
    features: dict[str, Any] = {
        "bias": 1.0,
        "ch": char_bucket(ch),
        "cat": unicodedata.category(ch),
        "is_alpha": float(ch.isalpha()),
        "is_upper": float(ch.isupper()),
        "is_lower": float(ch.islower()),
        "is_digit": float(ch.isdigit()),
        "is_space": float(ch.isspace()),
        "is_newline": float(ch in "\r\n"),
        "is_punct": float(unicodedata.category(ch).startswith("P")),
        "pos_frac": position / max(1, len(text) - 1),
        "from_start_log": math.log1p(position),
        "to_end_log": math.log1p(len(text) - 1 - position),
    }
    for offset in (-3, -2, -1, 1, 2, 3):
        neighbor = char_at(text, position + offset)
        prefix = f"n{offset:+d}"
        features[f"{prefix}_ch"] = char_bucket(neighbor)
        features[f"{prefix}_cat"] = "NONE" if neighbor == "" else unicodedata.category(neighbor)
        features[f"{prefix}_space"] = float(neighbor.isspace()) if neighbor else 1.0
        features[f"{prefix}_newline"] = float(neighbor in "\r\n") if neighbor else 1.0
    add_window_counts(features, text, position, 8, "r8")
    add_window_counts(features, text, position, 32, "r32")
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end == -1:
        line_end = len(text)
    features["line_pos_frac"] = (position - line_start) / max(1, line_end - line_start - 1)
    features["line_from_start_log"] = math.log1p(position - line_start)
    features["line_to_end_log"] = math.log1p(line_end - position)
    return features


def char_at(text: str, position: int) -> str:
    if position < 0 or position >= len(text):
        return ""
    return text[position]


def char_bucket(ch: str) -> str:
    if ch == "":
        return "<EDGE>"
    if ch == "\n":
        return "<LF>"
    if ch == "\r":
        return "<CR>"
    if ch == "\t":
        return "<TAB>"
    if ch.isspace():
        return "<SPACE>"
    if ch.isdigit():
        return "<DIGIT>"
    if ch.isalpha():
        return ch.lower() if ord(ch) < 128 else "<LETTER_NONASCII>"
    category = unicodedata.category(ch)
    if category.startswith("P"):
        return ch if ord(ch) < 128 else f"<PUNCT_{category}>"
    return f"<{category}>"


def add_window_counts(features: dict[str, Any], text: str, position: int, radius: int, prefix: str) -> None:
    start = max(0, position - radius)
    end = min(len(text), position + radius + 1)
    window = text[start:end]
    denom = max(1, len(window))
    counters = {
        "alpha": sum(ch.isalpha() for ch in window),
        "digit": sum(ch.isdigit() for ch in window),
        "space": sum(ch.isspace() for ch in window),
        "newline": sum(ch in "\r\n" for ch in window),
        "upper": sum(ch.isupper() for ch in window),
        "punct": sum(unicodedata.category(ch).startswith("P") for ch in window),
        "quote": sum(ch in "\"'“”‘’" for ch in window),
    }
    for name, count in counters.items():
        features[f"{prefix}_{name}_rate"] = count / denom


def fit_head(
    x_train: sparse.spmatrix,
    y_train: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    alpha: float,
    class_balance_cap: float,
) -> Any:
    unique = np.unique(y_train)
    if unique.size < 2:
        return DummyClassifier(strategy="constant", constant=int(unique[0])).fit(x_train, y_train)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    positive_weight = min(class_balance_cap, negatives / max(1, positives))
    sample_weight = np.where(y_train == 1, positive_weight, 1.0)
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=0.05,
        max_iter=max_iter,
        tol=1e-4,
        random_state=seed,
        n_jobs=1,
    )
    return model.fit(x_train, y_train, sample_weight=sample_weight)


def fit_sgd_heads(
    x_fit: sparse.spmatrix,
    x_calibration: sparse.spmatrix,
    x_valid: sparse.spmatrix,
    fit_target_arrays: dict[str, np.ndarray],
    head_names: list[str],
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    models = {}
    calibration_scores = np.zeros((x_calibration.shape[0], len(head_names)), dtype=np.float32)
    valid_scores = np.zeros((x_valid.shape[0], len(head_names)), dtype=np.float32)
    for column, name in enumerate(head_names):
        print(f"fitting sgd head {column + 1}/{len(head_names)} {name}", flush=True)
        model = fit_head(
            x_fit,
            fit_target_arrays[name],
            seed=args.seed,
            max_iter=args.max_iter,
            alpha=args.alpha,
            class_balance_cap=args.class_balance_cap,
        )
        models[name] = model
        calibration_scores[:, column] = predict_scores(model, x_calibration)
        valid_scores[:, column] = predict_scores(model, x_valid)
    return models, calibration_scores, valid_scores


def fit_extra_trees_multi_output(
    x_fit: sparse.spmatrix,
    y_fit: np.ndarray,
    *,
    args: argparse.Namespace,
) -> ExtraTreesClassifier:
    max_depth = None if args.max_depth <= 0 else args.max_depth
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    return model.fit(x_fit, y_fit)


def predict_multi_output_scores(model: ExtraTreesClassifier, x: sparse.spmatrix) -> np.ndarray:
    probabilities_by_output = model.predict_proba(x)
    scores = np.zeros((x.shape[0], len(probabilities_by_output)), dtype=np.float32)
    for column, probabilities in enumerate(probabilities_by_output):
        classes = list(model.classes_[column])
        if 1 in classes:
            scores[:, column] = probabilities[:, classes.index(1)]
        elif len(classes) == 1:
            scores[:, column] = float(classes[0])
    return scores


def predict_scores(model: Any, x: sparse.spmatrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.shape[1] == 1:
            return np.full(x.shape[0], float(model.classes_[0]), dtype=np.float32)
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
        return probabilities[:, positive_index]
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def best_threshold(scores: np.ndarray, labels: np.ndarray, *, threshold_steps: int) -> float:
    if labels.sum() == 0:
        return 1.0
    best = (0.5, -1.0)
    if threshold_steps < 2:
        raise SystemExit("--threshold-steps must be at least 2")
    for threshold in np.linspace(0.01, 0.99, threshold_steps):
        predictions = (scores >= threshold).astype(np.uint8)
        _precision, _recall, f1, _support = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        if f1 > best[1]:
            best = (float(threshold), float(f1))
    return best[0]


def model_config(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "threshold_source": "calibration_split",
        "threshold_steps": args.threshold_steps,
    }
    if args.model_kind == "extra-trees":
        return {
            **common,
            "n_estimators": args.n_estimators,
            "max_depth": None if args.max_depth <= 0 else args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "class_weight": "balanced",
            "n_jobs": args.n_jobs,
        }
    return {
        **common,
        "max_iter": args.max_iter,
        "alpha": args.alpha,
        "class_balance_cap": args.class_balance_cap,
        "loss": "log_loss",
        "penalty": "elasticnet",
        "l1_ratio": 0.05,
    }


def dataset_stats(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "focus_rows": dict(Counter(row.get("focus_label") for row in rows)),
        "positive_focus_rows": dict(Counter(row.get("focus_label") for row in rows if row.get("spans"))),
        "span_counts": dict(Counter(span["label"] for row in rows for span in row.get("spans", []))),
        "open_edges": {},
        "chars": sum(len(row["text"]) for row in rows),
    }
    edge_counter = Counter()
    for row in rows:
        for span in row.get("spans", []):
            if span.get("left_open"):
                edge_counter[f"{span['label']}:left_open"] += 1
            if span.get("right_open"):
                edge_counter[f"{span['label']}:right_open"] += 1
    stats["open_edges"] = dict(edge_counter)
    stats["labels"] = labels
    return stats


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def group_metrics(head_reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in head_reports.values():
        groups[report["kind"]].append(report)
    grouped = {}
    for kind, reports in groups.items():
        viable = [report for report in reports if report["valid_positive"] > 0]
        grouped[kind] = {
            "heads": len(reports),
            "viable_heads": len(viable),
            "macro_precision": float(np.mean([report["precision"] for report in viable])) if viable else 0.0,
            "macro_recall": float(np.mean([report["recall"] for report in viable])) if viable else 0.0,
            "macro_f1": float(np.mean([report["f1"] for report in viable])) if viable else 0.0,
            "valid_positive": int(sum(report["valid_positive"] for report in reports)),
        }
    return grouped


def print_summary(report: dict[str, Any]) -> None:
    print("trained streaming multi-label model")
    print(
        f"model_kind={report['model_kind']} rows={report['rows']} "
        f"fit_rows={report['fit_rows']} calibration_rows={report['calibration_rows']} "
        f"valid_rows={report['valid_rows']} positions={report['positions']} "
        f"feature_dim={report['feature_dim']}"
    )
    print(
        f"fit_positions={report['fit_positions']} "
        f"calibration_positions={report['calibration_positions']} "
        f"valid_positions={report['valid_positions']} "
        f"fit_total_positions={report['sampling']['fit']['total_positions']} "
        f"calibration_total_positions={report['sampling']['calibration']['total_positions']} "
        f"valid_total_positions={report['sampling']['valid']['total_positions']}"
    )
    print(
        f"model_seconds={report['model_seconds']:.3f} "
        f"training_seconds={report['training_seconds']:.3f}"
    )
    for kind, metrics in report["group_metrics"].items():
        print(
            f"{kind}: viable_heads={metrics['viable_heads']}/{metrics['heads']} "
            f"valid_positive={metrics['valid_positive']} "
            f"macro_precision={metrics['macro_precision']:.4f} "
            f"macro_recall={metrics['macro_recall']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
