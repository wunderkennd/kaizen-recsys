"""High-level Python wrapper with optional schema validation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import polars as pl
import rust_fease_recommender as fease
from cr_fease.schemas import EngagementSchema, MetadataSchema


@dataclass(frozen=True)
class SplitResult:
    """Outcome of a train/test split.

    Wraps the 4-tuple `(train_interactions, test_interactions, train_users,
    test_users)` returned by the Rust split functions and pairs it with the
    output paths the caller actually needs to feed back into `build_and_train`
    and `evaluate`. Iterates as the same 4-tuple of counts so callers that
    were already destructuring the Rust return value keep working.
    """

    train_path: str
    test_path: str
    train_interactions: int
    test_interactions: int
    train_users: int
    test_users: int

    def __iter__(self) -> Iterator[int]:
        # Backward-compat with the Rust 4-tuple unpacking style.
        yield self.train_interactions
        yield self.test_interactions
        yield self.train_users
        yield self.test_users


def _resolve_split_paths(
    train_output: Optional[str],
    test_output: Optional[str],
    output_dir: Optional[str],
) -> tuple[str, str]:
    """Pick output parquet paths, allocating a temp dir when the caller didn't.

    Either both `train_output` and `test_output` are provided (caller-managed)
    or both are None (we allocate). Mixing the two is a bug, not an
    ergonomic shortcut, so we reject it loudly.
    """
    if (train_output is None) ^ (test_output is None):
        raise ValueError(
            "train_output and test_output must both be provided or both be None"
        )
    if train_output is not None and test_output is not None:
        return train_output, test_output

    workspace = output_dir or tempfile.mkdtemp(prefix="fease_split_")
    os.makedirs(workspace, exist_ok=True)
    return (
        os.path.join(workspace, "train.parquet"),
        os.path.join(workspace, "test.parquet"),
    )


def random_split_safe(
    interactions_path: str,
    train_output: Optional[str] = None,
    test_output: Optional[str] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> SplitResult:
    """Random train/test split with named-attribute return.

    If `train_output`/`test_output` are omitted, a workspace is allocated under
    `output_dir` (or a fresh `tempfile.mkdtemp` when that is also None) and
    paths are returned in the result. The caller owns the workspace lifecycle.
    """
    train_out, test_out = _resolve_split_paths(train_output, test_output, output_dir)
    train_n, test_n, train_u, test_u = fease.random_split(
        interactions_path,
        train_out,
        test_out,
        test_ratio=test_ratio,
        seed=seed,
    )
    return SplitResult(train_out, test_out, train_n, test_n, train_u, test_u)


def temporal_split_safe(
    interactions_path: str,
    days_ago_cutoff: float,
    train_output: Optional[str] = None,
    test_output: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> SplitResult:
    """Temporal split: interactions with `days_ago <= cutoff` go to test.

    Same path-allocation behavior as `random_split_safe`.
    """
    train_out, test_out = _resolve_split_paths(train_output, test_output, output_dir)
    train_n, test_n, train_u, test_u = fease.temporal_split(
        interactions_path,
        train_out,
        test_out,
        days_ago_cutoff,
    )
    return SplitResult(train_out, test_out, train_n, test_n, train_u, test_u)


def leave_k_out_split_safe(
    interactions_path: str,
    train_output: Optional[str] = None,
    test_output: Optional[str] = None,
    k: int = 1,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> SplitResult:
    """Leave-K-out split: hold out exactly `k` random interactions per user.

    Same path-allocation behavior as `random_split_safe`.
    """
    train_out, test_out = _resolve_split_paths(train_output, test_output, output_dir)
    train_n, test_n, train_u, test_u = fease.leave_k_out_split(
        interactions_path,
        train_out,
        test_out,
        k=k,
        seed=seed,
    )
    return SplitResult(train_out, test_out, train_n, test_n, train_u, test_u)


def build_and_train_safe(
    interactions_path: str,
    user_features_path: str,
    item_features_path: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 100.0,
    meta_weight: float = 0.0,
    decay_rate: float = 0.0,
    ips_alpha: float = 0.0,
    sparsity_threshold: float = 0.0,
    event_weights: Optional[Dict[str, float]] = None,
) -> fease.FeaseModel:
    """
    Builds and trains a FEASE model from three long-format Parquet/CSV files.

    Args:
        interactions_path: Path to interactions file (user_id, item_id, value).
        user_features_path: Path to user features file (user_id, feature_name, value).
        item_features_path: Path to item features file (item_id, feature_name, value).
        alpha: Weight for item features in the Gram matrix.
        beta: Weight for user features in the Gram matrix.
        lambda_: L2 regularization term.
        meta_weight: Weight for metadata rows (0.0 = equal weighting).
        decay_rate: Exponential temporal decay rate (0.0 = no decay).
            Requires ``days_ago`` column in the interactions file.
        ips_alpha: Inverse propensity scoring strength (0.0 = disabled).
        sparsity_threshold: Prune S-matrix entries below this value (0.0 = no pruning).
        event_weights: Dict mapping event type to weight multiplier (None = disabled).
            Requires ``event_type`` column in the interactions file.

    Returns:
        Trained FeaseModel ready for predictions.
    """
    kwargs: dict = {}
    if decay_rate > 0.0:
        kwargs["decay_rate"] = decay_rate
    if ips_alpha > 0.0:
        kwargs["ips_alpha"] = ips_alpha
    if sparsity_threshold > 0.0:
        kwargs["sparsity_threshold"] = sparsity_threshold
    if event_weights is not None:
        kwargs["event_weights"] = event_weights

    return fease.build_and_train(
        interactions_path=interactions_path,
        user_features_path=user_features_path,
        item_features_path=item_features_path,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        meta_weight=meta_weight,
        **kwargs,
    )
