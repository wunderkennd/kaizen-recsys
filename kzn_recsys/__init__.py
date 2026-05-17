"""kzn_recsys — Python wrapper for the Rust FEASE recommender."""

from kzn_recsys._native import (
    FeaseModel,
    FeaseRegistry,
    build_and_train,
    coverage,
    grid_search_py as grid_search,
    hit_rate_at_k,
    load_model,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    random_search_py as random_search,
    recall_at_k,
    validate_data,
    random_split,
    temporal_split,
    leave_k_out_split,
)
from kzn_recsys.fease_wrapper import (
    SplitResult,
    leave_k_out_split_safe,
    random_split_safe,
    temporal_split_safe,
)
from kzn_recsys.schemas import EngagementSchema, MetadataSchema

# SASRec is only present when the extension is built with the `ml-models`
# Cargo feature (default-off; EASE-only wheels omit it and burn).
try:  # pragma: no cover - import guard, exercised by build matrix
    from kzn_recsys._native import (  # noqa: F401
        SASRecModel,
        build_and_train_sasrec,
        load_sasrec_model,
    )

    _HAS_ML_MODELS = True
except ImportError:
    _HAS_ML_MODELS = False

__all__ = [
    "FeaseModel",
    "FeaseRegistry",
    "build_and_train",
    "coverage",
    "grid_search",
    "hit_rate_at_k",
    "load_model",
    "mean_average_precision",
    "ndcg_at_k",
    "precision_at_k",
    "random_search",
    "recall_at_k",
    "validate_data",
    "random_split",
    "temporal_split",
    "leave_k_out_split",
    "SplitResult",
    "random_split_safe",
    "temporal_split_safe",
    "leave_k_out_split_safe",
    "EngagementSchema",
    "MetadataSchema",
]

if _HAS_ML_MODELS:
    __all__ += ["SASRecModel", "build_and_train_sasrec", "load_sasrec_model"]
