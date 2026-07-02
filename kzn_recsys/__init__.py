"""kzn_recsys — Python wrapper for the Rust FEASE recommender."""

from kzn_recsys._native import (
    FeaseModel,
    FeatureTransformationSchema,
    ModelRegistry,
    NumericalBucketConfig,
    build_and_train,
    coverage,
    grid_search_ease,
    grid_search_py as grid_search,
    hit_rate_at_k,
    load_model,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    random_search_ease,
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

# SASRec and Two-Tower are only present when the extension is built with
# the `ml-models` Cargo feature (default-off; EASE-only wheels omit them
# and burn).
try:  # pragma: no cover - import guard, exercised by build matrix
    from kzn_recsys._native import (  # noqa: F401
        SASRecModel,
        TwoTowerModel,
        build_and_train_sasrec,
        build_and_train_two_tower,
        grid_search_sasrec,
        grid_search_two_tower,
        load_sasrec_model,
        load_two_tower_model,
        random_search_sasrec,
        random_search_two_tower,
    )

    _HAS_ML_MODELS = True
except ImportError:
    _HAS_ML_MODELS = False

__all__ = [
    "FeaseModel",
    "FeatureTransformationSchema",
    "ModelRegistry",
    "NumericalBucketConfig",
    "build_and_train",
    "coverage",
    "grid_search",
    "grid_search_ease",
    "hit_rate_at_k",
    "load_model",
    "mean_average_precision",
    "ndcg_at_k",
    "precision_at_k",
    "random_search",
    "random_search_ease",
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
    __all__ += [
        "SASRecModel",
        "TwoTowerModel",
        "build_and_train_sasrec",
        "build_and_train_two_tower",
        "grid_search_sasrec",
        "grid_search_two_tower",
        "load_sasrec_model",
        "load_two_tower_model",
        "random_search_sasrec",
        "random_search_two_tower",
    ]

try:  # pragma: no cover - import guard, exercised by build matrix
    from kzn_recsys.onnx_export import export_onnx  # noqa: F401

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

if _HAS_ONNX:
    __all__ += ["export_onnx"]
