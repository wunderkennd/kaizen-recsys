"""Pure-Python reader/writer for the Rust FEAS model format.

Mirrors src/serialization.rs. bincode 1.3 default config: little-endian,
fixed-width ints, u64 length prefixes, no field names. usize serializes as u64.
version is u32 (4 bytes); all usize fields are u64 (8 bytes) on 64-bit targets.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_F64 = struct.Struct("<d")

_MAGIC = b"FEAS"
_FORMAT_VERSION = 2  # version Spark authors (EASE core; no native transform schema)
# Versions the reader accepts. v3 adds a trailing transformation_schema used only
# by native predict_raw; Spark reads the shared EASE core and carries v3's trailing
# bytes verbatim (see FeaseArtifact.transformation_schema_raw).
_SUPPORTED_VERSIONS = (1, 2, 3)


def _read_exact(buf, n: int) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


def _write_u32(buf, v: int) -> None:
    buf.write(_U32.pack(v))


def _read_u32(buf) -> int:
    return _U32.unpack(_read_exact(buf, 4))[0]


def _write_u64(buf, v: int) -> None:
    buf.write(_U64.pack(v))


def _read_u64(buf) -> int:
    return _U64.unpack(_read_exact(buf, 8))[0]


def _write_f64(buf, v: float) -> None:
    buf.write(_F64.pack(v))


def _read_f64(buf) -> float:
    return _F64.unpack(_read_exact(buf, 8))[0]


def _write_string(buf, s: str) -> None:
    raw = s.encode("utf-8")
    _write_u64(buf, len(raw))
    buf.write(raw)


def _read_string(buf) -> str:
    n = _read_u64(buf)
    return _read_exact(buf, n).decode("utf-8")


def _write_vec_f64(buf, xs) -> None:
    _write_u64(buf, len(xs))
    for x in xs:
        _write_f64(buf, float(x))


def _read_vec_f64(buf) -> list:
    n = _read_u64(buf)
    return [_read_f64(buf) for _ in range(n)]


def _write_vec_string(buf, xs) -> None:
    _write_u64(buf, len(xs))
    for s in xs:
        _write_string(buf, s)


def _read_vec_string(buf) -> list:
    n = _read_u64(buf)
    return [_read_string(buf) for _ in range(n)]


def _write_vec_pair_string_usize(buf, pairs) -> None:
    _write_u64(buf, len(pairs))
    for s, i in pairs:
        _write_string(buf, s)
        _write_u64(buf, i)


def _read_vec_pair_string_usize(buf) -> list:
    n = _read_u64(buf)
    return [(_read_string(buf), _read_u64(buf)) for _ in range(n)]


@dataclass
class WeightingConfig:
    # event_weights kept as an insertion-ordered dict so decode->encode is byte-stable.
    event_weights: Optional[dict] = None
    decay_rate: float = 0.0
    ips_alpha: float = 0.0
    sparsity_threshold: float = 0.0


@dataclass
class FeaseArtifact:
    version: int
    s_nrows: int
    s_ncols: int
    s_data: np.ndarray  # 2-D Fortran-order (s_nrows x s_ncols), or flat handled in write
    num_items: int
    num_user_features: int
    num_item_features: int
    alpha: float
    beta: float
    lambda_: float
    meta_weight: float
    user_to_idx: list
    idx_to_user: list
    item_to_idx: list
    idx_to_item: list
    user_feature_to_idx: list
    idx_to_user_feature: list
    item_feature_to_idx: list
    idx_to_item_feature: list
    weighting_config: Optional[WeightingConfig] = None
    # FEAS v3 appends `transformation_schema: Option<FeatureTransformationSchema>`
    # (native predict_raw only — irrelevant to Spark EASE scoring). We carry the
    # trailing bytes verbatim so a loaded v3 model re-saves byte-for-byte without
    # having to parse the schema. Empty for v1/v2.
    transformation_schema_raw: bytes = b""


def _write_map_string_f64(buf, m: dict) -> None:
    # bincode HashMap<String,f64> = u64 count then entries in iteration order.
    _write_u64(buf, len(m))
    for k, v in m.items():
        _write_string(buf, k)
        _write_f64(buf, float(v))


def _read_map_string_f64(buf) -> dict:
    n = _read_u64(buf)
    out = {}
    for _ in range(n):
        k = _read_string(buf)
        out[k] = _read_f64(buf)
    return out


def _write_weighting(buf, wc: Optional[WeightingConfig]) -> None:
    # Field is Option<WeightingConfig>: 1 tag byte then body if present.
    if wc is None:
        buf.write(b"\x00")
        return
    buf.write(b"\x01")
    # event_weights: Option<HashMap<String,f64>>
    if wc.event_weights is None:
        buf.write(b"\x00")
    else:
        buf.write(b"\x01")
        _write_map_string_f64(buf, wc.event_weights)
    _write_f64(buf, wc.decay_rate)
    _write_f64(buf, wc.ips_alpha)
    _write_f64(buf, wc.sparsity_threshold)


def _read_weighting(buf) -> Optional[WeightingConfig]:
    tag = _read_exact(buf, 1)
    if tag == b"\x00":
        return None
    ew_tag = _read_exact(buf, 1)
    event_weights = _read_map_string_f64(buf) if ew_tag == b"\x01" else None
    decay_rate = _read_f64(buf)
    ips_alpha = _read_f64(buf)
    sparsity_threshold = _read_f64(buf)
    return WeightingConfig(event_weights, decay_rate, ips_alpha, sparsity_threshold)


def _s_data_flat_colmajor(s_data, nrows, ncols) -> list:
    arr = np.asarray(s_data, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.asfortranarray(arr).reshape(-1, order="F")
    return arr.tolist()


def write_feas(artifact: FeaseArtifact, path: str) -> None:
    buf = io.BytesIO()
    a = artifact
    _write_u32(buf, a.version)
    _write_u64(buf, a.s_nrows)
    _write_u64(buf, a.s_ncols)
    _write_vec_f64(buf, _s_data_flat_colmajor(a.s_data, a.s_nrows, a.s_ncols))
    _write_u64(buf, a.num_items)
    _write_u64(buf, a.num_user_features)
    _write_u64(buf, a.num_item_features)
    _write_f64(buf, a.alpha)
    _write_f64(buf, a.beta)
    _write_f64(buf, a.lambda_)
    _write_f64(buf, a.meta_weight)
    _write_vec_pair_string_usize(buf, a.user_to_idx)
    _write_vec_string(buf, a.idx_to_user)
    _write_vec_pair_string_usize(buf, a.item_to_idx)
    _write_vec_string(buf, a.idx_to_item)
    _write_vec_pair_string_usize(buf, a.user_feature_to_idx)
    _write_vec_string(buf, a.idx_to_user_feature)
    _write_vec_pair_string_usize(buf, a.item_feature_to_idx)
    _write_vec_string(buf, a.idx_to_item_feature)
    if a.version >= 2:
        _write_weighting(buf, a.weighting_config)
    # v3+ trailing transformation_schema, carried verbatim (b"" for v1/v2).
    buf.write(a.transformation_schema_raw)
    with open(path, "wb") as fh:
        fh.write(_MAGIC)
        fh.write(buf.getvalue())


def read_feas(path: str) -> FeaseArtifact:
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] != _MAGIC:
        raise ValueError(f"not a FEAS file: magic={blob[:4]!r}")
    buf = io.BytesIO(blob[4:])
    version = _read_u32(buf)
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported FEAS version: {version}")
    s_nrows = _read_u64(buf)
    s_ncols = _read_u64(buf)
    s_flat = _read_vec_f64(buf)
    num_items = _read_u64(buf)
    num_user_features = _read_u64(buf)
    num_item_features = _read_u64(buf)
    alpha = _read_f64(buf)
    beta = _read_f64(buf)
    lambda_ = _read_f64(buf)
    meta_weight = _read_f64(buf)
    user_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_user = _read_vec_string(buf)
    item_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_item = _read_vec_string(buf)
    user_feature_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_user_feature = _read_vec_string(buf)
    item_feature_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_item_feature = _read_vec_string(buf)
    weighting_config = _read_weighting(buf) if version >= 2 else None
    transformation_schema_raw = buf.read()  # v3+ trailing bytes, carried verbatim

    s_data = np.reshape(np.asarray(s_flat, dtype=np.float64),
                        (s_nrows, s_ncols), order="F")
    return FeaseArtifact(
        version=version, s_nrows=s_nrows, s_ncols=s_ncols, s_data=np.asfortranarray(s_data),
        num_items=num_items, num_user_features=num_user_features,
        num_item_features=num_item_features, alpha=alpha, beta=beta, lambda_=lambda_,
        meta_weight=meta_weight, user_to_idx=user_to_idx, idx_to_user=idx_to_user,
        item_to_idx=item_to_idx, idx_to_item=idx_to_item,
        user_feature_to_idx=user_feature_to_idx, idx_to_user_feature=idx_to_user_feature,
        item_feature_to_idx=item_feature_to_idx, idx_to_item_feature=idx_to_item_feature,
        weighting_config=weighting_config,
        transformation_schema_raw=transformation_schema_raw,
    )
