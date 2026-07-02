import io

from kzn_recsys.spark import feas_codec as fc


def test_u64_roundtrip():
    buf = io.BytesIO()
    fc._write_u64(buf, 0)
    fc._write_u64(buf, 1)
    fc._write_u64(buf, 2**40 + 7)
    buf.seek(0)
    assert fc._read_u64(buf) == 0
    assert fc._read_u64(buf) == 1
    assert fc._read_u64(buf) == 2**40 + 7


def test_u64_is_little_endian_8_bytes():
    buf = io.BytesIO()
    fc._write_u64(buf, 1)
    assert buf.getvalue() == b"\x01\x00\x00\x00\x00\x00\x00\x00"


def test_f64_roundtrip():
    buf = io.BytesIO()
    for v in (0.0, -1.5, 3.141592653589793):
        fc._write_f64(buf, v)
    buf.seek(0)
    assert fc._read_f64(buf) == 0.0
    assert fc._read_f64(buf) == -1.5
    assert fc._read_f64(buf) == 3.141592653589793


def test_string_roundtrip_length_prefixed():
    buf = io.BytesIO()
    fc._write_string(buf, "héllo")  # multi-byte UTF-8
    buf.seek(0)
    # u64 length prefix == UTF-8 byte length (6), then bytes
    assert fc._read_string(buf) == "héllo"


def test_vec_f64_roundtrip():
    buf = io.BytesIO()
    fc._write_vec_f64(buf, [1.0, 2.0, 3.0])
    buf.seek(0)
    assert fc._read_vec_f64(buf) == [1.0, 2.0, 3.0]


def test_vec_string_and_pairs_roundtrip():
    buf = io.BytesIO()
    fc._write_vec_string(buf, ["a", "bb"])
    fc._write_vec_pair_string_usize(buf, [("a", 0), ("bb", 1)])
    buf.seek(0)
    assert fc._read_vec_string(buf) == ["a", "bb"]
    assert fc._read_vec_pair_string_usize(buf) == [("a", 0), ("bb", 1)]


import numpy as np

from kzn_recsys.spark.feas_codec import FeaseArtifact, WeightingConfig, write_feas, read_feas


def _toy_artifact(weighting=None, version=2):
    return FeaseArtifact(
        version=version,
        s_nrows=2,
        s_ncols=2,
        # column-major flat: column 0 then column 1
        s_data=np.asfortranarray(np.array([[0.0, 0.5], [0.7, 0.0]])),
        num_items=2,
        num_user_features=0,
        num_item_features=0,
        alpha=1.0, beta=1.0, lambda_=150.0, meta_weight=0.0,
        user_to_idx=[("u0", 0)], idx_to_user=["u0"],
        item_to_idx=[("i0", 0), ("i1", 1)], idx_to_item=["i0", "i1"],
        user_feature_to_idx=[], idx_to_user_feature=[],
        item_feature_to_idx=[], idx_to_item_feature=[],
        weighting_config=weighting,
    )


def test_write_then_read_roundtrip_v2(tmp_path):
    art = _toy_artifact()
    path = tmp_path / "m.fease"
    write_feas(art, str(path))
    back = read_feas(str(path))
    assert back.version == 2
    assert back.s_nrows == 2 and back.s_ncols == 2
    assert np.allclose(back.s_data, art.s_data)
    assert back.s_data.flags["F_CONTIGUOUS"]
    assert back.item_to_idx == [("i0", 0), ("i1", 1)]
    assert back.weighting_config is None


def test_magic_bytes_present(tmp_path):
    path = tmp_path / "m.fease"
    write_feas(_toy_artifact(), str(path))
    with open(path, "rb") as fh:
        assert fh.read(4) == b"FEAS"


def test_weighting_config_roundtrip(tmp_path):
    wc = WeightingConfig(event_weights={"click": 1.0, "purchase": 5.0},
                         decay_rate=0.01, ips_alpha=0.5, sparsity_threshold=0.0)
    path = tmp_path / "m.fease"
    write_feas(_toy_artifact(weighting=wc), str(path))
    back = read_feas(str(path))
    assert back.weighting_config.decay_rate == 0.01
    assert back.weighting_config.ips_alpha == 0.5
    assert back.weighting_config.event_weights == {"click": 1.0, "purchase": 5.0}


def test_byte_exact_reencode(tmp_path):
    """Decode-then-encode must reproduce identical bytes (preserves map order)."""
    wc = WeightingConfig(event_weights={"a": 1.0, "b": 2.0},
                         decay_rate=0.0, ips_alpha=0.0, sparsity_threshold=0.0)
    p1 = tmp_path / "a.fease"
    write_feas(_toy_artifact(weighting=wc), str(p1))
    original = p1.read_bytes()
    back = read_feas(str(p1))
    p2 = tmp_path / "b.fease"
    write_feas(back, str(p2))
    assert p2.read_bytes() == original


def test_v1_has_no_weighting_config(tmp_path):
    art = _toy_artifact(version=1)
    path = tmp_path / "m.fease"
    write_feas(art, str(path))
    back = read_feas(str(path))
    assert back.version == 1
    assert back.weighting_config is None
