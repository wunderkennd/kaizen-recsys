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
