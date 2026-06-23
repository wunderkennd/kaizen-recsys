"""Pure-Python reader/writer for the Rust FEAS model format.

Mirrors src/serialization.rs. bincode 1.3 default config: little-endian,
fixed-width ints, u64 length prefixes, no field names. usize serializes as u64.
"""
from __future__ import annotations

import struct

_U64 = struct.Struct("<Q")
_F64 = struct.Struct("<d")


def _read_exact(buf, n: int) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


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
