# Pure-Python wheel (`kzn_recsys.spark`)

The main wheel is built by maturin and contains the compiled Rust extension.
For environments where that native wheel cannot be installed, build this
pure-Python distribution, which ships only `kzn_recsys.spark` (NumPy/SciPy/
PySpark EASE) plus the native-optional `kzn_recsys/__init__.py`.

## Build

```bash
cd packaging/pure-python
python -m build --wheel
# -> dist/kzn_recsys_spark-0.1.0-py3-none-any.whl
```

## Install (restricted environment)

```bash
pip install kzn_recsys_spark-0.1.0-py3-none-any.whl
python -c "from kzn_recsys.spark import build_and_train; print('ok')"
```

The two distributions are intentionally separate: `kzn_recsys` (maturin,
native) and `kzn_recsys_spark` (pure-Python). They share the `kzn_recsys`
import namespace; do not install both into the same environment.
