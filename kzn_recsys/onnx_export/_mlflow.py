"""MLflow pyfunc wrapper that serves the ONNX graph by GUID (spec §7)."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd


class FeaseOnnxPyfunc(mlflow.pyfunc.PythonModel):
    """Maps GUIDs↔indices around the numeric ONNX graph. No kzn_recsys import."""

    def load_context(self, context):
        import onnxruntime as ort

        self._vocab = json.loads(Path(context.artifacts["vocab"]).read_text())
        self._sess = ort.InferenceSession(context.artifacts["onnx"])
        self._M = self._vocab["num_items"]
        self._K = self._vocab["num_user_features"]
        self._idx_to_guid = self._vocab["item_index_to_guid"]
        self._guid_to_idx = {g: i for i, g in enumerate(self._idx_to_guid)}
        self._feat_to_idx = self._vocab["feature_name_to_index"]
        self._default_rp = self._vocab["repeat_policy"]["default_penalty"]
        self._default_k = self._vocab["top_k_default"]
        # Cache output-name positions once to avoid per-row list rebuild.
        self._output_names = [o.name for o in self._sess.get_outputs()]
        self._top_idx_pos = self._output_names.index("top_indices")
        self._top_scr_pos = self._output_names.index("top_scores")

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        rows = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        frames = []
        for r, row in enumerate(rows):
            frames.append(self._predict_one(r, row))
        if not frames:
            return pd.DataFrame(columns=["user_row", "rank", "item_guid", "score"])
        return pd.concat(frames, ignore_index=True)

    def _predict_one(self, row_id, row):
        M, K = self._M, self._K
        inter = np.zeros((1, M), np.float32)
        seen = np.zeros((1, M), np.float32)
        for guid, val in (row.get("interactions") or {}).items():
            idx = self._guid_to_idx.get(guid)
            if idx is None:
                continue  # unknown GUID → skip (catalogs drift)
            inter[0, idx] = float(val)
            seen[0, idx] = 1.0  # key-based "seen", matching Rust semantics
        feat = np.zeros((1, K), np.float32)
        for name, val in (row.get("features") or {}).items():
            idx = self._feat_to_idx.get(name)
            if idx is not None:
                feat[0, idx] = float(val)
        mask = np.ones((1, M), np.float32)
        for guid in (row.get("exclude") or []):
            idx = self._guid_to_idx.get(guid)
            if idx is not None:
                mask[0, idx] = 0.0
        rp = float(row["repeat_penalty"]) if row.get("repeat_penalty") is not None else self._default_rp
        k = int(row["top_k"]) if row.get("top_k") is not None else self._default_k

        out = self._sess.run(
            None,
            {
                "interactions": inter,
                "features": feat,
                "mask": mask,
                "seen": seen,
                "repeat_penalty": np.array([[rp]], np.float32),
                "k": np.array([k], np.int64),
            },
        )
        top_idx = out[self._top_idx_pos][0]
        top_scr = out[self._top_scr_pos][0]
        return pd.DataFrame(
            {
                "user_row": row_id,
                "rank": np.arange(len(top_idx)),
                "item_guid": [self._idx_to_guid[i] for i in top_idx],
                "score": top_scr,
            }
        )


def build_mlflow(onnx_path: Path, vocab_path: Path, out_dir: Path) -> Path:
    import mlflow.pyfunc

    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    mlflow.pyfunc.save_model(
        path=str(out_dir),
        python_model=FeaseOnnxPyfunc(),
        artifacts={"onnx": str(onnx_path), "vocab": str(vocab_path)},
        code_paths=[str(Path(__file__))],
        pip_requirements=["onnxruntime>=1.18", "numpy>=1.24", "pandas>=1.5"],
    )
    return out_dir
