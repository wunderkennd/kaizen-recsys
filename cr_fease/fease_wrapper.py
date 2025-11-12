# fease_wrapper.py (optional high-level API)
import polars as pl
import rust_fease_recommender as fease
from schemas import EngagementSchema, MetadataSchema

def build_and_train_safe(
        engagement_path: str,
        metadata_path: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_: float = 100.0,
        validate: bool = True
) -> fease.FeaseModel:
    """
    Builds and trains a FEASE model with optional Python-side validation.

    Args:
        engagement_path: Path to engagement parquet
        metadata_path: Path to metadata parquet
        alpha: Item feature weight
        beta: User feature weight
        lambda_: L2 regularization
        validate: If True, validates schemas before passing to Rust

    Returns:
        Trained FeaseModel
    """
    if validate:
        # Load and validate in Python for better error messages
        print("Validating engagement data...")
        df_eng = pl.read_parquet(engagement_path)
        df_eng = EngagementSchema.validate_and_cast(df_eng)

        print("Validating metadata...")
        df_meta = pl.read_parquet(metadata_path)
        df_meta = MetadataSchema.validate_and_cast(df_meta)

        # Write validated data to temp files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as eng_tmp:
            df_eng.write_parquet(eng_tmp.name)
            eng_path = eng_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as meta_tmp:
            df_meta.write_parquet(meta_tmp.name)
            meta_path = meta_tmp.name

        try:
            return fease.build_and_train(eng_path, meta_path, alpha, beta, lambda_)
        finally:
            import os
            os.unlink(eng_path)
            os.unlink(meta_path)
    else:
        # Skip validation, let Rust handle it
        return fease.build_and_train(engagement_path, metadata_path, alpha, beta, lambda_)