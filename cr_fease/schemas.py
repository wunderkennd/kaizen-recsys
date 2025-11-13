# schemas.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import polars as pl

class SubscriptionPlan(str, Enum):
    FREE = "Free"
    PREMIUM = "Premium"
    TRIAL = "Free Trial"

class DeviceType(str, Enum):
    MOBILE = "Mobile"
    WEB = "Web"
    CONSOLE = "Console"
    TV = "TV"

class EngagementSchema(BaseModel):
    """Schema for engagement data validation"""

    # Define expected columns and types
    REQUIRED_COLUMNS = {
        "anonymous_id": pl.String,
        "view_media_id": pl.String,
        "view_seconds_watched": pl.Float64,
        "view_context_device_type": pl.String,
        "view_subscription_plan": pl.String,
        "view_country_code_view": pl.String,
        "account_country_code_account": pl.String,
        "account_tenure_days": pl.Int32,  # Enforce Int32
        "region_major_account": pl.String,
        "subscription_status": pl.String,
    }

    @classmethod
    def validate_and_cast(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        Validates and casts a DataFrame to the expected schemas.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with validated and casted columns

        Raises:
            ValueError: If required columns are missing or casting fails
        """
        # Check for missing columns
        missing = set(cls.REQUIRED_COLUMNS.keys()) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Cast to expected types
        try:
            df = df.select([
                pl.col(col).cast(dtype).alias(col)
                for col, dtype in cls.REQUIRED_COLUMNS.items()
            ])
        except Exception as e:
            raise ValueError(f"Failed to cast engagement data: {e}")

        return df

class MetadataSchema(BaseModel):
    """Schema for metadata validation"""

    REQUIRED_COLUMNS = {
        "media_guid": pl.String,
        "media_type": pl.String,
        "media_audio_language": pl.String,
        "media_hardsub_language": pl.String,
        "media_genres": pl.String,
        "media_tags": pl.String,
        "media_series_title": pl.String,
        "media_publisher_name": pl.String,
        "airtable_primary_genre": pl.String,
        "airtable_secondary_genres": pl.String,
        "airtable_japanese_audience": pl.String,
        "airtable_brand_grade_from_ca_data": pl.String,
        "airtable_original_release_year": pl.String,
    }

    @classmethod
    def validate_and_cast(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Validates and casts metadata DataFrame"""
        missing = set(cls.REQUIRED_COLUMNS.keys()) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        try:
            df = df.select([
                pl.col(col).cast(dtype).alias(col)
                for col, dtype in cls.REQUIRED_COLUMNS.items()
            ])
        except Exception as e:
            raise ValueError(f"Failed to cast metadata: {e}")

        return df