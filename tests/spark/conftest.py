"""Shared fixtures for PySpark EASE tests."""
import pytest


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local SparkSession. Skips the test if pyspark is absent."""
    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("kzn_recsys-spark-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
