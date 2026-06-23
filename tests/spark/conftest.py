"""Shared fixtures for PySpark EASE tests."""
import os
import pytest

# PySpark 4.x requires a JDK that still ships jdk.internal.ref.Cleaner (pre-JDK 9 API).
# Java 26 removed it; set JAVA_HOME to Microsoft JDK 25 which retains compatibility.
_MS_JDK25 = "/Library/Java/JavaVirtualMachines/microsoft-25.jdk/Contents/Home"
if os.path.isdir(_MS_JDK25) and os.environ.get("JAVA_HOME", "") != _MS_JDK25:
    os.environ["JAVA_HOME"] = _MS_JDK25


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
