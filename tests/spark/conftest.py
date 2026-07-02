"""Shared fixtures for PySpark EASE tests."""
import importlib.util
import os
import subprocess
import pytest

# Skip the whole spark suite (at collection time) when its optional deps are
# absent — the test modules import numpy at module top, which errors during
# collection before any fixture-level importorskip can fire. This keeps
# `pytest tests/` green in the native-only CI / any env without the [spark]
# extra installed. Install with `pip install -e '.[spark]'` to run them.
if any(importlib.util.find_spec(m) is None for m in ("numpy", "scipy", "pyspark")):
    collect_ignore_glob = ["test_*.py"]


def _ensure_compatible_java_home():
    """Best-effort: on macOS, pick a non-Java-26 JDK when JAVA_HOME is unset.

    PySpark 4.x is incompatible with Java 26 (removed jdk.internal.ref.Cleaner).
    This only fires when JAVA_HOME is not already set (so it never overrides
    CI/user config) and is a silent no-op off macOS or when
    /usr/libexec/java_home is unavailable — CI must set JAVA_HOME itself.

    Uses the XML plist output of java_home -X to enumerate all installed JDKs
    and selects the highest-version one that is not Java 26.  This avoids the
    ``-v '!26'`` exclusion syntax, which macOS java_home does not support.
    """
    if os.environ.get("JAVA_HOME"):
        return
    try:
        import plistlib
        result = subprocess.run(
            ["/usr/libexec/java_home", "-X"],
            capture_output=True, check=True,
        )
        jvms = plistlib.loads(result.stdout)
        candidates = [
            (int(jvm.get("JVMVersion", "0").split(".")[0]), jvm.get("JVMHomePath", ""))
            for jvm in jvms
            if "JVMHomePath" in jvm
        ]
        candidates = [(v, p) for v, p in candidates if v != 26]
        if candidates:
            os.environ["JAVA_HOME"] = sorted(candidates, reverse=True)[0][1]
    except Exception:
        pass  # not macOS, or java_home unavailable; CI/user must set JAVA_HOME


_ensure_compatible_java_home()


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
