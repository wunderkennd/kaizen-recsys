def test_spark_subpackage_imports_without_touching_native():
    # Importing the spark subpackage must not require _native.
    import importlib
    mod = importlib.import_module("kzn_recsys.spark")
    assert hasattr(mod, "build_and_train")
    assert hasattr(mod, "load_model")


def test_has_native_flag_exists():
    import kzn_recsys
    assert hasattr(kzn_recsys, "_HAS_NATIVE")
    assert isinstance(kzn_recsys._HAS_NATIVE, bool)
