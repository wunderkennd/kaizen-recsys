import pytest

pytest.importorskip("numpy")
from kzn_recsys.spark import (
    make_synthetic, feature_engineering, build_and_train, load_model,
    random_split, grid_search,
)


def test_end_to_end_pipeline(spark, tmp_path):
    # 01 ingest: synthetic already long-format; feature_engineering is exercised
    # separately, so here we consume make_synthetic output directly.
    interactions, users, items = make_synthetic(spark, n_users=80, n_items=30,
                                                n_personas=4, avg_interactions=10, seed=1)

    # 02 train (collect) + a tiny grid search
    res = grid_search(interactions, users, items,
                      {"lambda_": [50.0, 150.0]}, k_folds=2, eval_k=5, seed=1)
    assert "best_params" in res and "lambda_" in res["best_params"]
    best_lambda = res["best_params"]["lambda_"]

    model = build_and_train(interactions, users, items,
                            lambda_=best_lambda, strategy="collect")
    path = str(tmp_path / "model.feas")
    model.save(path)

    # distributed strategy trains too (parity of interface, not asserting equality here)
    model_dist = build_and_train(interactions, users, items,
                                 lambda_=best_lambda, strategy="distributed")
    assert model_dist.num_items == model.num_items

    # 03 evaluate
    train_df, test_df = random_split(interactions, test_ratio=0.2, seed=1)
    metrics = model.evaluate(test_df, train_df, users, k_values=[5, 10])
    ndcg5 = next(m["ndcg"] for m in metrics["metrics"] if m["k"] == 5)
    assert 0.0 <= ndcg5 <= 1.0

    # 04 predict from the reloaded artifact
    loaded = load_model(path)
    warm_uid = interactions.select("user_id").first()["user_id"]
    warm_inter = {r["item_id"]: r["value"]
                  for r in interactions.filter(interactions.user_id == warm_uid).collect()}
    recs = loaded.predict(warm_inter, {}, top_k=5)
    assert len(recs) <= 5
    # cold-start: no interactions, persona feature only
    cold = loaded.predict({}, {"persona=0": 1.0}, top_k=5)
    assert isinstance(cold, list)
