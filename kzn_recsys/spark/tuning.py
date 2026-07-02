"""Grid/random search with user k-fold CV, optimizing NDCG@eval_k.

Folds partition each user's interaction rows into k groups; fold f trains on
the complement and evaluates on fold f. Mirrors the intent of src/tuning.rs.
"""
from __future__ import annotations

import itertools
import random

from .model import build_and_train
from . import metrics as _metrics


def _assign_folds(interactions_df, k_folds, seed):
    """Return a copy of interactions_df with an added integer `_fold` in [0, k_folds).

    Collects rows once and assigns folds by per-user shuffled position, using the
    stable list index as the row id. Avoids F.monotonically_increasing_id(), which
    is nondeterministic across the separate executions a collect-then-join triggers.
    """
    from pyspark.sql.types import IntegerType, StructField, StructType

    rows = interactions_df.collect()
    by_user = {}
    for idx, r in enumerate(rows):
        by_user.setdefault(r["user_id"], []).append(idx)

    rng = random.Random(seed)
    fold_of = {}
    for uid in sorted(by_user):
        positions = list(by_user[uid])
        rng.shuffle(positions)
        for pos, idx in enumerate(positions):
            fold_of[idx] = pos % k_folds

    spark = interactions_df.sparkSession
    schema = StructType(interactions_df.schema.fields + [StructField("_fold", IntegerType())])
    new_rows = [tuple(r) + (fold_of[i],) for i, r in enumerate(rows)]
    return spark.createDataFrame(new_rows, schema)


def _score_params(interactions_df, user_features_df, item_features_df,
                  params, k_folds, eval_k, seed):
    """Mean NDCG@eval_k across folds for one parameter set."""
    from pyspark.sql import functions as F
    folded = _assign_folds(interactions_df, k_folds, seed).cache()
    ndcgs = []
    for f in range(k_folds):
        train_df = folded.where(F.col("_fold") != f).drop("_fold")
        test_df = folded.where(F.col("_fold") == f).drop("_fold")
        if test_df.count() == 0:
            continue
        model = build_and_train(
            train_df, user_features_df, item_features_df,
            alpha=params.get("alpha", 1.0), beta=params.get("beta", 1.0),
            lambda_=params.get("lambda_", 150.0),
            meta_weight=params.get("meta_weight", 0.0),
        )
        report = model.evaluate(test_df, train_df, user_features_df, k_values=[eval_k])
        ndcgs.append(report["metrics"][0]["ndcg"])
    folded.unpersist()
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def _run_trials(interactions_df, user_features_df, item_features_df,
                param_sets, k_folds, eval_k, seed):
    trials = []
    for params in param_sets:
        score = _score_params(interactions_df, user_features_df, item_features_df,
                              params, k_folds, eval_k, seed)
        trials.append({"params": params, "score": score})
    best = max(trials, key=lambda t: t["score"]) if trials else {"params": {}, "score": 0.0}
    return {"best_params": best["params"], "best_score": best["score"], "trials": trials}


def grid_search(interactions_df, user_features_df, item_features_df,
                param_grid, k_folds=3, eval_k=10, seed=42):
    keys = sorted(param_grid)
    param_sets = [dict(zip(keys, combo))
                  for combo in itertools.product(*(param_grid[k] for k in keys))]
    return _run_trials(interactions_df, user_features_df, item_features_df,
                       param_sets, k_folds, eval_k, seed)


def random_search(interactions_df, user_features_df, item_features_df,
                  param_distributions, n_iter=10, k_folds=3, eval_k=10, seed=42):
    rng = random.Random(seed)
    keys = sorted(param_distributions)
    param_sets = []
    for _ in range(n_iter):
        param_sets.append({k: rng.choice(param_distributions[k]) for k in keys})
    return _run_trials(interactions_df, user_features_df, item_features_df,
                       param_sets, k_folds, eval_k, seed)
