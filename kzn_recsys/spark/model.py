"""SparkEaseModel facade + build_and_train / load_model. Mirrors the public
shape of kzn_recsys.FeaseModel but consumes Spark DataFrames."""
from __future__ import annotations

import math

import numpy as np

from . import availability as _av
from . import dataframes as _df
from . import ease_core as _core
from . import gram as _gram
from . import feas_codec as _codec
from . import metrics as _metrics


class SparkEaseModel:
    def __init__(self, s_matrix, mappings, params, weighting=None, num_item_features=0):
        self.s_matrix = np.asfortranarray(s_matrix)
        self.mappings = mappings
        self.params = params
        self.weighting = weighting
        self.num_items = len(mappings.idx_to_item)
        self.num_user_features = len(mappings.idx_to_user_feature)
        self.num_item_features = num_item_features

    def predict(self, interactions: dict, features: dict, top_k: int):
        """interactions/features are {string_id: value}. Returns [(item_id, score)]."""
        m = self.mappings
        inter_idx = [(m.item_to_idx[k], v) for k, v in interactions.items()
                     if k in m.item_to_idx]
        feat_idx = [(m.user_feature_to_idx[k], v) for k, v in features.items()
                    if k in m.user_feature_to_idx]
        scores = _core.predict_scores(
            self.s_matrix, self.num_items, self.num_user_features,
            inter_idx, feat_idx, self.params.beta,
        )
        # exclude items the user already interacted with
        seen = {m.item_to_idx[k] for k in interactions if k in m.item_to_idx}
        order = np.argsort(-scores, kind="stable")
        out = []
        for j in order:
            j = int(j)
            if j in seen:
                continue
            out.append((m.idx_to_item[j], float(scores[j])))
            if len(out) == top_k:
                break
        return out

    def predict_similar_items(self, item_id: str, top_k: int):
        m = self.mappings
        if item_id not in m.item_to_idx:
            return []
        pairs = _core.predict_similar_items(
            self.s_matrix, m.item_to_idx[item_id], self.num_items, top_k
        )
        return [(m.idx_to_item[j], score) for j, score in pairs]

    def evaluate(self, test_interactions_df, train_interactions_df,
                 user_features_df, k_values, availability_df=None,
                 user_territory_feature=None, reference_days_ago=None,
                 item_age_bucket_edges=_av.DEFAULT_ITEM_AGE_BUCKET_EDGES):
        """Score test users and compute precision/recall/ndcg/map/hit_rate@k + coverage.

        Mirrors src/evaluation.rs::evaluate_model semantics for EASE: each user's
        training interactions form the input; held-out test items are the relevant set.

        With ``availability_df`` (see ``kzn_recsys.spark.availability``) the
        ranking is restricted to items eligible for the user at their
        reference time — ``reference_days_ago`` when given (the temporal
        split's cutoff), else per user the oldest held-out interaction
        (largest ``days_ago`` in the test frame, which must then carry a
        non-null ``days_ago``). Ineligible relevants are dropped and
        counted, coverage uses the eligible catalog, and the report gains an
        ``availability`` dict with an item-age breakdown. Without it the
        report is exactly the legacy one.
        """
        m = self.mappings
        max_k = max(k_values)
        per_user_reference = availability_df is not None and reference_days_ago is None

        def _collect_by_user(df, with_days):
            out, ref = {}, {}
            cols = ["user_id", "item_id", "value"] + (["days_ago"] if with_days else [])
            for i, r in enumerate(df.select(*cols).collect()):
                out.setdefault(r["user_id"], {})[r["item_id"]] = float(r["value"])
                if with_days and r["item_id"] in m.item_to_idx:
                    d = r["days_ago"]
                    if d is None:
                        raise ValueError(f"availability: test row {i} has a null days_ago")
                    d = float(d)
                    if d > ref.get(r["user_id"], float("-inf")):
                        ref[r["user_id"]] = d
            return out, ref

        if per_user_reference and "days_ago" not in test_interactions_df.columns:
            raise ValueError(
                "availability filtering without reference_days_ago needs a `days_ago` column "
                "in the test frame (the per-user reference time is the oldest held-out "
                "interaction)"
            )
        train_by_user, _ = _collect_by_user(train_interactions_df, False)
        test_by_user, test_reference = _collect_by_user(test_interactions_df, per_user_reference)

        # user features as {user_id: {feature_name: value}}
        feats_by_user = {}
        for r in user_features_df.select("user_id", "feature_name", "value").collect():
            feats_by_user.setdefault(r["user_id"], {})[r["feature_name"]] = float(r["value"])

        table = territories = None
        edges, bounds = [], []
        if availability_df is not None:
            if reference_days_ago is not None and not (
                math.isfinite(reference_days_ago) and reference_days_ago >= 0.0
            ):
                raise ValueError("reference_days_ago must be finite and >= 0")
            edges = _av.check_edges(item_age_bucket_edges or [])
            bounds = _av.bucket_bounds(edges)
            table = _av.AvailabilityTable.from_spark(availability_df, m.idx_to_item)
            territories = (_av.user_territories(feats_by_user, user_territory_feature)
                           if user_territory_feature else {})
        elif user_territory_feature is not None or reference_days_ago is not None:
            raise ValueError("user_territory_feature / reference_days_ago require availability_df")

        def _acc():
            return {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0,
                        "map": 0.0, "hit_rate": 0.0} for k in k_values}

        def _add(acc, rec_idx, relevant_idx):
            for k in k_values:
                acc[k]["precision"] += _metrics.precision_at_k(rec_idx, relevant_idx, k)
                acc[k]["recall"] += _metrics.recall_at_k(rec_idx, relevant_idx, k)
                acc[k]["ndcg"] += _metrics.ndcg_at_k(rec_idx, relevant_idx, k)
                acc[k]["hit_rate"] += _metrics.hit_rate_at_k(rec_idx, relevant_idx, k)
                acc[k]["map"] += _metrics.mean_average_precision(rec_idx[:k], relevant_idx)

        def _finish(acc, n):
            denom = max(n, 1)
            return [{"k": k, **{name: acc[k][name] / denom
                                for name in ("precision", "recall", "ndcg", "map", "hit_rate")}}
                    for k in sorted(k_values)]

        per_k = _acc()
        bucket_accs = [_acc() for _ in bounds]
        bucket_users = [0 for _ in bounds]
        eligible_cache = {}
        eligible_union = set()
        dropped = skipped = 0
        all_recs = []
        n_users = 0
        n_interactions = 0

        for uid, relevant_map in test_by_user.items():
            relevant = set(relevant_map)
            if not relevant:
                continue
            eligible = None
            reference = 0.0
            if table is not None:
                reference = (reference_days_ago if reference_days_ago is not None
                             else test_reference.get(uid))
                if reference is None:
                    raise ValueError(
                        f"availability: no reference time for test user {uid!r} "
                        "(no test row with a non-null days_ago on a known item)"
                    )
                territory = territories.get(uid)
                key = (territory, reference)
                if key not in eligible_cache:
                    eligible_cache[key] = table.eligible_set(territory, reference)
                eligible = eligible_cache[key]
                kept = {i for i in relevant if i in eligible}
                dropped += len(relevant) - len(kept)
                if not kept:
                    skipped += 1
                    continue
                eligible_union |= eligible
                relevant = kept
            interactions = train_by_user.get(uid, {})
            features = feats_by_user.get(uid, {})
            if eligible is None:
                recs = self.predict(interactions, features, top_k=max_k)
            else:
                recs = self._predict_within(interactions, features, max_k, eligible)
            rec_ids = [item_id for item_id, _ in recs]
            rec_idx = [m.item_to_idx[i] for i in rec_ids if i in m.item_to_idx]
            all_recs.append(rec_idx)
            relevant_idx = {m.item_to_idx[i] for i in relevant if i in m.item_to_idx}
            n_users += 1
            n_interactions += len(relevant)
            _add(per_k, rec_idx, relevant_idx)
            if table is not None and bounds:
                per_bucket = [set() for _ in bounds]
                for item in relevant:
                    age = table.item_age_days(item, reference)
                    if age is not None and item in m.item_to_idx:
                        per_bucket[_av.age_bucket(age, edges)].add(m.item_to_idx[item])
                for b, rel in enumerate(per_bucket):
                    if rel:
                        _add(bucket_accs[b], rec_idx, rel)
                        bucket_users[b] += 1

        if n_users == 0 and skipped:
            raise ValueError(
                "No test users could be evaluated: every test item was ineligible at its "
                f"user's reference time ({skipped} users skipped)"
            )

        report = {
            "metrics": _finish(per_k, n_users),
            "coverage": _metrics.coverage(
                all_recs, len(eligible_union) if table is not None else self.num_items
            ),
            "num_users": n_users,
            "num_interactions": n_interactions,
        }
        if table is not None:
            report["availability"] = {
                "reference_days_ago": reference_days_ago,
                "num_eligible_items": len(eligible_union),
                "num_items_without_availability": table.num_items_unlisted,
                "num_test_interactions_dropped": dropped,
                "num_users_skipped": skipped,
                "item_age_buckets": [
                    {"label": label, "min_age_days": lo, "max_age_days": hi,
                     "num_users": bucket_users[b],
                     "metrics": _finish(bucket_accs[b], bucket_users[b]) if bucket_users[b] else []}
                    for b, (label, lo, hi) in enumerate(bounds)
                ],
            }
        return report

    def _predict_within(self, interactions: dict, features: dict, top_k: int, eligible):
        """`predict` restricted to item ids in `eligible` (availability filter)."""
        m = self.mappings
        inter_idx = [(m.item_to_idx[k], v) for k, v in interactions.items()
                     if k in m.item_to_idx]
        feat_idx = [(m.user_feature_to_idx[k], v) for k, v in features.items()
                    if k in m.user_feature_to_idx]
        scores = _core.predict_scores(
            self.s_matrix, self.num_items, self.num_user_features,
            inter_idx, feat_idx, self.params.beta,
        )
        seen = {m.item_to_idx[k] for k in interactions if k in m.item_to_idx}
        order = np.argsort(-scores, kind="stable")
        out = []
        for j in order:
            j = int(j)
            if j in seen or m.idx_to_item[j] not in eligible:
                continue
            out.append((m.idx_to_item[j], float(scores[j])))
            if len(out) == top_k:
                break
        return out

    def save(self, path: str) -> None:
        m = self.mappings
        wc = self.weighting
        artifact = _codec.FeaseArtifact(
            version=2,
            s_nrows=self.s_matrix.shape[0],
            s_ncols=self.s_matrix.shape[1],
            s_data=self.s_matrix,
            num_items=self.num_items,
            num_user_features=self.num_user_features,
            num_item_features=self.num_item_features,
            alpha=self.params.alpha, beta=self.params.beta,
            lambda_=self.params.lambda_, meta_weight=self.params.meta_weight,
            user_to_idx=list(m.user_to_idx.items()), idx_to_user=m.idx_to_user,
            item_to_idx=list(m.item_to_idx.items()), idx_to_item=m.idx_to_item,
            user_feature_to_idx=list(m.user_feature_to_idx.items()),
            idx_to_user_feature=m.idx_to_user_feature,
            item_feature_to_idx=list(m.item_feature_to_idx.items()),
            idx_to_item_feature=m.idx_to_item_feature,
            weighting_config=wc,
        )
        _codec.write_feas(artifact, path)


def build_and_train(interactions_df, user_features_df, item_features_df,
                    alpha=1.0, beta=1.0, lambda_=150.0, meta_weight=0.0,
                    weighting=None, strategy="collect"):
    params = _core.EaseParams(alpha=alpha, beta=beta, lambda_=lambda_, meta_weight=meta_weight)
    mappings = _df.build_mappings(interactions_df, user_features_df, item_features_df)
    if strategy == "distributed":
        S = _gram.gram_distributed(interactions_df, user_features_df, item_features_df,
                                   mappings, params, weighting)
    elif strategy == "collect":
        S = _gram.gram_collect(interactions_df, user_features_df, item_features_df,
                               mappings, params, weighting)
    else:
        raise ValueError(f"unknown strategy {strategy!r}; expected 'collect' or 'distributed'")
    if weighting is not None and getattr(weighting, "sparsity_threshold", 0.0) > 0.0:
        _core.prune_sparse(S, weighting.sparsity_threshold)
    return SparkEaseModel(S, mappings, params, weighting,
                          num_item_features=len(mappings.idx_to_item_feature))


def load_model(path: str) -> SparkEaseModel:
    art = _codec.read_feas(path)
    mappings = _df.Mappings(
        user_to_idx=dict(art.user_to_idx), idx_to_user=art.idx_to_user,
        item_to_idx=dict(art.item_to_idx), idx_to_item=art.idx_to_item,
        user_feature_to_idx=dict(art.user_feature_to_idx),
        idx_to_user_feature=art.idx_to_user_feature,
        item_feature_to_idx=dict(art.item_feature_to_idx),
        idx_to_item_feature=art.idx_to_item_feature,
    )
    params = _core.EaseParams(alpha=art.alpha, beta=art.beta,
                              lambda_=art.lambda_, meta_weight=art.meta_weight)
    return SparkEaseModel(art.s_data, mappings, params, art.weighting_config,
                          num_item_features=art.num_item_features)
