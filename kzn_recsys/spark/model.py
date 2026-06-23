"""SparkEaseModel facade + build_and_train / load_model. Mirrors the public
shape of kzn_recsys.FeaseModel but consumes Spark DataFrames."""
from __future__ import annotations

import numpy as np

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
                 user_features_df, k_values):
        """Score test users and compute precision/recall/ndcg/map/hit_rate@k + coverage.

        Mirrors src/evaluation.rs::evaluate_model semantics for EASE: each user's
        training interactions form the input; held-out test items are the relevant set.
        """
        m = self.mappings
        max_k = max(k_values)

        def _collect_by_user(df, val=True):
            out = {}
            for r in df.select("user_id", "item_id", "value").collect():
                out.setdefault(r["user_id"], {})[r["item_id"]] = float(r["value"])
            return out

        train_by_user = _collect_by_user(train_interactions_df)
        test_by_user = _collect_by_user(test_interactions_df)

        # user features as {user_id: {feature_name: value}}
        feats_by_user = {}
        for r in user_features_df.select("user_id", "feature_name", "value").collect():
            feats_by_user.setdefault(r["user_id"], {})[r["feature_name"]] = float(r["value"])

        per_k = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0,
                     "map": 0.0, "hit_rate": 0.0} for k in k_values}
        all_recs = []
        n_users = 0
        n_interactions = 0

        for uid, relevant_map in test_by_user.items():
            relevant = set(relevant_map)
            if not relevant:
                continue
            interactions = train_by_user.get(uid, {})
            features = feats_by_user.get(uid, {})
            recs = self.predict(interactions, features, top_k=max_k)
            rec_ids = [item_id for item_id, _ in recs]
            rec_idx = [m.item_to_idx[i] for i in rec_ids if i in m.item_to_idx]
            all_recs.append(rec_idx)
            relevant_idx = {m.item_to_idx[i] for i in relevant if i in m.item_to_idx}
            n_users += 1
            n_interactions += len(relevant)
            for k in k_values:
                per_k[k]["precision"] += _metrics.precision_at_k(rec_idx, relevant_idx, k)
                per_k[k]["recall"] += _metrics.recall_at_k(rec_idx, relevant_idx, k)
                per_k[k]["ndcg"] += _metrics.ndcg_at_k(rec_idx, relevant_idx, k)
                per_k[k]["hit_rate"] += _metrics.hit_rate_at_k(rec_idx, relevant_idx, k)
                per_k[k]["map"] += _metrics.mean_average_precision(rec_idx[:k], relevant_idx)

        denom = max(n_users, 1)
        metrics_out = []
        for k in sorted(k_values):
            metrics_out.append({
                "k": k,
                "precision": per_k[k]["precision"] / denom,
                "recall": per_k[k]["recall"] / denom,
                "ndcg": per_k[k]["ndcg"] / denom,
                "map": per_k[k]["map"] / denom,
                "hit_rate": per_k[k]["hit_rate"] / denom,
            })
        return {
            "metrics": metrics_out,
            "coverage": _metrics.coverage(all_recs, self.num_items),
            "num_users": n_users,
            "num_interactions": n_interactions,
        }

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
