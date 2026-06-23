"""SparkEaseModel facade + build_and_train / load_model. Mirrors the public
shape of kzn_recsys.FeaseModel but consumes Spark DataFrames."""
from __future__ import annotations

import numpy as np

from . import dataframes as _df
from . import ease_core as _core
from . import gram as _gram
from . import feas_codec as _codec


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
