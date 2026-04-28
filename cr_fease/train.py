"""Standalone training script for FEASE model."""

import argparse
import cr_fease as fease


def main():
    parser = argparse.ArgumentParser(description="Train a FEASE recommender model")
    parser.add_argument("--interactions", required=True, help="Path to interactions parquet/csv")
    parser.add_argument("--user-features", required=True, help="Path to user features parquet/csv")
    parser.add_argument("--item-features", required=True, help="Path to item features parquet/csv")
    parser.add_argument("--alpha", type=float, default=1.0, help="Item feature weight")
    parser.add_argument("--beta", type=float, default=1.0, help="User feature weight")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=100.0, help="L2 regularization")
    parser.add_argument("--meta-weight", type=float, default=0.0, help="Metadata row weight")
    parser.add_argument("--output", default="model.fease", help="Output model path")
    args = parser.parse_args()

    model = fease.build_and_train(
        interactions_path=args.interactions,
        user_features_path=args.user_features,
        item_features_path=args.item_features,
        alpha=args.alpha,
        beta=args.beta,
        lambda_=args.lambda_,
        meta_weight=args.meta_weight,
    )

    print(f"Trained: {model.num_items} items, {model.num_user_features} user features, {model.num_item_features} item features")

    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
