import pandas as pd
import lightgbm as lgb
import os


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "training_data.csv"
MODEL_PATH = PROJECT_ROOT / "recommender_v2" / "training" / "ranker.txt"


def main():

    print("Loading training data...")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows: {len(df)}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Positive labels: {df['label'].sum()}")

    # Columns that should not be used as model features.
    drop_columns = [
        "track_id",
        "session_id",
        "label",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in drop_columns
    ]

    X = df[feature_columns]
    y = df["label"]

    # LambdaRank needs the number of candidates belonging
    # to each ranking query/session.
    groups = (
        df.groupby("session_id", sort=False)
        .size()
        .tolist()
    )

    print(f"Features: {len(feature_columns)}")
    print(f"Ranking groups: {len(groups)}")

    # LightGBM ranking dataset.
    train_data = lgb.Dataset(
        X,
        label=y,
        group=groups,
        feature_name=feature_columns,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [5, 10, 20],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "verbosity": -1,
    }

    print("Training LightGBM LambdaRank...")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
    )

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    model.save_model(MODEL_PATH)

    print()
    print("Training complete.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()