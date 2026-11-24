from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

DATA_PATH = "data/csgo_round_snapshots.csv"
MODEL_PATH = "models/csgo_model.joblib"

BASE_COLS = [
    "time_left", "map", "ct_score", "t_score", "bomb_planted",
    "ct_health", "t_health", "ct_armor", "t_armor",
    "ct_money", "t_money", "ct_players_alive", "t_players_alive",
    "ct_defuse_kits", "round_winner"
]

RIFLE_SUFFIXES = ["ak47", "m4a4", "m4a1s", "aug", "sg553", "famas", "galilar"]
UTILITY_SUFFIXES = ["hegrenade", "flashbang", "smokegrenade", "incendiarygrenade", "molotovgrenade"]


def sum_existing_cols(raw_df, col_names):
    valid = [c for c in col_names if c in raw_df.columns]
    return raw_df[valid].sum(axis=1) if valid else 0


def engineer_features(raw_df):
    df = raw_df[BASE_COLS].copy()
    df["bomb_planted"] = df["bomb_planted"].astype(int)

    df["ct_rifles"] = sum_existing_cols(raw_df, [f"ct_weapon_{w}" for w in RIFLE_SUFFIXES])
    df["t_rifles"] = sum_existing_cols(raw_df, [f"t_weapon_{w}" for w in RIFLE_SUFFIXES])
    df["ct_awps"] = sum_existing_cols(raw_df, ["ct_weapon_awp"])
    df["t_awps"] = sum_existing_cols(raw_df, ["t_weapon_awp"])
    df["ct_utility"] = sum_existing_cols(raw_df, [f"ct_grenade_{g}" for g in UTILITY_SUFFIXES])
    df["t_utility"] = sum_existing_cols(raw_df, [f"t_grenade_{g}" for g in UTILITY_SUFFIXES])

    df["diff_health"] = df["ct_health"] - df["t_health"]
    df["diff_armor"] = df["ct_armor"] - df["t_armor"]
    df["diff_players"] = df["ct_players_alive"] - df["t_players_alive"]
    df["diff_money"] = df["ct_money"] - df["t_money"]
    df["diff_firepower"] = (df["ct_rifles"] + 1.5 * df["ct_awps"]) - (df["t_rifles"] + 1.5 * df["t_awps"])
    df["diff_utility"] = df["ct_utility"] - df["t_utility"]

    df["plant_margin"] = (df["time_left"] - 3.2) * (1 - df["bomb_planted"])
    df["can_plant"] = (
        (df["bomb_planted"] == 0)
        & (df["t_players_alive"] > 0)
        & (df["time_left"] >= 3.2)
    ).astype(int)

    req_defuse = df["ct_defuse_kits"].gt(0).map({True: 5.0, False: 10.0})
    df["defuse_margin"] = (df["time_left"] - req_defuse) * df["bomb_planted"]
    df["can_defuse"] = (
        (df["bomb_planted"] == 1)
        & (df["ct_players_alive"] > 0)
        & (df["defuse_margin"] >= 0)
    ).astype(int)

    return pd.get_dummies(df, columns=["map"])


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    ct_idx = list(model.classes_).index("CT")
    ct_probs = model.predict_proba(X_test)[:, ct_idx]
    y_bin = (y_test == "CT").astype(int)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_bin, ct_probs)
    brier = brier_score_loss(y_bin, ct_probs)
    print(f"[{name}] Accuracy: {acc * 100:.2f}% | Log Loss: {ll:.4f} | Brier: {brier:.4f}")


def train():
    raw_df = pd.read_csv(DATA_PATH)
    df = engineer_features(raw_df)

    X = df.drop("round_winner", axis=1)
    y = df["round_winner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training models with weapons, armor & utility...")
    rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", rf, X_test, y_test)

    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    hgb.fit(X_train, y_train)
    evaluate_model("HistGradientBoosting", hgb, X_test, y_test)

    Path("models").mkdir(exist_ok=True)
    artifact = {"model": rf, "columns": list(X.columns)}
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model artifact to {MODEL_PATH}")


if __name__ == "__main__":
    train()