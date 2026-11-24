import joblib
import pandas as pd

MODEL_PATH = "models/csgo_model.joblib"
PLANT_TIME = 3.2

CSGO_MAPS = ["dust2", "mirage", "inferno", "nuke", "overpass", "train", "vertigo", "cache"]
CS2_MAPS = ["mirage", "inferno", "nuke", "overpass", "vertigo", "dust2", "train"]


def build_feature_row(
    feature_cols,
    map_name,
    time_left,
    bomb_planted,
    ct_players,
    t_players,
    ct_health,
    t_health,
    ct_money,
    t_money,
    ct_defuse_kits=0,
    ct_score=8,
    t_score=8,
    is_defusing=False,
    defuse_elapsed=0.0,
    game_mode="CS2 (MR12)",
    ct_armor=500,
    t_armor=500,
    ct_rifles=4,
    t_rifles=4,
    ct_awps=1,
    t_awps=1,
    ct_utility=10,
    t_utility=10,
):
    if game_mode == "CS2 (MR12)":
        scaled_ct_score = min(15, round(ct_score * (15.0 / 12.0)))
        scaled_t_score = min(15, round(t_score * (15.0 / 12.0)))
    else:
        scaled_ct_score = ct_score
        scaled_t_score = t_score

    req_defuse = 5.0 if ct_defuse_kits > 0 else 10.0
    time_needed = max(0.0, req_defuse - defuse_elapsed)
    effective_time = (time_left + defuse_elapsed) if is_defusing else time_left

    plant_margin = (time_left - PLANT_TIME) if not bomb_planted else 0.0
    can_plant = int((not bomb_planted) and t_players > 0 and time_left >= PLANT_TIME)
    defuse_margin = (effective_time - req_defuse) if bomb_planted else 0.0
    can_defuse = int(bomb_planted and ct_players > 0 and defuse_margin >= 0)

    row = pd.DataFrame(0, index=[0], columns=feature_cols)
    row["time_left"] = effective_time
    row["ct_score"] = scaled_ct_score
    row["t_score"] = scaled_t_score
    row["bomb_planted"] = int(bomb_planted)
    row["ct_health"] = ct_health
    row["t_health"] = t_health
    row["ct_armor"] = ct_armor
    row["t_armor"] = t_armor
    row["ct_money"] = ct_money
    row["t_money"] = t_money
    row["ct_players_alive"] = ct_players
    row["t_players_alive"] = t_players
    row["ct_defuse_kits"] = ct_defuse_kits
    row["ct_rifles"] = ct_rifles
    row["t_rifles"] = t_rifles
    row["ct_awps"] = ct_awps
    row["t_awps"] = t_awps
    row["ct_utility"] = ct_utility
    row["t_utility"] = t_utility

    map_col = f"map_de_{map_name}"
    if map_col in row.columns:
        row[map_col] = 1

    row["diff_health"] = ct_health - t_health
    row["diff_armor"] = ct_armor - t_armor
    row["diff_players"] = ct_players - t_players
    row["diff_money"] = ct_money - t_money
    row["diff_firepower"] = (ct_rifles + 1.5 * ct_awps) - (t_rifles + 1.5 * t_awps)
    row["diff_utility"] = ct_utility - t_utility
    row["plant_margin"] = plant_margin
    row["can_plant"] = can_plant
    row["defuse_margin"] = defuse_margin
    row["can_defuse"] = can_defuse

    context = {
        "bomb_planted": bool(bomb_planted),
        "time_left": time_left,
        "ct_players": ct_players,
        "t_players": t_players,
        "impossible_defuse": bool(bomb_planted) and (ct_players == 0 or time_left < time_needed),
    }
    return row, context


def compute_probabilities(model, features, ctx):
    if ctx["ct_players"] == 0:
        return {"CT": 0.0, "T": 1.0}
    if not ctx["bomb_planted"] and ctx["t_players"] == 0:
        return {"CT": 1.0, "T": 0.0}
    if ctx["impossible_defuse"]:
        return {"CT": 0.0, "T": 1.0}

    raw_probs = dict(zip(model.classes_, model.predict_proba(features)[0]))

    if not ctx["bomb_planted"] and ctx["time_left"] < PLANT_TIME:
        time_factor = max(0.0, ctx["time_left"] / PLANT_TIME)
        manpower_factor = min(1.0, ctx["t_players"] / (ctx["ct_players"] * 2.0))
        t_prob = min(raw_probs["T"], 0.25) * time_factor * manpower_factor
        return {"CT": 1.0 - t_prob, "T": t_prob}

    return raw_probs


def main():
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["columns"]

    mode_in = input("Mode (1 = CS2 MR12, 2 = CS:GO MR15): ").strip()
    game_mode = "CS2 (MR12)" if mode_in != "2" else "CS:GO (MR15)"
    valid_maps = CS2_MAPS if game_mode == "CS2 (MR12)" else CSGO_MAPS

    while True:
        map_name = input(f"\nMap ({'/'.join(valid_maps)}) or 'q' to quit: ").strip().lower()
        if map_name == "q":
            break
        if map_name not in valid_maps:
            print("Invalid map name.")
            continue

        ct_score = int(input("CT score: "))
        t_score = int(input("T score: "))
        bomb_planted = bool(int(input("Bomb planted (1 = yes, 0 = no): ")))
        time_left = float(input("Timer left (sec): "))
        ct_players = int(input("CT alive (0-5): "))
        t_players = int(input("T alive (0-5): "))
        ct_kits = int(input("CT defuse kits (0-5): "))

        is_defusing = False
        defuse_elapsed = 0.0
        if bomb_planted and ct_players > 0:
            is_defusing = bool(int(input("CT already defusing? (1 = yes, 0 = no): ")))
            if is_defusing:
                defuse_elapsed = float(input("Seconds already on defuse: "))

        ct_health = int(input("Total CT health: "))
        t_health = int(input("Total T health: "))
        ct_armor = int(input("Total CT armor (0-500): "))
        t_armor = int(input("Total T armor (0-500): "))
        ct_rifles = int(input("CT rifles (0-5): "))
        t_rifles = int(input("T rifles (0-5): "))
        ct_awps = int(input("CT AWPs (0-5): "))
        t_awps = int(input("T AWPs (0-5): "))
        ct_utility = int(input("CT grenades total (0-20): "))
        t_utility = int(input("T grenades total (0-20): "))
        ct_money = int(input("Total CT money: "))
        t_money = int(input("Total T money: "))

        row, ctx = build_feature_row(
            feature_cols, map_name, time_left, bomb_planted,
            ct_players, t_players, ct_health, t_health,
            ct_money, t_money, ct_kits, ct_score, t_score,
            is_defusing, defuse_elapsed, game_mode,
            ct_armor, t_armor, ct_rifles, t_rifles,
            ct_awps, t_awps, ct_utility, t_utility
        )
        probs = compute_probabilities(model, row, ctx)

        print("\nWin probabilities:")
        for team in ["CT", "T"]:
            print(f"  {team}: {probs[team] * 100:.1f}%")


if __name__ == "__main__":
    main()