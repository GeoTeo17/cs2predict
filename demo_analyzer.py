import pandas as pd
from predict import CS2_MAPS, build_feature_row, compute_probabilities

TICK_RATE = 64
SAMPLE_STEP_TICKS = 128  # Every 2.0 seconds in 64-tick CS2 demos

RIFLE_KEYWORDS = ("ak-47", "m4a4", "m4a1", "aug", "sg 553", "famas", "galil")
GRENADE_KEYWORDS = ("high explosive", "flashbang", "smoke", "incendiary", "molotov")


def classify_inventory(inv):
    rifles = awps = utility = 0
    if inv is None:
        return rifles, awps, utility
    items = inv if isinstance(inv, (list, tuple)) else [str(inv)]
    for item in items:
        name = str(item).lower()
        if "awp" in name:
            awps += 1
        elif any(k in name for k in RIFLE_KEYWORDS):
            rifles += 1
        elif any(k in name for k in GRENADE_KEYWORDS):
            utility += 1
    return rifles, awps, utility


def normalize_winner(raw_val):
    s = str(raw_val).strip().upper()
    if s in ("CT", "3"):
        return "CT"
    if s in ("T", "2"):
        return "T"
    return "UNKNOWN"


def analyze_demo_file(demo_path, model, feature_cols):
    try:
        from demoparser2 import DemoParser
    except ImportError as exc:
        raise RuntimeError("Missing dependency: run 'pip install demoparser2' first.") from exc

    parser = DemoParser(str(demo_path))
    header = parser.parse_header() or {}
    raw_map = str(header.get("map_name", "de_mirage")).replace("de_", "").lower()
    map_name = raw_map if raw_map in CS2_MAPS else "mirage"

    event_names = ["round_freeze_end", "round_end", "bomb_planted", "round_announce_match_start"]
    events_dict = dict(parser.parse_events(event_names))

    freeze_df = events_dict.get("round_freeze_end", pd.DataFrame())
    end_df = events_dict.get("round_end", pd.DataFrame())
    plant_df = events_dict.get("bomb_planted", pd.DataFrame())
    match_start_df = events_dict.get("round_announce_match_start", pd.DataFrame())

    if freeze_df.empty or end_df.empty:
        raise ValueError("Could not find valid round events in this .dem file.")

    min_tick = int(match_start_df["tick"].max()) if not match_start_df.empty else 0
    freeze_ticks = sorted(int(t) for t in freeze_df["tick"] if int(t) >= min_tick)
    end_rows = end_df[end_df["tick"] >= min_tick].sort_values("tick").to_dict("records")
    plant_ticks = sorted(int(t) for t in plant_df["tick"]) if not plant_df.empty else []

    rounds = []
    for f_tick in freeze_ticks:
        matching_ends = [r for r in end_rows if int(r["tick"]) > f_tick]
        if not matching_ends:
            continue
        end_row = matching_ends[0]
        e_tick = int(end_row["tick"])
        if e_tick - f_tick < TICK_RATE * 3:
            continue

        p_ticks = [p for p in plant_ticks if f_tick <= p <= e_tick]
        p_tick = p_ticks[0] if p_ticks else None
        winner = normalize_winner(end_row.get("winner", ""))
        rounds.append({"start": f_tick, "end": e_tick, "plant": p_tick, "winner": winner})

    if not rounds:
        raise ValueError("No playable rounds detected after warmup.")

    # Keep final competitive match rounds if multiple restarts occurred
    if len(rounds) > 30:
        rounds = rounds[-30:]

    sample_map = {}
    all_sample_ticks = []
    for r_idx, r in enumerate(rounds, start=1):
        ticks = list(range(r["start"], r["end"] + 1, SAMPLE_STEP_TICKS))
        if ticks[-1] != r["end"]:
            ticks.append(r["end"])
        sample_map[r_idx] = ticks
        all_sample_ticks.extend(ticks)

    props = ["team_num", "is_alive", "health", "armor_value", "balance", "has_defuser", "inventory"]
    ticks_df = parser.parse_ticks(props, ticks=sorted(set(all_sample_ticks)))
    grouped_ticks = {int(t): grp for t, grp in ticks_df.groupby("tick")}

    ct_score = 0
    t_score = 0
    timeline = []
    rounds_summary = []

    for r_idx, r in enumerate(rounds, start=1):
        r_probs = []
        r_timeline = []
        p_tick = r["plant"]

        for t in sample_map[r_idx]:
            grp = grouped_ticks.get(t)
            if grp is None:
                continue

            bomb_planted = p_tick is not None and t >= p_tick
            if bomb_planted:
                elapsed_plant = (t - p_tick) / TICK_RATE
                time_left = max(0.0, 40.0 - elapsed_plant)
            else:
                elapsed_round = (t - r["start"]) / TICK_RATE
                time_left = max(0.0, 115.0 - elapsed_round)

            ct_players = t_players = 0
            ct_health = t_health = 0
            ct_armor = t_armor = 0
            ct_money = t_money = 0
            ct_kits = 0
            ct_rifles = t_rifles = 0
            ct_awps = t_awps = 0
            ct_utility = t_utility = 0

            for _, prow in grp.iterrows():
                team_num = int(prow.get("team_num", 0))
                alive = bool(prow.get("is_alive", False))
                hp = int(prow.get("health", 0)) if alive else 0
                armor = int(prow.get("armor_value", 0)) if alive else 0
                money = int(prow.get("balance", 0))
                r_cnt, a_cnt, u_cnt = classify_inventory(prow.get("inventory")) if alive else (0, 0, 0)

                if team_num == 3:
                    ct_money += money
                    if alive and hp > 0:
                        ct_players += 1
                        ct_health += hp
                        ct_armor += armor
                        ct_rifles += r_cnt
                        ct_awps += a_cnt
                        ct_utility += u_cnt
                        if bool(prow.get("has_defuser", False)):
                            ct_kits += 1
                elif team_num == 2:
                    t_money += money
                    if alive and hp > 0:
                        t_players += 1
                        t_health += hp
                        t_armor += armor
                        t_rifles += r_cnt
                        t_awps += a_cnt
                        t_utility += u_cnt

            row, ctx = build_feature_row(
                feature_cols=feature_cols,
                map_name=map_name,
                time_left=round(time_left, 1),
                bomb_planted=bomb_planted,
                ct_players=ct_players,
                t_players=t_players,
                ct_health=ct_health,
                t_health=t_health,
                ct_money=ct_money,
                t_money=t_money,
                ct_defuse_kits=ct_kits,
                ct_score=min(12, ct_score),
                t_score=min(12, t_score),
                is_defusing=False,
                defuse_elapsed=0.0,
                game_mode="CS2 (MR12)",
                ct_armor=ct_armor,
                t_armor=t_armor,
                ct_rifles=ct_rifles,
                t_rifles=t_rifles,
                ct_awps=ct_awps,
                t_awps=t_awps,
                ct_utility=ct_utility,
                t_utility=t_utility,
            )
            probs = compute_probabilities(model, row, ctx)
            ct_pct = round(probs["CT"] * 100.0, 1)

            pt = {
                "round_num": r_idx,
                "ct_pct": ct_pct,
                "bomb_planted": bomb_planted,
                "ct_alive": ct_players,
                "t_alive": t_players,
                "label": f"R{r_idx} ({ct_players}v{t_players}{' C4' if bomb_planted else ''}) -> CT {ct_pct:.1f}%",
            }
            r_probs.append(ct_pct)
            r_timeline.append(pt)
            timeline.append(pt)

        winner = r["winner"]
        min_ct = min(r_probs) if r_probs else 50.0
        max_ct = max(r_probs) if r_probs else 50.0
        start_ct = r_probs[0] if r_probs else 50.0

        # Swing measures how hard the winning team came back from behind
        if winner == "CT":
            comeback_swing = round(100.0 - min_ct, 1)
        elif winner == "T":
            comeback_swing = round(max_ct, 1)
        else:
            comeback_swing = round(max_ct - min_ct, 1)

        rounds_summary.append({
            "Round": r_idx,
            "Score Before": f"{ct_score} - {t_score}",
            "Winner": winner,
            "C4 Planted": "Yes" if p_tick is not None else "No",
            "Start CT %": f"{start_ct:.1f}%",
            "Lowest CT %": f"{min_ct:.1f}%",
            "Peak CT %": f"{max_ct:.1f}%",
            "Swing": comeback_swing,
        })

        if winner == "CT":
            ct_score += 1
        elif winner == "T":
            t_score += 1

    top_swings = sorted(rounds_summary, key=lambda x: x["Swing"], reverse=True)[:3]

    return {
        "map_name": raw_map.upper(),
        "final_score": f"CT {ct_score} - {t_score} T",
        "total_rounds": len(rounds_summary),
        "timeline": timeline,
        "rounds_summary": rounds_summary,
        "top_swings": top_swings,
    }