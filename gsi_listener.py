import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 3000
STATE_FILE = Path("live_state.json")
TEMP_FILE = Path("live_state.tmp")

SUPPORTED_MAPS = {"mirage", "inferno", "nuke", "overpass", "vertigo", "dust2", "train", "cache"}


def extract_gear_counts(weapons_dict):
    rifles = 0
    awps = 0
    utility = 0
    for w_info in (weapons_dict or {}).values():
        w_name = w_info.get("name", "").lower()
        w_type = w_info.get("type", "")
        if "awp" in w_name:
            awps += 1
        elif w_type == "Rifle":
            rifles += 1
        elif w_type == "Grenade" and "decoy" not in w_name:
            utility += 1
    return rifles, awps, utility


def parse_gsi_payload(payload):
    map_info = payload.get("map", {})
    round_info = payload.get("round", {})
    phase_cd = payload.get("phase_countdowns", {})
    bomb_info = payload.get("bomb", {})
    all_players = payload.get("allplayers", {})

    raw_map = map_info.get("name", "de_mirage").replace("de_", "").lower()
    model_map = raw_map if raw_map in SUPPORTED_MAPS else "mirage"

    ct_score = int(map_info.get("team_ct", {}).get("score", 0))
    t_score = int(map_info.get("team_t", {}).get("score", 0))

    bomb_state = bomb_info.get("state") or round_info.get("bomb") or ""
    phase = phase_cd.get("phase", "")
    bomb_planted = bomb_state in ("planted", "defusing") or phase in ("bomb", "defuse")
    is_defusing = (bomb_state == "defusing") or (phase == "defuse")

    if bomb_planted and bomb_info.get("countdown") is not None:
        raw_time = float(bomb_info["countdown"])
    else:
        raw_time = float(phase_cd.get("phase_ends_in", 30.0 if bomb_planted else 75.0))

    max_time = 40.0 if bomb_planted else 115.0
    time_left = max(0.0, min(max_time, round(raw_time, 1)))

    ct_players = t_players = 0
    ct_health = t_health = 0
    ct_armor = t_armor = 0
    ct_money = t_money = 0
    ct_kits = 0
    ct_rifles = t_rifles = 0
    ct_awps = t_awps = 0
    ct_utility = t_utility = 0
    full_telemetry = bool(all_players)

    if full_telemetry:
        for p_data in all_players.values():
            team = p_data.get("team")
            state = p_data.get("state", {})
            hp = int(state.get("health", 0))
            money = int(state.get("money", 0))
            armor = int(state.get("armor", 0))
            r_cnt, a_cnt, u_cnt = extract_gear_counts(p_data.get("weapons", {}))

            if team == "CT":
                ct_money += money
                if hp > 0:
                    ct_players += 1
                    ct_health += hp
                    ct_armor += armor
                    ct_rifles += r_cnt
                    ct_awps += a_cnt
                    ct_utility += u_cnt
                    if state.get("defusekit"):
                        ct_kits += 1
            elif team == "T":
                t_money += money
                if hp > 0:
                    t_players += 1
                    t_health += hp
                    t_armor += armor
                    t_rifles += r_cnt
                    t_awps += a_cnt
                    t_utility += u_cnt

    return {
        "timestamp": time.time(),
        "raw_map": raw_map.upper(),
        "map_name": model_map,
        "ct_score": min(12, ct_score),
        "t_score": min(12, t_score),
        "bomb_planted": bomb_planted,
        "is_defusing": is_defusing,
        "time_left": time_left,
        "ct_players": ct_players,
        "t_players": t_players,
        "ct_health": ct_health,
        "t_health": t_health,
        "ct_armor": ct_armor,
        "t_armor": t_armor,
        "ct_money": ct_money,
        "t_money": t_money,
        "ct_kits": ct_kits,
        "ct_rifles": ct_rifles,
        "t_rifles": t_rifles,
        "ct_awps": ct_awps,
        "t_awps": t_awps,
        "ct_utility": ct_utility,
        "t_utility": t_utility,
        "full_telemetry": full_telemetry,
    }


class GSIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_response(200)
            self.end_headers()
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
            if "map" in payload:
                parsed = parse_gsi_payload(payload)
                TEMP_FILE.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                os.replace(TEMP_FILE, STATE_FILE)
        except Exception as exc:
            print(f"[GSI Error] {exc}")

        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), GSIHandler)
    print(f"CS2 GSI Listener active on http://{HOST}:{PORT}")
    print("Waiting for live CS2 match telemetry... (Press Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GSI listener.")
        server.server_close()


if __name__ == "__main__":
    main()