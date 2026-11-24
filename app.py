import base64
import json
import os
import tempfile
import time
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from demo_analyzer import analyze_demo_file
from predict import CS2_MAPS, CSGO_MAPS, MODEL_PATH, build_feature_row, compute_probabilities

st.set_page_config(page_title="CS2 / CS:GO Tactical Win Predictor", page_icon="🎯", layout="wide")

ASSETS_DIR = Path("assets")
LIVE_STATE_PATH = Path("live_state.json")

FEATURE_LABELS = {
    "diff_firepower": "Firepower Differential (Rifles + AWP)",
    "diff_armor": "Armor Differential (CT - T)",
    "diff_health": "Health Pool Differential",
    "diff_players": "Manpower Advantage (Alive)",
    "diff_utility": "Utility Differential (Grenades)",
    "diff_money": "Economy Differential (CT - T)",
    "ct_armor": "CT Total Armor",
    "t_armor": "T Total Armor",
    "ct_rifles": "CT Active Rifles",
    "t_rifles": "T Active Rifles",
    "ct_awps": "CT AWPs Active",
    "t_awps": "T AWPs Active",
    "ct_utility": "CT Grenade Utility",
    "t_utility": "T Grenade Utility",
    "ct_money": "CT Unspent Economy ($)",
    "t_money": "T Unspent Economy ($)",
    "ct_defuse_kits": "CT Defuse Kits Available",
    "t_score": "T Rounds Won (Momentum)",
    "ct_score": "CT Rounds Won (Momentum)",
    "plant_margin": "Pre-Plant Time Margin (3.2s)",
    "defuse_margin": "Post-Plant Defuse Margin",
    "time_left": "Round / C4 Clock Remaining",
}


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def init_session_state():
    defaults = {
        "game_mode": "CS2 (MR12)",
        "map_name": "MIRAGE",
        "bomb_planted": False,
        "time_left": 75.0,
        "ct_score": 6,
        "t_score": 6,
        "ct_money": 2500,
        "t_money": 2500,
        "ct_players": 5,
        "t_players": 5,
        "ct_health": 500,
        "t_health": 500,
        "ct_armor": 500,
        "t_armor": 500,
        "ct_rifles": 4,
        "t_rifles": 4,
        "ct_awps": 1,
        "t_awps": 1,
        "ct_utility": 12,
        "t_utility": 12,
        "ct_kits": 2,
        "is_defusing": False,
        "defuse_elapsed": 0.0,
        "gsi_auto_sync": False,
        "gsi_status": "Disconnected (Run gsi_listener.py)",
        "prob_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sync_from_gsi():
    if not LIVE_STATE_PATH.exists():
        st.session_state["gsi_status"] = "Waiting for CS2 packet (live_state.json not created yet)"
        return False

    try:
        data = json.loads(LIVE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False

    age = time.time() - data.get("timestamp", 0)
    if age > 15:
        st.session_state["gsi_status"] = f"Stale feed (last packet {int(age)}s ago)"
        return False

    st.session_state["game_mode"] = "CS2 (MR12)"
    st.session_state["map_name"] = data["map_name"].upper()
    st.session_state["bomb_planted"] = bool(data["bomb_planted"])
    st.session_state["time_left"] = float(data["time_left"])
    st.session_state["ct_score"] = int(data["ct_score"])
    st.session_state["t_score"] = int(data["t_score"])
    st.session_state["is_defusing"] = bool(data["is_defusing"])

    if data.get("full_telemetry", False):
        for field in (
            "ct_players", "t_players", "ct_health", "t_health",
            "ct_armor", "t_armor", "ct_money", "t_money", "ct_kits",
            "ct_rifles", "t_rifles", "ct_awps", "t_awps", "ct_utility", "t_utility"
        ):
            if field in data:
                st.session_state[field] = int(data[field])
        mode_tag = "Full Spectator / Demo Feed"
    else:
        mode_tag = "Live Player Feed (adjust enemy loadout manually)"

    st.session_state["gsi_status"] = f"Connected • {data['raw_map']} ({mode_tag})"
    return True


def get_bg_css():
    for ext in ("jpg", "jpeg", "png", "webp"):
        img_path = ASSETS_DIR / f"background.{ext}"
        if img_path.exists():
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            encoded = base64.b64encode(img_path.read_bytes()).decode()
            return (
                f".stApp {{ background-image: linear-gradient(180deg, rgba(8, 12, 20, 0.75), rgba(8, 12, 20, 0.90)), "
                f"url('data:image/{mime};base64,{encoded}'); background-size: cover; background-position: center; background-attachment: fixed; }}"
            )
    return ".stApp { background: radial-gradient(circle at top, #162033 0%, #080c14 100%); }"


def apply_theme():
    css_path = ASSETS_DIR / "style.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    st.markdown(f"<style>{css}\n{get_bg_css()}</style>", unsafe_allow_html=True)


def render_svg_timeline(points, height=200, grad_id="ctSwingGrad"):
    w, h = 960, height
    pad_x, pad_y = 44, 22
    plot_w = w - 2 * pad_x
    plot_h = h - 2 * pad_y
    n = len(points)

    coords = []
    for i, pt in enumerate(points):
        x = pad_x + (i / (n - 1)) * plot_w
        y = pad_y + ((100.0 - pt["ct_pct"]) / 100.0) * plot_h
        coords.append((x, y, pt))

    line_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
    area_pts = f"{coords[0][0]:.1f},{pad_y + plot_h:.1f} {line_pts} {coords[-1][0]:.1f},{pad_y + plot_h:.1f}"
    mid_y = pad_y + plot_h * 0.5

    dots_svg = []
    step = max(1, n // 90)
    for idx, (x, y, pt) in enumerate(coords):
        if idx % step != 0 and idx != n - 1:
            continue
        color = "#38BDF8" if pt["ct_pct"] >= 50.0 else "#F59E0B"
        r = "4.5" if n <= 40 else "3.0"
        dots_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="#0F172A" stroke-width="1.2">'
            f'<title>{pt["label"]}</title></circle>'
        )

    last_x, last_y, last_pt = coords[-1]
    return (
        f'<svg viewBox="0 0 {w} {h}" style="width: 100%; height: {height}px; overflow: visible;">'
        f'<defs><linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#38BDF8" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="#38BDF8" stop-opacity="0.0"/>'
        f'</linearGradient></defs>'
        f'<line x1="{pad_x}" y1="{pad_y}" x2="{w - pad_x}" y2="{pad_y}" stroke="rgba(255,255,255,0.07)" stroke-dasharray="4"/>'
        f'<line x1="{pad_x}" y1="{mid_y:.1f}" x2="{w - pad_x}" y2="{mid_y:.1f}" stroke="rgba(255,255,255,0.22)" stroke-dasharray="6"/>'
        f'<line x1="{pad_x}" y1="{pad_y + plot_h}" x2="{w - pad_x}" y2="{pad_y + plot_h}" stroke="rgba(255,255,255,0.07)" stroke-dasharray="4"/>'
        f'<text x="6" y="{pad_y + 4}" fill="#64748B" font-size="11" font-family="JetBrains Mono, monospace">100%</text>'
        f'<text x="12" y="{mid_y + 4:.1f}" fill="#94A3B8" font-size="11" font-family="JetBrains Mono, monospace">50%</text>'
        f'<text x="18" y="{pad_y + plot_h + 4}" fill="#64748B" font-size="11" font-family="JetBrains Mono, monospace">0%</text>'
        f'<polygon points="{area_pts}" fill="url(#{grad_id})"/>'
        f'<polyline fill="none" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{line_pts}"/>'
        f'{"".join(dots_svg)}'
        f'<text x="{last_x:.1f}" y="{max(14.0, last_y - 10):.1f}" text-anchor="middle" fill="#FFFFFF" '
        f'font-size="11" font-weight="700" font-family="JetBrains Mono, monospace">{last_pt["ct_pct"]:.1f}%</text>'
        f'</svg>'
    )


def record_swing_point(ct_pct, bomb_planted, time_left, ct_players, t_players, ct_health, t_health, is_defusing):
    history = st.session_state["prob_history"]
    state_sig = (bomb_planted, ct_players, t_players, ct_health, t_health, is_defusing, round(ct_pct, 1))

    if history:
        last = history[-1]
        if last["sig"] == state_sig and abs(last["time_left"] - time_left) < 4.0:
            return

    phase_tag = "C4" if bomb_planted else "RND"
    label = f"{phase_tag} {time_left:.0f}s | {ct_players}v{t_players} -> CT {ct_pct:.1f}%"
    history.append({"sig": state_sig, "time_left": time_left, "ct_pct": round(ct_pct, 1), "label": label})
    if len(history) > 25:
        st.session_state["prob_history"] = history[-25:]


def render_tutorial():
    with st.expander("📖 How to Use & Feature Guide (English Tutorial)", expanded=False):
        t_col1, t_col2 = st.columns(2, gap="large")
        with t_col1:
            st.markdown("#### Quick Start Workflow")
            st.markdown(
                "1. **Live & Tactical Tab:** Adjust round sliders manually or enable **Auto-Sync CS2 Live** while `gsi_listener.py` is running.\n"
                "2. **Configure Loadouts:** Set surviving players, HP, armor, active rifles/AWPs, grenades, and unspent bank.\n"
                "3. **Simulate Retakes:** Toggle **C4 Bomb Planted** and **CT Actively Defusing C4** to test exact timer cutoffs.\n"
                "4. **Demo Replay Tab:** Upload any CS2 `.dem` match file to chart win probabilities across the entire game."
            )
            st.markdown("#### Hybrid Engine Logic")
            st.markdown(
                "- **Random Forest Core:** Weighs firepower (`Rifles + AWP`), armor, utility, HP, and map win-rate biases.\n"
                "- **Pre-Plant Crunch (< 3.2s):** Planting C4 takes **3.2 seconds**; below 3.2s, Terrorists can only win via full elimination.\n"
                "- **Post-Plant Guardrail:** Defusing takes **5.0s** with a kit or **10.0s** without; impossible defuses lock CT odds to **0.0%**."
            )
        with t_col2:
            st.markdown("#### Role of Each Control")
            st.markdown(
                "- **Rifles & AWPs:** Captures true combat firepower (e.g., full buy with $0 bank vs. pistol eco).\n"
                "- **Armor & Utility:** Accounts for Kevlar protection and smoke/flash/molotov site control.\n"
                "- **CS2 Live GSI Sync:** Reads real-time JSON telemetry directly from Counter-Strike 2.\n"
                "- **Match Format:** Scales CS2 MR12 scores (first to 13) to match the MR15 training dataset.\n"
                "- **CT Defuse Kits & Progress:** Cuts required defuse time in half (10s to 5s) and tracks live defuse completion.\n"
                "- **`.dem` Replay Analyzer:** Extracts tick-level telemetry from match demos to highlight the biggest clutches and comebacks."
            )


def render_telemetry_chart(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    max_val = float(importances.max()) if importances.max() > 0 else 1.0

    rows_html = ['<div class="telemetry-grid">']
    for idx, (feat, val) in enumerate(importances.items(), start=1):
        label = FEATURE_LABELS.get(feat, f"Map: {feat.replace('map_de_', '').upper()}")
        bar_width = (float(val) / max_val) * 100.0
        delay = idx * 0.05
        rows_html.append(
            f'<div class="telemetry-row" title="Feature column: {feat}">'
            f'<span class="telemetry-rank">#{idx:02d}</span>'
            f'<span class="telemetry-label">{label}</span>'
            f'<div class="telemetry-track">'
            f'<div class="telemetry-fill" style="width: {bar_width:.1f}%; animation-delay: {delay:.2f}s;"></div>'
            f'</div>'
            f'<span class="telemetry-val">{val * 100:.1f}%</span>'
            f'</div>'
        )
    rows_html.append("</div>")

    with st.expander("📊 Model Telemetry & Feature Importance (Top 10 Decision Weights)", expanded=False):
        st.caption("Live feature weights extracted directly from the trained Random Forest ensemble.")
        st.markdown("".join(rows_html), unsafe_allow_html=True)


def render_demo_tab(model, feature_cols):
    with st.container(border=True):
        st.markdown(
            '<div style="font-family: Rajdhani, sans-serif; font-size: 1.4rem; font-weight: 700; '
            'letter-spacing: 1.5px; color: #38BDF8; text-transform: uppercase;">'
            '🎬 CS2 Match Replay Analyzer (.dem)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Upload a CS2 competitive demo (.dem) from Steam/FACEIT to reconstruct round-by-round win probabilities, "
            "momentum swings, and clutch comebacks."
        )
        uploaded_demo = st.file_uploader("Select CS2 Demo File (.dem)", type=["dem"])

    if uploaded_demo is None:
        return

    with st.spinner("Parsing CS2 demo ticks and computing round win probabilities..."):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dem") as tmp:
                tmp.write(uploaded_demo.getbuffer())
                tmp_path = tmp.name
            report = analyze_demo_file(tmp_path, model, feature_cols)
        except Exception as exc:
            st.error(f"Demo parsing failed: {exc}")
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    m1, m2, m3 = st.columns(3)
    m1.metric("Map", report["map_name"])
    m2.metric("Final Score", report["final_score"])
    m3.metric("Rounds Analyzed", report["total_rounds"])

    if len(report["timeline"]) >= 2:
        with st.container(border=True):
            st.markdown(
                '<div style="font-family: Rajdhani, sans-serif; font-size: 1.25rem; font-weight: 700; '
                'letter-spacing: 1.4px; color: #E2E8F0; text-transform: uppercase;">'
                '📈 Full-Match CT Win Probability Timeline</div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_svg_timeline(report["timeline"], height=220, grad_id="demoMatchGrad"), unsafe_allow_html=True)

    if report["top_swings"]:
        st.subheader("🔥 Biggest Comeback / Clutch Rounds")
        s_cols = st.columns(len(report["top_swings"]))
        for col, item in zip(s_cols, report["top_swings"]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**Round {item['Round']}** (`{item['Score Before']}`)")
                    st.markdown(f"Winner: **{item['Winner']}** &bull; Swing: **{item['Swing']:.1f}%**")
                    st.caption(f"CT Range: {item['Lowest CT %']} → {item['Peak CT %']} | C4: {item['C4 Planted']}")

    with st.container(border=True):
        st.subheader("Round-by-Round Breakdown")
        st.dataframe(pd.DataFrame(report["rounds_summary"]), use_container_width=True, hide_index=True)


init_session_state()
if st.session_state["gsi_auto_sync"]:
    sync_from_gsi()

apply_theme()
artifact = load_artifact()
model = artifact["model"]
feature_cols = artifact["columns"]

st.markdown(
    '<div class="hud-header">'
    '<h1 class="hud-title">Counter-Strike Round Win Predictor</h1>'
    '<div class="hud-subtitle"><span class="live-dot"></span> Hybrid ML Engine &bull; Real-Time Tactical Telemetry</div>'
    '</div>',
    unsafe_allow_html=True,
)

render_tutorial()

tab_live, tab_demo = st.tabs(["🎯 Live & Tactical Predictor", "🎬 CS2 Demo Replay Analyzer (.dem)"])

with tab_live:
    with st.container(border=True):
        g1, g2, g3 = st.columns([1.2, 1.0, 2.2])
        with g1:
            st.toggle("📡 Auto-Sync CS2 Live (1s)", key="gsi_auto_sync")
        with g2:
            if st.button("Sync Snapshot Now", use_container_width=True):
                sync_from_gsi()
        with g3:
            st.caption(f"GSI Status: **{st.session_state['gsi_status']}**")

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
        with c1:
            game_mode = st.radio("Match Format", ["CS2 (MR12)", "CS:GO (MR15)"], horizontal=True, key="game_mode")
        with c2:
            maps = [m.upper() for m in (CS2_MAPS if game_mode == "CS2 (MR12)" else CSGO_MAPS)]
            if st.session_state["map_name"] not in maps:
                st.session_state["map_name"] = maps[0]
            map_name = st.selectbox("Active Map", maps, key="map_name").lower()
        with c3:
            bomb_planted = st.toggle("C4 Bomb Planted", key="bomb_planted")
            pill_cls = "status-pill-planted" if bomb_planted else "status-pill-normal"
            pill_txt = "POST-PLANT ACTIVE (40s TIMER)" if bomb_planted else "PRE-PLANT PHASE (3.2s PLANT TIME)"
            st.markdown(f'<span class="{pill_cls}">{pill_txt}</span>', unsafe_allow_html=True)

        max_time = 40.0 if bomb_planted else 115.0
        st.session_state["time_left"] = max(0.0, min(max_time, float(st.session_state["time_left"])))
        time_left = st.slider("Round / C4 Timer Remaining (seconds)", 0.0, max_time, step=0.5, key="time_left")

    max_score = 12 if game_mode == "CS2 (MR12)" else 15
    st.session_state["ct_score"] = min(max_score, int(st.session_state["ct_score"]))
    st.session_state["t_score"] = min(max_score, int(st.session_state["t_score"]))

    col_ct, col_t = st.columns(2, gap="medium")

    with col_ct:
        with st.container(border=True):
            st.markdown('<div class="team-header-ct">Counter-Terrorists</div>', unsafe_allow_html=True)
            s_col, k_col = st.columns(2)
            with s_col:
                ct_score = st.number_input("CT Rounds Won", 0, max_score, key="ct_score")
            with k_col:
                ct_money = st.number_input("CT Bank ($)", 0, 80000, step=500, key="ct_money")

            ct_players = st.slider("CT Players Alive", 0, 5, key="ct_players")
            ct_health = st.slider("CT Total Health (HP)", 0, 500, key="ct_health")
            ct_armor = st.slider("CT Total Armor", 0, 500, key="ct_armor")

            w1, w2 = st.columns(2)
            with w1:
                ct_rifles = st.number_input("CT Rifles (M4/AUG/FAMAS)", 0, 5, key="ct_rifles")
                ct_kits = st.number_input("CT Defuse Kits", 0, 5, key="ct_kits")
            with w2:
                ct_awps = st.number_input("CT AWPs", 0, 5, key="ct_awps")
                ct_utility = st.number_input("CT Grenades (Utility)", 0, 20, key="ct_utility")

    with col_t:
        with st.container(border=True):
            st.markdown('<div class="team-header-t">Terrorists</div>', unsafe_allow_html=True)
            ts_col, tm_col = st.columns(2)
            with ts_col:
                t_score = st.number_input("T Rounds Won", 0, max_score, key="t_score")
            with tm_col:
                t_money = st.number_input("T Bank ($)", 0, 80000, step=500, key="t_money")

            t_players = st.slider("T Players Alive", 0, 5, key="t_players")
            t_health = st.slider("T Total Health (HP)", 0, 500, key="t_health")
            t_armor = st.slider("T Total Armor", 0, 500, key="t_armor")

            tw1, tw2 = st.columns(2)
            with tw1:
                t_rifles = st.number_input("T Rifles (AK/SG/Galil)", 0, 5, key="t_rifles")
            with tw2:
                t_awps = st.number_input("T AWPs", 0, 5, key="t_awps")
            t_utility = st.slider("T Grenades (Utility)", 0, 20, key="t_utility")

    is_defusing = False
    defuse_elapsed = 0.0
    if bomb_planted and ct_players > 0:
        with st.container(border=True):
            d1, d2 = st.columns([1, 2])
            with d1:
                is_defusing = st.toggle("CT Actively Defusing C4", key="is_defusing")
            with d2:
                if is_defusing:
                    max_elapsed = 5.0 if ct_kits > 0 else 10.0
                    st.session_state["defuse_elapsed"] = min(max_elapsed, float(st.session_state["defuse_elapsed"]))
                    defuse_elapsed = st.slider("Defuse Progress (seconds)", 0.0, max_elapsed, step=0.5, key="defuse_elapsed")
                else:
                    st.caption("Enable if a CT player is currently holding E on the bomb.")

    row, ctx = build_feature_row(
        feature_cols=feature_cols,
        map_name=map_name,
        time_left=time_left,
        bomb_planted=bomb_planted,
        ct_players=ct_players,
        t_players=t_players,
        ct_health=ct_health,
        t_health=t_health,
        ct_money=ct_money,
        t_money=t_money,
        ct_defuse_kits=ct_kits,
        ct_score=ct_score,
        t_score=t_score,
        is_defusing=is_defusing,
        defuse_elapsed=defuse_elapsed,
        game_mode=game_mode,
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
    ct_pct = probs["CT"] * 100
    t_pct = probs["T"] * 100

    record_swing_point(ct_pct, bomb_planted, time_left, ct_players, t_players, ct_health, t_health, is_defusing)

    st.markdown(
        f'<div class="prob-container">'
        f'<div class="prob-labels">'
        f'<div><span class="ct-score-text">CT {ct_pct:.1f}%</span></div>'
        f'<div class="vs-badge">Win Probability &bull; {map_name.upper()}</div>'
        f'<div><span class="t-score-text">{t_pct:.1f}% T</span></div>'
        f'</div>'
        f'<div class="dual-bar">'
        f'<div class="bar-ct" style="width: {ct_pct}%;"></div>'
        f'<div class="clash-spark" style="left: {ct_pct}%;"></div>'
        f'<div class="bar-t" style="width: {t_pct}%;"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    with st.container(border=True):
        h_col1, h_col2 = st.columns([4, 1])
        with h_col1:
            st.markdown(
                '<div style="font-family: Rajdhani, sans-serif; font-size: 1.25rem; font-weight: 700; '
                'letter-spacing: 1.4px; color: #E2E8F0; text-transform: uppercase;">'
                '📈 Live Round Win Probability Swing (CT %)</div>',
                unsafe_allow_html=True,
            )
        with h_col2:
            if st.button("Reset Timeline", use_container_width=True):
                st.session_state["prob_history"] = []
                st.rerun()

        if len(st.session_state["prob_history"]) >= 2:
            st.markdown(render_svg_timeline(st.session_state["prob_history"], height=190), unsafe_allow_html=True)
        else:
            st.caption("Adjust timers, kills, loadouts, or bomb status (or enable CS2 Live Sync) to plot round momentum swings.")

    render_telemetry_chart(model, feature_cols)

with tab_demo:
    render_demo_tab(model, feature_cols)

if st.session_state["gsi_auto_sync"]:
    time.sleep(1.0)
    st.rerun()