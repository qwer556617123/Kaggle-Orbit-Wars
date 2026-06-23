import argparse
import importlib.util
import math
import os
import statistics
import sys
import time
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


ROOT = Path(__file__).resolve().parents[1]
SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0


def load_agent(path):
    spec = importlib.util.spec_from_file_location("orbit_agent", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def reset_agent_state(mod):
    resets = {
        "_turn": 0,
        "_turns_no_attack": 0,
        "_agent_step": 0,
        "_hammer_plan": None,
        "_planet_idle_counts": {},
        "_promoted_stockpiles": set(),
        "_game_num_players": None,
        "_2p_patient_streak": 0,
        "_2p_prod_share_history": [],
        "_pending_commitments": [],
        "_opp_profile": {},
    }
    for name, value in resets.items():
        if hasattr(mod, name):
            setattr(mod, name, value.copy() if hasattr(value, "copy") else value)
    for name in (
        "_neutral_prev_ships",
        "_neutral_wounded",
        "_enemy_prev_ships",
        "_enemy_recently_launched",
        "_planet_prev_owner",
        "_freshly_lost_planets",
        "_freshly_captured_planets",
        "_planet_capture_age",
    ):
        obj = getattr(mod, name, None)
        if hasattr(obj, "clear"):
            obj.clear()
    runtime = getattr(mod, "_RUNTIME", None)
    if runtime is not None and hasattr(runtime, "reset"):
        runtime.reset()


def _g(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + 5.0 * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


def _hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx * dx + dy * dy
    if a < 1e-9:
        return math.hypot(fx, fy) < SUN_R
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_R * SUN_R
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 < t1 < 1.0) or (0.0 < t2 < 1.0) or (t1 <= 0.0 and t2 >= 1.0)


def _basic_plan(obs, style):
    player = _g(obs, "player", 0)
    step = _g(obs, "step", 0)
    planets = [Planet(*p) for p in _g(obs, "planets", [])]
    fleets = _g(obs, "fleets", [])
    my_planets = [p for p in planets if p.owner == player]
    neutral = [p for p in planets if p.owner == -1]
    enemies = [p for p in planets if p.owner not in (-1, player)]
    if not my_planets:
        return []

    moves = []
    incoming_to_neutral = {
        f[5] for f in fleets if f[1] not in (-1, player) and f[6] >= 8
    }

    for src in sorted(my_planets, key=lambda p: p.ships, reverse=True):
        reserve = max(5, int(src.ships * (0.18 if style == "heavy" else 0.10)))
        avail = src.ships - reserve
        if avail < 6:
            continue

        targets = []
        if style in ("fast", "sniper") and neutral:
            for t in neutral:
                dist = max(math.hypot(src.x - t.x, src.y - t.y), 1.0)
                score = (t.production ** (1.8 if style == "fast" else 1.5)) / (dist * max(t.ships, 1))
                if style == "sniper" and t.id in incoming_to_neutral:
                    score *= 3.0
                targets.append((score, t, t.ships + 2))
        if style == "turtle" and neutral and step < 95:
            for t in neutral:
                if t.production >= 4:
                    dist = max(math.hypot(src.x - t.x, src.y - t.y), 1.0)
                    targets.append((t.production / (dist * max(t.ships, 1)), t, t.ships + 2))
        if style in ("heavy", "turtle") and enemies and (style == "heavy" or step >= 95):
            for t in enemies:
                dist = max(math.hypot(src.x - t.x, src.y - t.y), 1.0)
                need = t.ships + int(t.production * dist / max(_speed(avail), 1)) + 8
                targets.append(((t.production + 2) / max(dist, 1), t, need))
        if not targets and neutral:
            t = min(neutral, key=lambda n: math.hypot(src.x - n.x, src.y - n.y))
            targets.append((1.0, t, t.ships + 2))
        if not targets:
            continue

        _, target, needed = max(targets, key=lambda x: x[0])
        ships = min(avail, max(6, int(needed)))
        if ships <= target.ships and target.owner == -1:
            continue
        if style == "heavy" and target.owner not in (-1, player):
            ships = min(avail, max(ships, int(avail * 0.75)))
        if style == "sniper" and target.owner == -1 and target.production < 3:
            ships = min(avail, target.ships + 1)
        angle = math.atan2(target.y - src.y, target.x - src.x)
        if _hits_sun(src.x, src.y, target.x, target.y):
            continue
        moves.append([src.id, angle, ships])
        if len(moves) >= (3 if style == "fast" else 2):
            break
    return moves


def proxy_fast_expander(obs):
    return _basic_plan(obs, "fast")


def proxy_heavy_attacker(obs):
    return _basic_plan(obs, "heavy")


def proxy_sniper_retaker(obs):
    return _basic_plan(obs, "sniper")


def proxy_turtler(obs):
    return _basic_plan(obs, "turtle")


PROXIES = {
    "fast-expander": proxy_fast_expander,
    "heavy-attacker": proxy_heavy_attacker,
    "sniper-retaker": proxy_sniper_retaker,
    "turtler": proxy_turtler,
}

SUITES = {
    "starter": ["starter"],
    "public-proxy": list(PROXIES),
    "reference": [],
}


_OPPONENT_MODULES = {}


def opponent_from_name(name):
    if name in PROXIES:
        return PROXIES[name]
    path = Path(name)
    if path.exists() and path.suffix == ".py":
        key = str(path.resolve())
        if key not in _OPPONENT_MODULES:
            spec = importlib.util.spec_from_file_location(
                f"opponent_{len(_OPPONENT_MODULES)}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            _OPPONENT_MODULES[key] = mod
        return _OPPONENT_MODULES[key].agent
    return name


def run_match(mod, opponents, seed):
    reset_agent_state(mod)
    for opp_mod in _OPPONENT_MODULES.values():
        reset_agent_state(opp_mod)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([mod.agent] + opponents)
    final = env.steps[-1]
    rewards = [state.reward or 0 for state in final]
    us = rewards[0]
    best_opp = max(rewards[1:], default=0)
    return {
        "seed": seed,
        "us": us,
        "them": best_opp,
        "rewards": rewards,
        "won": us > best_opp,
        "first": us == max(rewards),
        "steps": len(env.steps),
    }


def run_group(mod, opponent_name, players, seeds):
    opponents = [opponent_from_name(opponent_name)] * (players - 1)
    rows = [run_match(mod, opponents, seed) for seed in range(seeds)]
    wins = sum(row["won"] for row in rows)
    firsts = sum(row["first"] for row in rows)
    losses = [row for row in rows if not row["won"]]
    avg_us = statistics.mean(row["us"] for row in rows)
    avg_them = statistics.mean(row["them"] for row in rows)
    return rows, {
        "opponent": opponent_name,
        "players": players,
        "wins": wins,
        "firsts": firsts,
        "total": len(rows),
        "avg_us": avg_us,
        "avg_them": avg_them,
        "losses": losses,
    }


def print_result(summary, elapsed):
    print(
        f"{summary['opponent']} x{summary['players'] - 1}: "
        f"{summary['wins']}/{summary['total']} wins "
        f"firsts={summary['firsts']}/{summary['total']} "
        f"avg_us={summary['avg_us']:.1f} avg_opp={summary['avg_them']:.1f} "
        f"time={elapsed:.1f}s"
    )
    losses = summary["losses"]
    if losses:
        print(
            "  losses: "
            + ", ".join(
                f"seed{row['seed']} us={row['us']} opp={row['them']} "
                f"rewards={row['rewards']} steps={row['steps']}"
                for row in losses[:12]
            )
        )
    else:
        print("  losses: none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=str(ROOT / "main.py"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--opponents", nargs="+", default=None)
    parser.add_argument("--players", type=int, choices=[2, 4], default=2)
    parser.add_argument("--suite", choices=sorted(SUITES), default=None)
    parser.add_argument("--disable-style", action="store_true")
    parser.add_argument("--trace-style", action="store_true")
    parser.add_argument("--env", action="append", default=[],
                        help="Set KEY=VALUE before loading the evaluated agent.")
    args = parser.parse_args()

    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        os.environ[key] = value

    mod = load_agent(Path(args.agent))
    if args.disable_style and hasattr(mod, "_opponent_style_2p"):
        mod._opponent_style_2p = lambda *a, **k: "UNKNOWN"
    style_counts = {}
    if args.trace_style and hasattr(mod, "_opponent_style_2p"):
        original_classifier = mod._opponent_style_2p

        def traced_classifier(*classifier_args, **classifier_kwargs):
            style = original_classifier(*classifier_args, **classifier_kwargs)
            style_counts[style] = style_counts.get(style, 0) + 1
            return style

        mod._opponent_style_2p = traced_classifier
    opponents = args.opponents
    if args.suite:
        opponents = SUITES[args.suite]
        if args.suite == "reference" and not opponents:
            print("reference suite: no local reference agents configured")
            return
    if opponents is None:
        opponents = ["random", "starter"]

    for opponent in opponents:
        started = time.perf_counter()
        _, summary = run_group(mod, opponent, args.players, args.seeds)
        print_result(summary, time.perf_counter() - started)
        if args.trace_style:
            print(
                "  styles: "
                + ", ".join(f"{k}={v}" for k, v in sorted(style_counts.items()))
            )
            style_counts.clear()


if __name__ == "__main__":
    main()
