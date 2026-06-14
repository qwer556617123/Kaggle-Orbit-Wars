import argparse
import importlib.util
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]


def load_agent(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reset(mod):
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


def summarize_obs(obs, player):
    planets = obs["planets"]
    fleets = obs["fleets"]
    owned = [p for p in planets if p[1] == player]
    garrison = sum(p[5] for p in owned)
    fleet = sum(f[6] for f in fleets if f[1] == player)
    total = garrison + fleet
    return {
        "planets": len(owned),
        "prod": sum(p[6] for p in owned),
        "garrison": garrison,
        "fleet": fleet,
        "in_flight_ratio": round(fleet / max(1, total), 3),
        "neutral_remaining": sum(1 for p in planets if p[1] == -1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--snapshots", type=int, nargs="+", default=[20, 40, 60, 80, 100, 120, 140])
    args = parser.parse_args()

    us = load_agent(ROOT / "main.py", "us_agent")
    opp = load_agent(ROOT / args.opponent, "opp_agent")

    for seed in args.seeds:
        reset(us)
        reset(opp)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([us.agent, opp.agent])
        print(f"SEED {seed} steps={len(env.steps)} final={[s.reward for s in env.steps[-1]]}")
        for turn in args.snapshots:
            if turn >= len(env.steps):
                break
            obs = env.steps[turn][0].observation
            left = summarize_obs(obs, 0)
            right = summarize_obs(obs, 1)
            print(f"  t{turn} us={left} opp={right}")


if __name__ == "__main__":
    main()
