import argparse
import importlib.util
import statistics
import time
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]


def load_agent():
    spec = importlib.util.spec_from_file_location("orbit_agent", ROOT / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reset_agent_state(mod):
    if hasattr(mod, "_turn"):
        mod._turn = 0
    if hasattr(mod, "_turns_no_attack"):
        mod._turns_no_attack = 0


def run_match(mod, opponents, seed):
    reset_agent_state(mod)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--opponents", nargs="+", default=["random", "starter"])
    parser.add_argument("--players", type=int, choices=[2, 4], default=2)
    args = parser.parse_args()

    mod = load_agent()
    for opponent in args.opponents:
        opponents = [opponent] * (args.players - 1)
        started = time.perf_counter()
        rows = [run_match(mod, opponents, seed) for seed in range(args.seeds)]
        wins = sum(row["won"] for row in rows)
        firsts = sum(row["first"] for row in rows)
        losses = [row for row in rows if not row["won"]]
        avg_us = statistics.mean(row["us"] for row in rows)
        avg_them = statistics.mean(row["them"] for row in rows)

        print(
            f"{opponent} x{args.players - 1}: {wins}/{len(rows)} wins "
            f"firsts={firsts}/{len(rows)} "
            f"avg_us={avg_us:.1f} avg_opp={avg_them:.1f} "
            f"time={time.perf_counter() - started:.1f}s"
        )
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


if __name__ == "__main__":
    main()
