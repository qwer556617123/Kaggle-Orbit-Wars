"""
Print midgame transit and planet distribution diagnostics for selected seeds.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as M
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def get_attr(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)

def run_diag(seed, label):
    M._turn = 0
    if hasattr(M, "_turns_no_attack"):
        M._turns_no_attack = 0
    env = make('orbit_wars', configuration={'seed': seed}, debug=False)
    env.run([M.agent, 'starter'])
    final = env.steps[-1]
    result = 'WIN' if (final[0].reward or 0) > (final[1].reward or 0) else 'LOSS'
    print(f"  [{label}] seed={seed} -> {result}")
    for t in [65, 70, 75, 80, 85, 90]:
        if t >= len(env.steps): break
        obs0 = env.steps[t][0].observation
        if obs0 is None: continue
        planets  = [Planet(*p) for p in get_attr(obs0, 'planets', [])]
        fleets_r = get_attr(obs0, 'fleets', [])
        player   = get_attr(obs0, 'player', 0)
        my_p = [p for p in planets if p.owner == player]
        en_p = [p for p in planets if p.owner not in (-1, player)]
        ne_p = [p for p in planets if p.owner == -1]
        my_s  = sum(p.ships for p in my_p)
        en_s  = sum(p.ships for p in en_p)
        # ships distribution
        my_sorted = sorted(p.ships for p in my_p)
        en_fleets  = [f for f in fleets_r if f[1] != player]
        my_fleets  = [f for f in fleets_r if f[1] == player]
        en_t = sum(f[6] for f in en_fleets)
        my_t = sum(f[6] for f in my_fleets)
        ratio = my_s / max(en_s, 1)
        print(f"    T{t:3d}: mine={len(my_p)}p/{my_s:.0f}s en={len(en_p)}p/{en_s:.0f}s"
              f" neut={len(ne_p)} r={ratio:.2f}"
              f" my_trans={my_t:.0f} en_trans={en_t:.0f}"
              f" my_planet_dist={my_sorted}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="*", type=int, default=[19, 31])
    args = parser.parse_args()

    print("=== Transit diagnostics ===")
    for seed in args.seeds:
        run_diag(seed, "current")


if __name__ == "__main__":
    main()
