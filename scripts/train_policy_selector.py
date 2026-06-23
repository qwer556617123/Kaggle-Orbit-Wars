import importlib.util
import math
import sys
from pathlib import Path

from kaggle_environments import make
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def reset(mod):
    runtime = getattr(mod, "_RUNTIME", None)
    if runtime is not None and hasattr(runtime, "reset"):
        runtime.reset()
    for name in ("_turn", "_agent_step"):
        if hasattr(mod, name):
            setattr(mod, name, 0)


def features(obs):
    player = obs["player"]
    planets = obs["planets"]
    mine = [p for p in planets if p[1] == player]
    neutral = [p for p in planets if p[1] == -1]
    enemies = [p for p in planets if p[1] not in (-1, player)]
    home = mine[0] if mine else planets[0]
    hx, hy = home[2], home[3]

    def dist(p):
        return math.hypot(p[2] - hx, p[3] - hy)

    nearest = sorted(neutral, key=dist)
    high = [p for p in neutral if p[6] >= 4]
    static = [p for p in neutral if math.hypot(p[2] - 50.0, p[3] - 50.0) >= 50.0]
    rotating = [p for p in neutral if math.hypot(p[2] - 50.0, p[3] - 50.0) < 50.0]
    return [
        len(planets),
        len(neutral),
        len(high),
        len(static),
        len(rotating),
        sum(p[6] for p in neutral),
        sum(p[5] for p in neutral),
        nearest[0][6] if nearest else 0.0,
        nearest[0][5] if nearest else 0.0,
        dist(nearest[0]) if nearest else 999.0,
        sum(p[6] / max(dist(p), 1.0) for p in neutral),
        sum((p[6] ** 1.5) / max(dist(p), 1.0) for p in neutral),
        min((dist(p) for p in high), default=999.0),
        len(enemies),
        home[5],
        home[6],
        math.hypot(hx - 50.0, hy - 50.0),
    ]


def main():
    v39d = load(ROOT / "archive/agents/agent_v39d_light_intruder_dynamic.py", "v39d_train")
    v38a = load(ROOT / "archive/agents/agent_v38a_multifocus_best.py", "v38a_train")
    names = [
        "n_planets",
        "n_neutral",
        "n_high",
        "n_static",
        "n_rot",
        "neutral_prod",
        "neutral_ships",
        "near_prod",
        "near_ships",
        "near_dist",
        "prod_dist",
        "prod15_dist",
        "nearest_high_dist",
        "n_enemies",
        "home_ships",
        "home_prod",
        "home_r",
    ]

    X, y = [], []
    for seed in range(24):
        reset(v39d)
        reset(v38a)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        obs0 = env.steps[0][0].observation
        env.run([v39d.agent, v38a.agent])
        rewards = [state.reward or 0 for state in env.steps[-1]]
        label = 1 if rewards[0] > rewards[1] else 0
        X.append(features(obs0))
        y.append(label)
        print(
            f"seed {seed:02d} label={label} rewards={rewards} steps={len(env.steps)}",
            flush=True,
        )

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=7)
    clf.fit(X, y)
    print(f"positives {sum(y)}/{len(y)}")
    print(f"train_acc {clf.score(X, y):.3f}")
    print(export_text(clf, feature_names=names))
    print("feature", clf.tree_.feature.tolist())
    print("threshold", clf.tree_.threshold.tolist())
    print("left", clf.tree_.children_left.tolist())
    print("right", clf.tree_.children_right.tolist())
    print("value", clf.tree_.value.squeeze().tolist())


if __name__ == "__main__":
    main()
