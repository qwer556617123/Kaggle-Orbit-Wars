import argparse
import csv
import json
import statistics
from pathlib import Path


SNAPSHOTS = (40, 80, 120, 160)


def _obs(step, slot):
    try:
        return step[slot].get("observation") or {}
    except (IndexError, AttributeError):
        return {}


def _team_name(info, slot):
    teams = info.get("TeamNames") or []
    agents = info.get("Agents") or []
    if slot < len(teams):
        return teams[slot]
    if slot < len(agents):
        agent = agents[slot]
        if isinstance(agent, dict):
            return agent.get("Name") or "?"
        return str(agent)
    return "?"


def _find_our_slot(info, slots, team_hint):
    hint = team_hint.lower()
    for i in range(slots):
        if hint and hint in _team_name(info, i).lower():
            return i
    return slots - 1 if slots else 0


def _rank(rewards, slot):
    order = sorted(
        range(len(rewards)),
        key=lambda i: rewards[i] if rewards[i] is not None else -999999,
        reverse=True,
    )
    return order.index(slot) + 1 if slot in order else None


def _slot_metrics(obs, slot, comet_ids):
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    mine = [p for p in planets if p[1] == slot]
    enemies = [p for p in planets if p[1] not in (-1, slot)]
    neutral = [p for p in planets if p[1] == -1 and p[0] not in comet_ids]
    my_fleet = sum(f[6] for f in fleets if f[1] == slot)
    enemy_fleet = sum(f[6] for f in fleets if f[1] not in (-1, slot))
    my_garrison = sum(p[5] for p in mine)
    enemy_garrison = sum(p[5] for p in enemies)
    enemy_launch_sources = {
        f[5] for f in fleets if f[1] not in (-1, slot) and f[6] >= 10
    }
    enemy_large = sum(1 for f in fleets if f[1] not in (-1, slot) and f[6] >= 28)
    return {
        "planets": len(mine),
        "prod": sum(p[6] for p in mine),
        "garrison": my_garrison,
        "fleet_mass": my_fleet,
        "enemy_planets": len(enemies),
        "enemy_prod": sum(p[6] for p in enemies),
        "enemy_garrison": enemy_garrison,
        "enemy_fleet_mass": enemy_fleet,
        "neutral_left": len(neutral),
        "enemy_launch_tempo": len(enemy_launch_sources),
        "enemy_large_fleets": enemy_large,
        "transit_pressure": my_fleet / max(my_garrison, 1),
        "source_drain": my_fleet / max(my_garrison + my_fleet, 1),
    }


def _taxonomy(row):
    r = row["rank"]
    t40_prod = row.get("t40_prod", 0)
    t80_prod = row.get("t80_prod", 0)
    t80_enemy_prod = row.get("t80_enemy_prod", 0)
    t120_prod = row.get("t120_prod", 0)
    t120_enemy_prod = row.get("t120_enemy_prod", 0)
    t80_transit = row.get("t80_transit_pressure", 0)
    t120_transit = row.get("t120_transit_pressure", 0)
    t80_garrison = row.get("t80_garrison", 0)
    t120_garrison = row.get("t120_garrison", 0)
    t80_neutral = row.get("t80_neutral_left", 0)
    t120_enemy_fleet = row.get("t120_enemy_fleet_mass", 0)
    t120_enemy_garrison = row.get("t120_enemy_garrison", 0)

    if row.get("early_wipe"):
        return "early_expansion_loss"
    if r and r > 1 and t80_enemy_prod > max(t80_prod * 1.35, t40_prod + 8):
        return "enemy_snowball"
    if r and r > 1 and max(t80_transit, t120_transit) > 1.45 and min(t80_garrison, t120_garrison) < 85:
        return "overextension"
    if r and r > 1 and t120_enemy_fleet > t120_enemy_garrison * 0.85:
        return "failed_focus_fire"
    if row.get("players", 2) >= 4 and r and r > 1 and t120_enemy_prod > max(t120_prod * 1.25, 1):
        return "dogpile"
    if r and r > 1 and t80_neutral >= 6 and row.get("t80_fleet_mass", 0) < 35:
        return "idle_too_passive"
    return "win_or_unclear" if r == 1 else "unclear_loss"


def analyze_file(path, version, team_hint, slot_override=None):
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps") or []
    info = data.get("info") or {}
    rewards = data.get("rewards") or []
    statuses = data.get("statuses") or []
    players = len(steps[0]) if steps else len(rewards)
    our_slot = slot_override if slot_override is not None else _find_our_slot(info, players, team_hint)
    opponents = [
        _team_name(info, i) for i in range(players) if i != our_slot
    ]
    row = {
        "version": version or path.parent.name,
        "episode": info.get("EpisodeId") or data.get("id") or path.stem,
        "seed": info.get("seed"),
        "players": players,
        "slot": our_slot,
        "team": _team_name(info, our_slot),
        "opponents": "|".join(opponents),
        "reward": rewards[our_slot] if our_slot < len(rewards) else None,
        "rank": _rank(rewards, our_slot),
        "status": statuses[our_slot] if our_slot < len(statuses) else None,
        "steps": len(steps),
    }

    early_wipe = False
    for snap in SNAPSHOTS:
        idx = min(snap, max(len(steps) - 1, 0))
        obs = _obs(steps[idx], our_slot)
        comet_ids = set(obs.get("comet_planet_ids") or [])
        metrics = _slot_metrics(obs, our_slot, comet_ids)
        if snap <= 160 and metrics["planets"] == 0:
            early_wipe = True
        for key, value in metrics.items():
            row[f"t{snap}_{key}"] = round(value, 4) if isinstance(value, float) else value

    row["early_wipe"] = early_wipe or (
        row.get("status") not in (None, "DONE") and row.get("steps", 999) < 250
    )
    row["taxonomy"] = _taxonomy(row)
    return row


def collect(paths, team_hint, slot_override=None):
    rows = []
    for root in paths:
        root = Path(root)
        files = sorted(root.rglob("episode-*-replay.json"))
        if root.is_file():
            files = [root]
        for path in files:
            try:
                version = path.parent.name
                rows.append(analyze_file(path, version, team_hint, slot_override=slot_override))
            except Exception as exc:
                print(f"skip {path}: {exc}")
    return rows


def write_csv(rows, out_path):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print(f"files={len(rows)}")
    by_version = {}
    for row in rows:
        by_version.setdefault(row["version"], []).append(row)
    for version, version_rows in sorted(by_version.items()):
        ranks = [row["rank"] for row in version_rows if row["rank"] is not None]
        wins = sum(1 for row in version_rows if row["rank"] == 1)
        avg_rank = statistics.mean(ranks) if ranks else 0
        tax = {}
        for row in version_rows:
            tax[row["taxonomy"]] = tax.get(row["taxonomy"], 0) + 1
        tax_text = ", ".join(f"{k}={v}" for k, v in sorted(tax.items()))
        print(
            f"{version}: wins={wins}/{len(version_rows)} "
            f"avg_rank={avg_rank:.2f} taxonomy: {tax_text}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["replays/historical"])
    parser.add_argument("--team", default="This cat is called Fumi")
    parser.add_argument("--slot", type=int, default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    rows = collect(args.paths, args.team, slot_override=args.slot)
    print_summary(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
