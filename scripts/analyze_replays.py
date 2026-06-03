import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="replays/v16")
    args = parser.parse_args()

    files = sorted(Path(args.path).glob("episode-*-replay.json"))
    print(f"files={len(files)}")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        rewards = data.get("rewards") or []
        statuses = data.get("statuses") or []
        info = data.get("info") or {}
        teams = info.get("TeamNames") or []
        agents = info.get("Agents") or []
        ranking = sorted(
            range(len(rewards)),
            key=lambda i: rewards[i] if rewards[i] is not None else -999999,
            reverse=True,
        )

        print(f"\nEP {data.get('id')} seed={info.get('seed')} steps={len(data.get('steps') or [])}")
        for i, reward in enumerate(rewards):
            rank = ranking.index(i) + 1
            team = teams[i] if i < len(teams) else "?"
            agent = agents[i] if i < len(agents) else ""
            status = statuses[i] if i < len(statuses) else None
            print(
                f"  slot={i} rank={rank} reward={reward} "
                f"status={status} team={team} agent={agent}"
            )


if __name__ == "__main__":
    main()
