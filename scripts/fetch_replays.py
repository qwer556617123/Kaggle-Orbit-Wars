import argparse
import csv
import subprocess
from pathlib import Path


HISTORICAL_SUBMISSIONS = {
    "v6": "53262817",
    "v17": "53337723",
    "v18": "53339528",
    "v20a": "53511212",
    "v21": "53546090",
}


def _run_kaggle(args):
    cmd = ["kaggle"] + args
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def public_episode_ids(submission_id, limit):
    result = _run_kaggle([
        "competitions", "episodes", submission_id, "-v",
    ])
    rows = csv.DictReader(result.stdout.splitlines())
    ids = [
        row["id"] for row in rows
        if row.get("type") == "EpisodeType.EPISODE_TYPE_PUBLIC"
        and row.get("state") == "EpisodeState.COMPLETED"
    ]
    return ids[:limit]


def fetch_replay(episode_id, dest):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"episode-{episode_id}-replay.json"
    if path.exists():
        return False
    _run_kaggle([
        "competitions", "replay", episode_id, "-p", str(dest),
    ])
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=list(HISTORICAL_SUBMISSIONS))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", default="replays/historical")
    args = parser.parse_args()

    root = Path(args.out)
    for version in args.versions:
        submission_id = HISTORICAL_SUBMISSIONS.get(version, version)
        dest = root / version
        episode_ids = public_episode_ids(submission_id, args.limit)
        print(f"{version} {submission_id}: public={len(episode_ids)}")
        downloaded = 0
        for episode_id in episode_ids:
            if fetch_replay(episode_id, dest):
                downloaded += 1
        print(f"  downloaded={downloaded} cached={len(episode_ids) - downloaded}")


if __name__ == "__main__":
    main()
