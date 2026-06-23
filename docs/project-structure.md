# Project Structure

## Root

- `main.py`: final Kaggle agent entry point. Kaggle expects this file to export
  `agent(obs)`.
- `README.md`: public-facing overview and quick-start instructions.
- `.gitignore`: excludes local data, replays, submissions, notebooks, caches,
  and environment artifacts.

## Archive

- `archive/agents/`: selected historical agent files. The archive is curated,
  not exhaustive.
- `archive/README.md`: table explaining why each archived agent is kept.

## Docs

- `docs/experiments.md`: detailed chronological development notes, local gates,
  public scores, and rejected experiments.
- `docs/project-structure.md`: this file.
- `docs/security-audit.md`: cleanup and sensitive-information scan notes.

## Scripts

- `scripts/evaluate.py`: local match runner with starter/proxy/reference support.
- `scripts/smoke.py`: minimal environment smoke test.
- `scripts/analyze_replays.py`: replay diagnostics.
- `scripts/fetch_replays.py`: Kaggle replay downloader.
- `scripts/diagnose_match.py`: match-level diagnostic helpers.
- `scripts/train_policy_selector.py`: reproduces the small v40 selector tree.

## Excluded Local Artifacts

The following are intentionally not versioned:

- `references/`: downloaded public notebooks and copied external material.
- `replays/` and `logs/`: Kaggle replay/log data.
- `build/`, `submissions/`, `*.tar.gz`, `*.zip`: local packaging output.
- `__pycache__/`, `.ipynb_checkpoints/`, virtual environments, IDE files.
