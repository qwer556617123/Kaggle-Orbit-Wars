# Kaggle Orbit Wars

This repository contains the final local workspace for the Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars/overview) competition. The root `main.py` is the best-settled submitted agent kept as the default runnable entry point.

## Final Status

Best public submissions observed before cleanup:

| Version | Submission | Public score | Notes |
| --- | ---: | ---: | --- |
| v38a | `53907609` | `1107.3` | Final default `main.py`; Multi-Focus derived baseline. |
| v40c | `53943293` | `1032.7` | Producer Hybrid / "I'm Stronger" challenger. |
| v39d | `53913691` | `1021.5` | Light Intruder dynamic baseline. |
| v40f | `53952907` | `915.7` | Single-file decision-tree selector experiment. |

The final closing-window re-submissions were:

| Submission | Message |
| ---: | --- |
| `53983805` | `final slot v38a multifocus best baseline` |
| `53983816` | `final slot v40c im-stronger challenger` |

## Repository Layout

```text
.
├── main.py                 # Kaggle entry point; exports agent(obs)
├── archive/
│   ├── README.md           # Notes on preserved historical agents
│   └── agents/             # Selected archived versions only
├── docs/
│   ├── experiments.md      # Long-form version history and outcomes
│   ├── project-structure.md
│   └── security-audit.md
└── scripts/
    ├── evaluate.py         # Local evaluation harness
    ├── smoke.py            # Minimal smoke runner
    ├── analyze_replays.py  # Replay analysis utilities
    ├── fetch_replays.py    # Kaggle replay downloader
    └── train_policy_selector.py
```

Large local-only materials such as downloaded public notebooks, replays,
submission archives, build directories, and Python caches are intentionally
excluded from git.

## Quick Start

Use the `kaggle-dev` conda environment used during development:

```powershell
conda run -n kaggle-dev python -m py_compile main.py scripts\evaluate.py
conda run -n kaggle-dev python scripts\evaluate.py --agent main.py --suite starter --players 2 --seeds 10
conda run -n kaggle-dev python scripts\evaluate.py --agent main.py --suite starter --players 4 --seeds 10
```

Submit the root agent directly:

```powershell
conda run -n kaggle-dev kaggle competitions submit orbit-wars -f main.py -m "message"
```

## Attribution

The strongest final agents are public-code-derived Kaggle competition agents.
The archived names and docs distinguish self-developed experiments from public
reference or public-derived lines. See [docs/experiments.md](docs/experiments.md)
for the full development log and score history.

## Public Repository Notes

Before pushing, this repository was cleaned to remove:

- downloaded public notebook folders;
- local replays, logs, build output, and submission archives;
- local Copilot/agent workflow configuration;
- machine-specific paths and credential-like material.

See [docs/security-audit.md](docs/security-audit.md) for the latest scan notes.
