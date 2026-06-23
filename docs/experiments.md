# Orbit Wars Agent Strategy Notes

This document tracks the current agent direction so code changes can be tied back
to validation results.

## Current Public Baseline

The active public baseline remains:

```text
v31 exp34 public reference variant: 948.6
```

Recent follow-up variants have converged below it:

```text
v35 production-ordered evaluator window: 897.5
v34 hybrid risk-punish strategy layer:   893.5
v33 exp34 value-weight 5.1:              850.4
```

That means the current problem is not "find a slightly better local reorder."
The problem is that our local gates still let through changes that disturb the
public pool's broader matchup mix. Replay-calibrated diagnosis is now the next
required step before another submission branch.

## v35 Replay Diagnosis

The first public replay comparison between `v31` and `v35` gave a useful
directional result even though the sample is still small:

```text
v31 sample taxonomy: early_expansion_loss=7, enemy_snowball=1,
  failed_focus_fire=1, unclear_loss=1, wins=2/12
v35 sample taxonomy: early_expansion_loss=5, enemy_snowball=4,
  wins=3/12
```

The important point is not the tiny win delta. It is the shape of the losses.
`v35` reduced some early wipes, but in 2-player games its turn-40 and turn-80
production was lower on average than `v31`, and those slower openings turned
into more enemy snowball losses later. The working interpretation is:

1. Globally reordering the 2P evaluator window by target production changed too
   many opening and early-mid candidate evaluations.
2. A future retry should keep the original local/short-ETA window as the base
   and only inject value-biased candidates in narrow pressure states.

## v36 Local Experiment

One follow-up local branch tested that narrower idea: keep `v31` as the base
window and only blend in a few value-biased 2P candidates when clearly behind
or when neutrals are nearly exhausted.

Local result:

```text
starter 2P: 40/40
starter 4P: 16/16
public-proxy 2P: 48/48
vs v25: 30/30
```

But the longer head-to-head gate failed:

```text
vs v31, 50 seeds: 23/50 wins, 28/50 firsts
vs v30, 30 seeds: 13/30 wins
```

So the branch is rejected locally and `main.py` should stay on the v31
baseline. The lesson is that even a narrower value-injection window still
damaged the stable 2P matchups too often once the sample was large enough.

## v36 Public Result: Opening Guard Rejected

v36 avoided touching the 2-player evaluator entirely. The replay sample for
`v31` suggested the more actionable weakness was 4-player early expansion
collapse, not 2-player tactical search.

Implementation idea:

1. Keep the `v31` 2P search and timing behavior untouched.
2. Add a 4P-only opening guard in `plan_solo_capture` for neutral captures.
3. When a neutral is both early and contested, reject low-production races that
   nearby enemies can immediately punish.
4. For the remaining contested openings, run a longer hold check before
   allowing the capture.

Current local readout:

```text
starter 2P: 20/20
starter 4P: 20/20
public-proxy 2P: 32/32
vs v31, 2P: 10/20 wins, 13/20 firsts
vs v31 x3, 4P with guard on:  12/24 wins, 15/32 wins
vs v31 x3, 4P with guard off: 11/24 wins, 14/32 wins
```

Public result:

```text
v36a opening contested neutral guard:      870.1
v36b opening contested neutral guard wide: 892.2
v31 exp34 public reference variant:        948.6
```

Conclusion:

1. `v36b > v36a`, so making the opening guard too strict clearly damages tempo.
2. Both are below `v31`, so the whole opening neutral guard idea is not strong
   enough as a public-score direction.
3. `main.py` should stay on v31 until the next candidate passes a stronger
   replay-driven gate.

Replay sample after convergence:

```text
v31:  wins=2/12, avg_rank=2.67,
      early_expansion_loss=7, enemy_snowball=1,
      failed_focus_fire=1, unclear_loss=1
v36a: wins=2/12, avg_rank=2.17,
      early_expansion_loss=7, enemy_snowball=3
v36b: wins=2/12, avg_rank=2.58,
      early_expansion_loss=9, enemy_snowball=1
```

The split by player count is more revealing:

```text
v31 4P sample:  t40 prod=12.25, t80 prod=9.00,  early=6/8
v36a 4P sample: t40 prod=12.75, t80 prod=16.12, early=4/8
v36b 4P sample: t40 prod=12.45, t80 prod=12.18, early=8/11
```

v36a did improve some 4P survival/production metrics in the sample, but its
public score still landed far below v31. That means the guard is not a robust
score lever by itself. It may be useful later as a very narrow emergency
condition, but not as a main branch.

The next candidate should move away from broad opening restrictions. Better
targets are midgame conversion mechanics: 4P leader-bash timing, hammer target
selection, post-capture reinforcement, or opponent-specific replay clusters.

## v37 Submission: Midgame Conversion

v37 returns to the clean v31 baseline and changes midgame conversion instead of
opening expansion.

Implementation:

```text
v37a: leader-aware hammer/mega-hammer retarget
v37b: v37a plus 4P post-capture consolidation
```

Local gates:

```text
v37a starter 2P: 100/100
v37a starter 4P: 40/40
v37b starter 2P: 100/100
v37b starter 4P: 40/40
v37a vs v31, 4P x3: 15/32 firsts
v37a with leader targeting off vs v31, 4P x3: 14/32 firsts
v37b vs v37a, 4P x3: 16/32 firsts
v37b with consolidation off vs v37a, 4P x3: 13/32 firsts
v37a vs v31, 2P: 10/20 wins, 13/20 firsts
v37b public-proxy 2P smoke: 32/32
```

Submissions:

```text
53784139: v37a leader-aware hammer retarget, 827.5
53784138: v37b hammer retarget plus consolidation, 857.4
```

The local signal did not transfer publicly. The conclusion is that v31 exp34
small modifications are not a good use of the remaining submission budget.

## v38 Final Attempts: Multi-Focus Baseline

The final two attempts switch to the public Multi-Focus high-score family rather
than continuing v31 micro-edits. The pulled public notebook describes a
PyTorch-based planner with predictive movement, exact competitive flow-diff
scoring, multi-source focus fire, and defensive regrouping.

Local readout before submission:

```text
v38a vs starter 2P smoke: 4/4
v38a vs starter 4P smoke: 4/4
v38a vs v31 2P: 18/20 wins
v38a vs v31 4P x3: 15/20 firsts
```

Rejected local variants:

```text
v38b 4P potential-risk regroup vs v38a: 3/20 wins, 4/20 firsts
v38c 4P strike4 vs v38a:               3/16 wins, 4/16 firsts
v38d 4P horizon14 vs v38a:             3/16 wins, 3/16 firsts
```

Because all tested modifications underperformed the public Multi-Focus baseline,
the first v38 submissions used the same high-score baseline twice, with the
second as a public-matchmaking resample:

```text
53907609: v38a multifocus public high baseline, 1107.3
53907608: v38e multifocus baseline resample, 887.5
```

v38a is the current best settled score. The v38e resample confirmed large public
matchmaking variance, so future candidates must be meaningfully different
strategy branches rather than tiny local probes.

## v39 Final-Window Large Branches

With two days left, the next attempts prioritize large behavior changes:

- v39f keeps v38a's 4P focus-fire fallback, but expands 2P single-source search
  from full-drain only to multi-size candidate tiers `(0.5, 0.75, 1.0)`.
- v39d packages the public Light Intruder dynamic planner as a different
  public-code-derived architecture: reinforcement-risk sizing, continuous
  behind/leader adjustment, and late-game candidate suppression.

Local gate:

```text
v39f py_compile: pass
v39f starter 2P/4P smoke: 8/8, 8/8
v39f vs v38a 2P: 11/16 wins, 12/16 firsts
v39f vs v38a 4P x3: 2/12 wins, 3/12 firsts
v38a vs v38a 4P x3 baseline: 2/12 wins, 3/12 firsts

v39d py_compile: pass
v39d starter 2P/4P smoke: 10/10, 8/8
v39d public-proxy 2P: 32/32
v39d vs v38a 2P: 7/12 wins, 8/12 firsts
v39d vs v38a 4P x3: 4/12 wins, 4/12 firsts
```

Submission:

```text
53913681: v39f 2p multisize with v38a 4p fallback, 932.1
53913691: v39d light intruder dynamic baseline, 1006.5
```

Public conclusion: v39f is rejected. The 2P multi-size search looked strong
against v38a locally, but public convergence dropped to the low 900s. v39d is
kept as the second-best settled fallback behind v38a.

## v40 Final-Day Challengers

The final-day policy is one protected fallback slot plus one or two high-risk
challenger attempts. We do not treat v40 as replacing v38a/v39d unless public
scores converge higher.

Rejected before submission:

```text
v40a star-wars LB1224 heuristic vs v38a 2P: 0/8
v40a star-wars LB1224 heuristic vs v38a 4P x3: 0/6
v40b heuristic LB1110 vs v38a 2P: 1/8
v40b heuristic LB1110 vs v38a 4P x3: 1/6
```

Submitted challengers:

```text
53943293: v40c im-stronger producer hybrid challenger, 1032.7
53943297: v40d decision-tree v38a-v39d policy selector, ERROR
53944332: v40e lazy decision-tree policy selector, ERROR
53952907: v40f single-file decision-tree selector, 915.7
```

v40c is a self-contained Producer Hybrid / "I'm Stronger" branch. Local gate:

```text
starter 2P/4P: 6/6, 6/6
v40c vs v38a 2P: 5/8 wins
v40c vs v38a 4P x3: 1/6 wins
```

Public conclusion: v40c is rejected. It converged near v38a but still below
the best-settled fallback.

v40d is the small decision-tree attempt requested for the final day. It packages
v38a and v39d as separate policies. Runtime policy:

```text
4P: always v38a
2P: use a depth-3 tree trained on seeds 0..23 from v39d-vs-v38a self-play
```

The tree only uses opening map features and compiles to hand-written thresholds:

```text
if neutral_prod <= 59: v38a
elif nearest_neutral_dist <= 11.2825: v39d
elif nearest_neutral_ships <= 23.5: v39d
else: v38a
```

Local gate:

```text
selector training positives: 14/24
selector training accuracy: 0.875
starter 2P/4P: 8/8, 8/8
v40d vs v38a 2P: 13/16 wins, 15/16 firsts
v40d vs v38a 4P x3: 2/8 wins, matching v38a mirror baseline shape
```

Kaggle validation rejected v40d with submission ERROR. The likely cause is that
the wrapper imported both large agent files at module import time. v40e keeps
the same policy tree but lazy-loads only the selected policy after reading the
initial observation:

```text
53944332: v40e lazy decision-tree policy selector, ERROR
v40e starter 2P/4P: 6/6, 6/6
v40e vs v38a 2P: 9/12 wins, 11/12 firsts
v40e vs v38a 4P x3: 2/6 wins
```

v40e also errored, which suggests the Kaggle runner did not accept the
multi-file/importlib selector pattern. v40f converts the same selector into a
single-file submission by embedding both v38a and v39d source strings and
executing them into registered module namespaces. Local behavior is unchanged:

```text
53952907: v40f single-file decision-tree selector, 915.7
v40f starter 2P/4P: 4/4, 4/4
v40f vs v38a 2P: 9/12 wins, 11/12 firsts
v40f vs v38a 4P x3: 2/6 wins
```

Final re-submissions for the closing window:

```text
53983805: final slot v38a multifocus best baseline, pending
53983816: final slot v40c im-stronger challenger, pending
```

## Baselines

- v19 restored the fast 2-player v16/v17 tempo while keeping the 4-player
  campaign layer.
- v20a added a 2-player do-nothing forecast and safe-drain cap. It does not
  choose targets; it only limits how much each source planet can safely spend.
- The strongest recent public signal is the resubmitted v20a, which has settled
  above v19 and the older v20a run. Public sampling is noisy, so local validation
  remains a gate before submission.

## v21 Result

v21 tested a planner-first 2-player pressure path. It was deliberately not a
heuristic multiplier layered on the old score: it had a separate planner that
could take over the turn before the legacy Phase 3 dispatcher.

The planner path was:

1. Build a 14-turn forecast from inferred fleet arrivals.
2. Reuse v20a safe-drain source budgets.
3. Shortlist high-budget sources and high-value targets.
4. Generate concrete single-source launch candidates.
5. Score each candidate by the simulated future value delta of the source and
   target trajectories.
6. Greedily select a small action set with one action per source and target.
7. Fall back to v20a if fewer than two coherent planner actions are found.

The accepted local gate enabled it only in 2-player pressure states:

- after turn 70 while behind on production or planets;
- elimination mode;
- dominant mode.

This gate preserved the v20a local benchmark, but public scoring regressed:

```text
v20a resubmission: 522-536 range during observation
v21 pressure gate: 487.4
```

The working conclusion is that a broad single-source planner is not the right
next step. It does not create enough value in rescue states to justify the risk
of overriding v20a. The next planner experiment should be narrower: pooled
focus-fire candidates for enemy planets only.

## Validation Log

Local tests use the `kaggle-dev` conda environment.

```text
v20a baseline:
2P starter, 100 seeds: 95/100
4P starter x3, 40 seeds: 34/40

v21 broad planner gate:
2P starter, 100 seeds: 91/100
4P starter x3, 40 seeds: 34/40

v21 battle-phase planner gate:
2P starter, 100 seeds: 93/100
4P starter x3, 40 seeds: 34/40

v21 pressure-only planner gate:
2P starter, 100 seeds: 95/100
4P starter x3, 40 seeds: 34/40
```

The pressure-only gate was the best local v21 configuration, but it is not the
accepted public baseline.

## Next Gates

Do not expand the general planner into 4-player games. The next meaningful
planner extension should be pooled focus-fire candidates, not another heuristic
multiplier or broad turn-level planner.

## v22 Design

v22 implements the narrower planner extension proposed after v21 regressed:
2-player pooled focus-fire against enemy planets only.

The pooled strike path:

1. Runs after v20a source budgets are computed.
2. Activates only in combat states: few neutrals, losing production, behind on
   planets, elimination/dominant mode, or after turn 95.
3. Considers enemy planets with meaningful production or elimination value.
4. Finds two or three nearby sources whose arrivals land within a short ETA
   window.
5. Fires only if the pooled total clears projected defenders with an overkill
   margin.
6. Marks used sources and the target so the legacy dispatcher can handle the
   remaining map without double-spending.

This is intentionally narrower than v21. It does not choose neutral captures and
does not replace Phase 3 broadly.

## v22 Validation

```text
v22 pooled focus-fire:
2P starter, 100 seeds: 95/100
4P starter x3, 40 seeds: 34/40
```

This matches the v20a local benchmark while giving public games a new mechanism
against enemy planets that require coordinated force.

## v23 Design

v23 keeps v22 and adds late 2-player frontier regrouping. The purpose is to help
against real opponents that pressure frontier planets while our rear planets keep
safe surplus.

The regroup path:

1. Runs only in 2-player games after turn 80.
2. Skips TURTLE, FINAL_PUSH, and logistics-guard states.
3. Computes distance-decayed enemy pressure on owned planets.
4. Moves at most two small surplus packets from lower-pressure planets to higher
   pressure owned planets.
5. Uses the current v20a/v22 source budgets, so it should not drain unsafe
   planets.

```text
v23 pooled focus + frontier regroup:
2P starter, 100 seeds: 95/100
4P starter x3, 40 seeds: 34/40
```

v22 and v23 are the two active submission candidates for the next convergence
window.

## v24 Plan And Implementation

v24 follows the revised historical replay-calibrated plan. It does not use
v22/v23 public games for diagnosis while they are still converging. The replay
research set is the settled historical group:

```text
v6   53262817
v17  53337723
v18  53339528
v20a 53511212
v21  53546090
```

The first replay sample was downloaded with:

```text
conda run -n kaggle-dev python scripts/fetch_replays.py --versions v6 v17 v18 v20a v21 --limit 3
conda run -n kaggle-dev python scripts/analyze_replays.py replays/historical --csv replays/historical/summary.csv
```

Small-sample taxonomy from those 15 public replays:

```text
v17:  enemy_snowball=2, win_or_unclear=1
v18:  early_expansion_loss=2, win_or_unclear=1
v20a: overextension=2, win_or_unclear=1
v21:  enemy_snowball=1, overextension=1, win_or_unclear=1
v6:   early_expansion_loss=1, overextension=1, win_or_unclear=1
```

This is not enough to tune exact thresholds by itself, but it does identify the
right failure families: early tempo loss, enemy snowball, and source-drain
overextension. The runtime response is intentionally narrow:

1. Base runtime is v22, not v23. The v23 2-player frontier regroup path is
   removed until replay evidence shows it helps a specific style.
2. Add a 2-player no-dependency opponent-style classifier using only current
   map/fleet features available by roughly turns 40-80.
3. Compile the classifier to hand-written thresholds in `main.py`; no runtime
   sklearn/pandas/torch dependency.
4. Keep `UNKNOWN` on v22 behavior.
5. Apply style knobs only where they match the taxonomy:
   fast expanders get more early neutral tempo, heavy attackers get safer drain,
   sniper retakers get high-production neutral buffers, and turtlers get earlier
   pooled focus-fire with conservative margins.

## v24 Local Gate

The test harness now supports suites:

```text
starter       existing starter/random style checks
public-proxy  scripted fast-expander, heavy-attacker, sniper-retaker, turtler
reference     reserved for local-only reference agents
```

Formal v24 gate after classifier-order fix:

```text
v24:
2P starter, 100 seeds: 96/100
4P starter x3, 40 seeds: 34/40
2P public-proxy, 20 seeds:
  fast-expander 18/20
  heavy-attacker 20/20
  sniper-retaker 20/20
  turtler 14/20

UNKNOWN ablation, 2P public-proxy, 20 seeds:
  fast-expander 19/20
  heavy-attacker 20/20
  sniper-retaker 19/20
  turtler 14/20
```

The turtler proxy remains a known local stress case. The UNKNOWN ablation shows
the same 14/20 turtler score, so those losses are not caused by the v24 style
branch. The positive local adaptive signal is sniper-retaker: 20/20 versus
19/20 with the classifier disabled.

Submission gate before replacing an active slot:

```text
conda run -n kaggle-dev python scripts/evaluate.py --suite starter --players 2 --seeds 100
conda run -n kaggle-dev python scripts/evaluate.py --suite starter --players 4 --seeds 40
conda run -n kaggle-dev python scripts/evaluate.py --suite public-proxy --players 2 --seeds 20
```

Required minimum remains:

```text
2P starter >= 95/100
4P starter >= 34/40
public-proxy: at least one clear style win signal, no broad regression versus
UNKNOWN ablation
```

## v24 Public Result And v25 Recovery

v24 regressed publicly:

```text
v24 historical replay adaptive style policy: 436.5
v23 pooled focus plus frontier regroup:     472.1
```

Downloaded public replay samples for v24 and v23 showed the key 2-player
difference:

```text
v23 2P sample: avg turn80 prod 36.3 vs enemy 27.6, avg t80 transit 5.72
v24 2P sample: avg turn80 prod 19.8 vs enemy 31.5, avg t80 transit 8.03
```

The diagnosis is that v24 did not fix early expansion losses and introduced more
midgame overextension. The classifier was too willing to alter attack tempo.

v25 response:

1. Restore the v23 2-player frontier regroup path.
2. Remove v24's offensive fast-expander and turtler knobs.
3. Keep the classifier only as a defensive risk sensor:
   heavy-attacker raises reserve/chip caution, sniper-retaker protects
   high-production neutral captures.
4. Do not ship the stricter early source-drain guard. It improved the turtler
   proxy but failed the 2P starter gate at 94/100.

v25 local gate:

```text
2P starter, 100 seeds: 96/100
4P starter x3, 40 seeds: 34/40
2P public-proxy, 20 seeds:
  fast-expander 19/20
  heavy-attacker 20/20
  sniper-retaker 20/20
  turtler 14/20

UNKNOWN ablation, 2P public-proxy, 20 seeds:
  fast-expander 19/20
  heavy-attacker 20/20
  sniper-retaker 19/20
  turtler 14/20
```

The only local adaptive gain is still narrow but real: sniper-retaker improves
from 19/20 to 20/20 without broad proxy regression.

## v25 Public Result And v26 Ablation

```text
v25 guarded regroup rollback: 542.2
v26 no-style regroup rollback: 495.4
```

v25's initial low score was an under-sampled public signal, not a settled
regression. After convergence, v25 became the current strongest submitted
variant and beat both v20a and v23 on public score.

v26 was submitted as a no-style ablation after the early v25 score looked bad.
It disables the classifier at runtime:

```text
opponent_style = 'UNKNOWN'
```

That returns the action path to the v23-like structure:

1. v20a safe-drain forecast.
2. v22 2-player pooled focus-fire.
3. v23 2-player frontier regroup.
4. No fast-expander/heavy-attacker/sniper-retaker/turtler knobs in runtime.

v26 local gate:

```text
2P starter, 100 seeds: 95/100
4P starter x3, 40 seeds: 34/40
2P public-proxy, 20 seeds:
  fast-expander 19/20
  heavy-attacker 20/20
  sniper-retaker 19/20
  turtler 14/20
```

This was not a breakthrough version; it was a stabilization rollback and now
serves as a useful ablation against v25.

The ablation result is useful: no-style v26 reached 495.4, while guarded-style
v25 reached 542.2. The defensive-only classifier appears to add public value,
but the broader offensive v24 classifier still remains rejected.

## Current Best Submission Records

Keep these as the known best pair until a newer candidate converges higher:

```text
53907609  v38a multifocus public high baseline  1107.3
53913691  v39d light intruder dynamic baseline   1018.4
```

Self-developed line reference:

```text
53611826  v25 guarded regroup rollback           542.2
53511212  v20a mini forecast safe-drain cap      522.3
```

v26 is a useful ablation but not a best-slot candidate:

```text
53612737  v26 no-style regroup rollback      495.4
```

## v27 Capture Blackout

The next safe experiment is based on a high-score reference-agent idea:
newly captured productive planets should not always be treated as ordinary
sources immediately. Draining them too early can convert a capture into a weak
frontier liability.

v27 keeps v25's runtime shape and adds a conservative 2-player launch blackout:

1. Applies only to non-initial planets we recently captured.
2. Applies only before turn 150 and only to production >= 3 planets.
3. Lasts for 9 turns after ownership flips to us.
4. Does not block defensive Phase 1 reinforcements.
5. Caps available launch ships so the planet keeps roughly `production * 6 + 14`
   ships unless it has already grown beyond that.

v27 local gate:

```text
2P starter, 100 seeds: 95/100
4P starter x3, 40 seeds: 34/40
2P public-proxy, 20 seeds:
  fast-expander 20/20
  heavy-attacker 20/20
  sniper-retaker 20/20
  turtler 16/20
```

Public result:

```text
53629409  v27 capture blackout guarded regroup  472.3
```

Despite improving the proxy suite, v27 regressed below v25 and v20a. The
capture-blackout hypothesis is rejected for now. It likely delayed useful
second-wave pressure more than it protected fragile captures.

## Path Toward 1000

The gap to 1000 is unlikely to be closed by heuristic multipliers. The strong
public references point to a different architecture:

1. Exact movement and garrison projection over a short horizon.
2. Candidate search over source-target pairs, not one greedy score pass.
3. Forward simulation/ROI scoring before launch.
4. Multi-source hammer timing and second-prong attacks against reinforcers.
5. Local opponent suite built from downloaded public replays, not only starter
   and hand-written proxies.

The practical next milestone is a local `reference-lite` planner that can run
inside `main.py` without torch: 2-player only, short horizon, top sources,
top targets, one or two waves. It should be introduced behind a gate and
compared directly against v25/v27 before submission.

## v28 Second-Prong Anti-Reinforcer

v28 is the first small `reference-lite` tactic. It does not replace the greedy
dispatcher. It only fires after an enemy-planet strike is already committed and
only in 2-player combat states. If an enemy source is sending fleets to reinforce
that primary target, v28 may attack the depleted source using safe surplus.

Runtime gates:

1. Turn >= 95.
2. 2-player only, not logistics guard, not final push.
3. Must already be attacking an enemy planet.
4. Must be losing production, behind on planets, or in low-neutral combat state.
5. One second prong per turn.

v28 local gate:

```text
2P starter, 100 seeds: 96/100
4P starter x3, 40 seeds: 34/40
2P public-proxy, 20 seeds:
  fast-expander 19/20
  heavy-attacker 20/20
  sniper-retaker 20/20
  turtler 14/20
```

This is a conservative public candidate to replace the failed v27 slot. It is
not expected to be enough for 1000 by itself; it is a first narrow step toward
multi-prong planning.

Public result:

```text
53635769  v28 second-prong anti-reinforcer  495.8
```

v28 is rejected as a candidate. It passed starter gates but did not improve
public score over v25 or v20a.

## v29 Early Transit Cooling

Reference diagnostics showed v25 loses to public high-score references by
putting too much mass in flight early and being cleared around turns 80-150.
An early transit-cooling experiment prevented early "barely-capture" neutral
all-ins while fleet mass was high.

Local result:

```text
vs loraus/ldausl-style reference, 5 seeds: 0/5, only slightly delayed losses
2P starter, 40 seeds: 37/40
```

v29 is rejected before submission.

## v30 Public Reference Baseline

A public reference notebook extract was tested as a calibration baseline. It
crushes v25 locally and passes our current gates:

```text
2P starter, 30 seeds: 30/30
4P starter x3, 20 seeds: 20/20
2P public-proxy, 10 seeds:
  fast-expander 10/10
  heavy-attacker 10/10
  sniper-retaker 10/10
  turtler 10/10
```

This is not a self-developed continuation of v25. It is submitted as a public
reference calibration candidate so we can verify the score range and use it as a
target for a future self-developed reference-lite planner.

Public result:

```text
53647029  v30 public reference calibration baseline  906.1
```

The user observed earlier 956.1 and 933.5 readings, but the latest user-visible
score after v31 landed is 906.1. v30 is no longer the overall best, but remains
the second public-reference calibration baseline.

## v31 Public Reference Variant

v31 switches `main.py` to the `orbit-wars-exp34` extracted reference variant.
Compared with v30/ldausl, the visible top-level differences include:

```text
SEARCH_MAX_ACTIONS_TO_PICK_2P: 8 instead of 7
SEARCH_DEPTH2_ENABLED: False instead of True
VALUE_WEIGHT_2P: 5.2 instead of 4.86118
```

Local smoke:

```text
2P starter, 20 seeds: 20/20
4P starter x3, 12 seeds: 12/12
2P public-proxy, 6 seeds each: 24/24
```

## v34 Hybrid Risk-Punish Strategy Layer

v34 returns `main.py` to the v31/exp34 baseline and adds public-code-derived
strategy logic beyond parameter tuning. The active default keeps v31's search
engine but adds:

```text
HybridRiskState: 2P style/in-flight/source-risk sensor
drained_source_punisher: attacks recently depleted enemy sources
style score nudges: fast-expander/turtler/sniper/heavy-attacker risk signals
```

Ablation toggles are available:

```text
V34_SAFE_TRANSIT=1
V34_HOLD_FILTER=1
V34_DRAINED_PUNISH=0
```

Ablation result against v31 showed `safe_transit_governor` was too disruptive
and should remain off by default. `post_capture_hold_filter` was neutral alone
but interacted poorly with punish, so it also remains off by default. The
submitted/default candidate keeps the drained-source punish layer plus style
scoring.

Local gates:

```text
py_compile: pass
2P starter, 100 seeds: 100/100
4P starter x3, 40 seeds: 40/40
2P public-proxy, 30 seeds each: 120/120
v34 vs v31, 100 seeds: 47/100
v34 vs v30, 50 seeds: 18/50
v31 vs v30, 50 seeds: 19/50
v34 vs v25, 50 seeds: 50/50
```

Decision before submission: v34 was a valid public candidate because it was not
a simple parameter tune, passed full regression, stayed within the v31
head-to-head gray band, and did not regress versus v30 beyond noise.

Submission:

```text
53661855  v34 hybrid risk-punish strategy layer  893.5
```

Public conclusion: rejected. The drained-source punish layer and style scoring
looked locally neutral, but public convergence landed well below v31. This
suggests that extra tactical interference in the exp34 search loop hurts pool
robustness even when local head-to-head appears acceptable. Keep v34 archived as
a negative experiment and use v31 as the fallback working baseline.

This is another public-reference calibration candidate, not a self-developed
agent. The self-developed best remains v25 at 542.2.

Public result:

```text
53648046  v31 exp34 public reference variant  948.6
```

The user observed earlier 978.5 and 968.5 readings, but the latest API check on
2026-06-14 shows 948.6. v31 is still the current overall best submitted agent
and the primary final-window candidate unless a later parameter variant
converges higher.

## v32 Exp34 Value-Weight Probe

v32 keeps the v31/exp34 search shape and only raises the 2-player value weight:

```text
SEARCH_MAX_ACTIONS_TO_PICK_2P: 8
SEARCH_DEPTH2_ENABLED: False
VALUE_WEIGHT_2P: 5.4 instead of 5.2
```

This is intentionally a narrow probe near the 1000 boundary. The hypothesis is
that v31's improvement over v30 came partly from valuing the forward score more
aggressively in 2-player games, while disabling depth-2 avoids overfitting the
near-term branch search.

Local smoke:

```text
2P starter, 20 seeds: 20/20
4P starter x3, 12 seeds: 12/12
2P public-proxy, 6 seeds each: 24/24
```

Public result:

```text
53650973  v32 exp34 value-weight 5.4  914.3
```

v32 is rejected as a best candidate. The 5.4 weight appears to overvalue
forward-score pressure and loses the balance that made v31 strong.

## v33 Exp34 Value-Weight Downshift

v33 keeps the v31/exp34 search shape and probes the other side of the v31 peak:

```text
SEARCH_MAX_ACTIONS_TO_PICK_2P: 8
SEARCH_DEPTH2_ENABLED: False
VALUE_WEIGHT_2P: 5.1 instead of 5.2
```

The hypothesis is that v31's useful region is near 5.2, but v32 showed the high
side is brittle. A slight downshift may preserve the exp34 tempo while reducing
over-committed forward-score attacks against stronger public opponents.

Local smoke:

```text
2P starter, 20 seeds: 20/20
4P starter x3, 12 seeds: 12/12
2P public-proxy, 6 seeds each: 24/24
```

Public result:

```text
53660453  v33 exp34 value-weight 5.1  863.1
```

v33 is below v31 and is rejected as a final-window candidate.

## v35 Production-Ordered Candidate Evaluation

v35 keeps the v31/exp34 constants and restores one v30-like structural idea:
sort generated 2-player step actions by target production before the expensive
Melis evaluator truncates to the first candidate window. This is not a value
weight tweak and does not add tactical vetoes; it only changes which generated
actions get evaluated first.

Local gates:

```text
py_compile: pass
2P starter, 100 seeds: 100/100
4P starter x3, 40 seeds: 40/40
2P public-proxy, 30 seeds each: 120/120
v35 vs v31, 50 seeds: 24/50 wins, 29/50 firsts
v35 vs v30, 50 seeds: 19/50
v35 vs v25, 30 seeds: 30/30
```

Decision: submit as a low-intrusion candidate. It is weaker than a clear local
breakthrough, but unlike v34 it does not add a separate tactical layer and its
head-to-head result is close enough to v31 to justify one public slot.

Submission:

```text
53683060  v35 production-ordered evaluator window  PENDING
```
