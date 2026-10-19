# fiber_single_small on scrolls it was not trained on

`scrollprize/fiber_single_small` traced every annotated fiber on PHerc0139, PHerc1451 and PHerc0332 (not in its
training metadata), against held-out Paris4 fibers. 3,248 traces, villa's own scorer, conf 0.5.

| | Paris4 | 0139 | 1451 | 0332 | unseen vs Paris4 |
|---|---|---|---|---|---|
| within 1.5 vx (release notes' "90%") | 0.91 | 0.77 | 0.82 | 0.79 | 0.86× [0.83, 0.90] |
| departures per cm | 0.16 | 0.55 | 0.29 | 0.47 | **3.05×** [2.2, 4.5] |
| strict precision (villa + PR #2143) | 0.82 | 0.64 | 0.78 | 0.77 | |
| real jumps to another fiber, per cm | 0.030 | 0.073 | 0.079 | 0.022 | 2.2× [1.3, 4.4] |
| jumps caught: fwd/back check | 70% | 67% | 76% | 3/8 | **68%** |
| jumps caught: confidence | 15% | 38% | 46% | 3/8 | **40%** (p = 1e-7) |

- On unseen scrolls it leaves the fiber about **3× as often**.
- A simple **fwd/back check catches most jumps** (68% vs 40%), with fewer false alarms.
- Jumps depend on **which fiber**, not the spot: they slide sideways onto the nearest same-direction fiber (~3.6 vx).
- villa's scorer counted length past the annotation end as wrong: fixed in
  [PR #2143](https://github.com/ScrollPrize/villa/pull/2143).

Correction (19 Oct): an earlier version counted 472 jumps; 271 were traces running past the label end.

Details, methods and limits: [REPORT.md](REPORT.md). Plans fixed before each run: `ANALYSIS_PLAN_v2.md`, `ANALYSIS_PLAN_v3.md`.
Brackets: 95% CIs, bootstrap over fibers. GPU cost ~$13. Built with Claude Code as a coding assistant; questions,
decisions and mistakes are mine. MIT. Model, tracer and scorer: Vesuvius Challenge team; fibers: annotation team.
