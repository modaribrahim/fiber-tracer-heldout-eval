# fiber_single_small on scrolls it was not trained on

How well does `scrollprize/fiber_single_small` trace annotated fibers on PHerc0139, PHerc1451 and PHerc0332 (not
listed in its training metadata), compared with held-out Paris4 fibers? 3,248 traces, villa's own scorer, conf 0.5.

| | Paris4 (held out) | PHerc0139 | PHerc1451 | PHerc0332 | 3 unseen vs Paris4 |
|---|---|---|---|---|---|
| traces / fibers | 496 / 62 | 1608 / 402 | 576 / 72 | 568 / 71 | |
| strict length precision (villa + PR #2143) | 0.82 | 0.64 | 0.78 | 0.77 | |
| departures per cm | 0.16 | 0.55 | 0.29 | 0.47 | **3.05×** [2.2, 4.5] |
| real hard switches per cm | 0.030 | 0.073 | 0.079 | 0.022 | 2.2× [1.3, 4.4] |
| hard switches caught by fwd/back | 70% | 67% | 76% | 3 of 8 | **68%** pooled |
| hard switches caught by confidence | 15% | 38% | 46% | 3 of 8 | **40%** pooled (p = 1e-7) |

- **It leaves the fiber about 3× as often per cm on unseen scrolls.** Bend, kink, family and seed/end distance
  explain little of it (adjusted 2.6×).
- **A simple fwd/back check flags most real switches**, with about 4× fewer false alarms per cm than confidence.
- **Switches depend on the fiber, not the spot.** Within the same fiber, bends, crossings and nearby parallel fibers
  don't predict where it leaves. Real switches mostly slide sideways onto a same-direction fiber, often the nearest one
  (median 3.6 vx away).
- **Scoring past the annotation end.** After a departure, villa counted length past the annotation end as wrong;
  [villa PR #2143](https://github.com/ScrollPrize/villa/pull/2143) makes it unknown (table row 2).

Hard switch: a departure (> 3 vx for ≥ 3 vx) that never returns and peaks > 15 vx away. "Real": not a trace running
past the annotation end (see the correction below). Brackets: 95% CIs from a bootstrap over fibers.

**Correction (19 Oct 2026).** v2 reported fwd/back 42% vs confidence 32% and said the check failed on PHerc0332.
271 of the 472 "hard switches" were traces running past the untagged annotation end after an earlier small
excursion (77 of 85 on 0332). The departure headline is unchanged (3.04×).

## More

- [REPORT.md](REPORT.md): methods, all tables, limits, how to reproduce.
- `ANALYSIS_PLAN_v2.md`, `ANALYSIS_PLAN_v3.md`: plans fixed before each run (hashes in `results/v2`, `results/v3`).
- `results/v2/`, `results/v3/`: tables and figures. Raw traces: release `data-v1`.

**Author's note.** When the fiber tracer was released, I wanted to see how it performs on scrolls outside its training
data, since that is where it will likely be used most. The release notes also mention that identifying tracing errors
is not straightforward. I tested it on all available annotated fibers from PHerc0139, PHerc1451, and PHerc0332, and
evaluated a simple version of the forward/backward consistency check described in the notes. I fixed each analysis
plan before running it. Total GPU cost was about $13. I used Claude Code as a coding assistant, but the research
questions and decisions were mine, as is responsibility for any mistakes. Feedback and corrections are welcome.

Credits: model, tracer and scorer by the Vesuvius Challenge team (`scrollprize/fiber_single_small`, ScrollPrize/villa);
fiber annotations by the annotation team. License: MIT.
