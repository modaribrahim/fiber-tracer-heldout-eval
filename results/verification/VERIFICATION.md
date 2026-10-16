# Verification of the evaluation (2026-10-16)

Every check below was run against villa at the pinned commit `ccb431818fa68059138f2c964ae299fc474ff116`, fetched fresh
from GitHub, and against `scrollprize/fiber_single_small` revision `451182558cb3270cfc2ddf967acbd4f36b9b816b`. The
checkpoint's sha256 is `3268a0ac1807debdeaded05fa7bc90b82d3ca8102125247e5cba688a10a4811f`, equal to the HF etag.

## 1. The tracer is run exactly as villa runs it

| Item | villa | ours | Same |
|---|---|---|---|
| Evaluation route | `evaluation/protocol.py`: `FiberTracer(..., TraceParams.from_policy(policy, max_len))` then `scoring.evaluate` | identical calls (`code/heldout_eval.py`) | yes |
| Confidence | `tracing/inference.py`: `DEFAULT_CONFIDENCE = 0.5` ("a3ms 50k: usable 0.3-0.5"). The checkpoint's recorded policy says 0.7, and the CLI overrides it to 0.5. | 0.5 (v1 also ran 0.7) | yes |
| max_len | protocol default 2000 | 2000 | yes |
| n_commit / gate / recovery / refinement | checkpoint policy: 8 / full / 6.0 / 3 | same (`config.json`) | yes |
| Precision, refit retry | `TraceParams` defaults: bf16, off | defaults: bf16, off | yes |
| CT level, scale | checkpoint: level 2, ct_grid_scale 4, grid_scale 8 (1 trace vx = 19.2 um) | same | yes |
| Normalization | `prepare_normalization(out, [spec], known=ck['ct_normalization'])` (crop z-score) | identical call | yes |
| Frame model | `resolve_checkpoint` downloads `heading_model.pt`. If it is missing, villa exits. | log: "Fetching 2 files" | yes |
| Strict scorer | `scoring.score_trace`: tol 3, patience 3, arclength-matched GT distance | villa's own `evaluate` and `trace_events` | yes |
| Confidence signal | gate = `cummin(confidence)[:16][-1] >= threshold` (`shared/policy.py:128`) | `conf16` = cumulative min at plane 16 | yes |

villa has no commits touching the tracer, scoring or policy code after the pinned commit, and the HF model head is
still the evaluated revision.

## 2. The data are what we say they are

- **Unseen scrolls.** The checkpoint's `dataset_config` and `dataset_provenance` list the training sources: Paris4 human
  fibers, plus automated fiber volumes on PHerc0175A, PHerc1447, Paris4-central and PHerc0125. PHerc0139, 1451 and
  0332 are absent, and none of their fiber names is in `fiber_manifest`.
- **CT volumes.** The fiber JSONs of 0139 and 1451 record `vc_open_data_coordinate_space` 20260102150214 and
  20260319101107, the volumes we traced. The 0332 JSONs (version 3) carry no such field. The bucket has exactly one
  0332 volume (20251211183505), and the fiber coordinates fit inside it: z max 33591 of 33592 slices.
- **Paris4 control.** Training used a `fiber_hash` split (fraction 0.1, seed 7349) over the fibers outside a legacy
  reserved list that is not public. The checkpoint's own counts fix the maximum size of that list: 765 manifest fibers,
  563 trained. `code/verify_paris4_split.py` shows that **55 of the 62** control fibers are held out under every
  reserved list consistent with those counts. The other 7 are held out under the natural reading, where the reserved
  list is the 120 non-`anon_*` names, but this cannot be proven. Sensitivity using only the 55:

  | | All 62 | 55 guaranteed |
  |---|---|---|
  | Strict precision | 0.796 | 0.797 |
  | Departures per cm | 0.155 | 0.153 |
  | Unseen/Paris4 ratio | 3.05 | 3.11 |
  | Hard-switch recall, fwd/back vs confidence | 0.47 vs 0.20, p = 0.007 | 0.48 vs 0.20, p = 0.017 |

  **Caveat:** the model also trained on *automated* fibers of Paris4-central, on the same CT. The Paris4 control is
  therefore "same scroll, held-out human annotations", not an unseen region.
- **Seeds.** villa's `make_seeds` produced none for 7 of 409 fibers on 0139 and 6 of 77 on 0332, so 402 and 71
  fibers were traced.

## 3. The numbers reproduce

- `code/confirm_v2.py` re-run from the raw run files into a scratch directory gives **byte-identical** outputs: all 10
  CSVs plus `confirm_stdout.txt`.
- Inside that run, every one of the 3,248 traces reproduces villa's stored departure index and its correct/wrong
  length split. The exceptions are 9 rounding ties at exactly 3.000 vx, where villa's index is used.
- `code/verify_independent.py` is a second implementation that shares no code with `stats.py` or `detect.py`: its own
  excursion classifier and its own detector application. It matches every headline number (`independent_stdout.txt`):

  | | Precision | Departures/cm | Hard switches (checked) | Caught: fwd/back vs confidence | McNemar (fb only / conf only) |
  |---|---|---|---|---|---|
  | Paris4 | 0.796 | 0.155 | 51 | 0.471 vs 0.196 | 19 / 5, p = 0.0066 |
  | 0139 | 0.579 | 0.546 | 271 | 0.450 vs 0.332 | 64 / 32, p = 0.0014 |
  | 1451 | 0.776 | 0.293 | 48 | 0.750 vs 0.458 | 20 / 6, p = 0.0094 |
  | 0332 | 0.586 | 0.465 | 75 | 0.120 vs 0.173 | 5 / 9, p = 0.42 |
  | Unseen pooled | | ratio 3.053 | 394 | 0.424 vs 0.317 | 89 / 47, p = 0.0004 |

## 4. Pre-registration

The sha256 of `ANALYSIS_PLAN_v2.md` and of `results/final/detector_frozen.json` equal the values in
`results/v2/plan_hash.txt`, which was recorded at 23:15:56Z. The pod's `v2_status.log` shows the v2 run starting at
23:20:05Z and the first trace at 23:23:37Z. The timestamps are our own; no third party has witnessed them.

## 5. Points the write-up must state

1. Detection recall is over departures that had a fwd/back check within 192 vx after onset: 1,698 of 1,930. Both
   detectors use the same set, so the comparison stays paired.
2. On the v1 Paris4 clean checks used to freeze the thresholds, fwd/back (fb > 0) had a 0.0% false-alarm rate and
   confidence 5.1% (392 checks). Fwd/back is therefore compared at a lower false-alarm rate, which does not favour it.
3. Fwd/back is *our* implementation (`code/fwdback.py`). villa has no forward/backward code, and we are not
   evaluating Sean's own fwd/back procedure.
4. The fwd/back stop rule is an offline *trim*: it cuts back to the last passing check. The confidence curve is
   exact for any threshold of 0.5 or above, because a higher threshold stops on the same trajectory.
5. Sean's "90% within 1.5 vx" and "<1%" metrics are not computed here.
