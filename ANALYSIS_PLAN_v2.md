# Analysis plan for run v2 (written before any v2 trace was scored)

Run v1 (323 directed seeds, `runs/c050`, `results/final`) was exploratory: the stop rules, the excursion classes and
the error detector were all developed on it. Run v2 is a larger, independent set of traces used to **confirm** those
findings. Everything below is fixed before v2 results exist; the SHA-256 of this file and its timestamp are recorded
in `results/v2/plan_hash.txt` before the v2 outputs are copied off the pod. Anything not listed here is reported as
exploratory.

## Data (identical settings to v1 unless stated)

- Checkpoint `scrollprize/fiber_single_small`, HF revision `451182558cb3270cfc2ddf967acbd4f36b9b816b` (step 50000);
  villa commit `ccb431818fa68059138f2c964ae299fc474ff116`; confidence 0.5 (CLI default); max_len 2000 trace voxels per
  direction; CT level and normalization from the checkpoint.
- Seeds: villa's `make_seeds` on **every** annotated fiber (v1 used villa's `validation_manifest`, which caps a split at
  40 fibers), at `per_fiber` annotated positions with RNG seed 7349+fiber index, both directions:
  Paris4 hash-held-out 62 fibers x 4; PHerc1451 72 x 4; PHerc0332 77 x 4; PHerc0139 409 x 2.
- v2 shares fibers with v1 (all of 1451/0332 and 119 of 0139's fibers) but uses new seed positions; 0139 fibers never
  seeded in v1 are reported separately as the cleanest subset.
- Trace batch 128 (verified bit-identical to batch 16 on 16 Paris4 seeds). Fwd/back: `code/fwdback.py` with the v1
  defaults (tolerance 3, check every 128, back length 192).

## Scoring (unchanged)

villa strict scorer (tolerance 3, patience 3), pointwise length, and the excursion classes of `code/stats.py`
(returned: back within 3 vx for 16 vx; hard switch: not returned and peak > 15 vx; drift; ended).
Uncertainty: bootstrap by **fiber** (not by trace, since several seeds share a fiber), B = 4000, seed 7349.

## Primary questions

1. **Generalization.** Strict departures per cm, unseen scrolls pooled vs Paris4. Confirmed if the rate ratio > 1 with
   a 95% fiber-bootstrap CI excluding 1.
2. **Fwd/back catches fiber switches; confidence does not.** Detector frozen on v1 (`results/final/detector_frozen.json`):
   fwd/back flag = any back-trace point > 3 vx from the forward stretch (fb > 0); confidence flag = min conf16 over
   the stretch < 0.566 (conf > 0.434); a departure is caught if a check at forward length [onset, onset+192] flags.
   On hard switches pooled over the unseen scrolls: confirmed if fwd/back recall > confidence recall, exact McNemar
   test on paired outcomes, p < 0.05. Reported per scroll with fiber-bootstrap CIs.
3. **Fwd/back false alarms.** Flags on checks whose 192-vx stretch is entirely on the fiber, per cm of clean path, per
   scroll, with CIs; v1 measured 0–0.09 / cm.
4. **Frozen combined detector.** AUROC of the v1-frozen logistic model per scroll; confirmed where the 95% CI excludes 0.5.
5. **Stop rule.** The pre-specified segmented fwd/back rule vs confidence stopping at equal correct length (v1
   method, `stats.py` stop_rules), per scroll, pointwise and strict.

## Exploratory (labelled as such)

- Why fwd/back misses some switches: CT image features around caught vs missed hard switches (layer spacing,
  contrast), plus geometry (peak distance, return).
- Results restricted to 0139 fibers not used in v1; per-annotator breakdown.

## Not changed after seeing v2

Thresholds, the frozen model, class definitions, windows and tolerances above. If a bug is found, the fix and its
effect on both v1 and v2 numbers will be reported explicitly.
