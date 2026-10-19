# Analysis plan v3: where and why the tracer leaves the fiber (written before any v3 number is computed)

v2 measured how often `fiber_single_small` leaves the human fiber on unseen scrolls. v3 asks **where** it happens and
**what the place looks like**. It reuses the saved v2 traces unchanged (`runs/v2`, release `data-v1`, sha256
`3215b1ae…6540`); no new tracing, CPU only. The SHA-256 of this file is recorded before any v3 output exists.

Lesson from v2: 58% of the wrong length on 0332 was a scoring artifact (length past the annotation end). So every
"cause" below is also checked for a non-model explanation (annotation geometry, annotation ends, annotation density).

## Data checks (run first, reported)

- Human fibers re-downloaded on 2026-10-19 with `code/fetch_fibers.py`. Each evaluated fiber's source hash must match
  the v2 manifest; fibers that changed are reported and excluded.
- The stored per-vertex GT distance must be reproduced from the re-downloaded geometry (99th percentile of the
  absolute difference reported, for vertices within 10 vx).

## Events and exposure (definitions unchanged from v2)

- Episodes from `stats.episodes` on the villa-scored region (tol 3 vx, patience 3 vx): `hard_switch` (primary event),
  `drift`, `returned`, `ended`. "Persistent" = hard switch or drift (sensitivity).
- Departure location: the GT point nearest to the last path vertex before the episode onset.
- At-risk exposure: path segments whose end vertex is within 3 vx of the GT, before the trace's first persistent or
  ended episode. Each segment is assigned to the GT point nearest its end vertex. 1 vx = 19.2 um; rates per cm.
- Uncertainty: bootstrap by fiber within scroll, B = 4000, seed 7349. Two-sided bootstrap p = 2 x min(share <= 1,
  share >= 1), floor 1/B.

## Covariates at a GT point P(s) (villa `load_fibers`, grid_scale 8, spacing 1)

1. bend: angle (deg) between chords P(s)-P(s-16) and P(s+16)-P(s); undefined within 16 vx of an end.
2. kink (annotation roughness): max over |u| <= 8 of the angle between 4-vx chords at s+u.
3. crossing distance: 3-D distance to the nearest point of another annotated fiber whose local direction (chord
   P(s+4)-P(s-4)) is >= 45 deg from this one. Other fibers = every file in the scroll's fiber directory. Capped at 100 vx.
4. parallel distance: the same for < 45 deg.
5. junction distance: 3-D distance to the nearest annotated branch point (JSON `branches`) of any fiber.
6. family: H or V (villa's automatic tag).
7. seed distance |s - t0|; 8. end distance (to the nearer end of the controlled span); 9. distance to the nearest
   human control point.

Tertiles for tests: cut points at the 1/3 and 2/3 quantiles of at-risk exposure **within each scroll** (annotation
density differs between scrolls; the Paris4 directory has 645 fibers). Pooled cut points are used only for S1.

## Primary questions (Holm over the 5 tests, alpha 0.05)

- **Q1 Place.** Are hard switches tied to places? Each GT is cut into fixed 32-vx bins. For every pair of
  opposite-direction traces on the same fiber, each bin both entered while on the fiber is one 2x2 cell (A switches
  in the bin / B switches in the bin); a trace that ends in a bin without a hard switch is censored there.
  Mantel-Haenszel odds ratio stratified by fiber (so harder fibers do not count as places). Confirmed if the
  bootstrap 95% CI excludes 1. Sensitivity: 16 and 64 vx bins; persistent instead of hard. Same-direction pairs are
  reported, not tested (their paths are nearly identical).
- **Q2 Bend.** Hard switches per cm, top vs bottom bend tertile, Mantel-Haenszel rate ratio stratified by scroll.
- **Q3 Crossing fibers.** Nearest vs farthest tertile of crossing distance (same test).
- **Q4 Parallel neighbours.** Nearest vs farthest tertile of parallel distance (same test).
- **Q5 Family.** V vs H (same test).

Per-scroll estimates are reported for every question.

## Secondary (estimates with CIs, no tests)

- **S1 Does geometry explain the unseen-scroll gap?** Strict first departures per cm (the v2 3.05x quantity), unseen
  pooled vs Paris4, crude and adjusted (Poisson, log exposure) for pooled tertiles of bend and kink, family,
  seed-distance band (< 100, 100-400, > 400 vx) and end-distance band (< 32, >= 32 vx). Proximity covariates are left
  out because annotation density differs between scrolls.
- **S2 Annotation check.** Hard switches and returned excursions per cm by kink tertile.
- **S3 Junctions.** Hard switches per cm within 16 vx of a junction vs farther (few junctions; low power).
- **S4 When.** Hard switches per cm by seed-distance band.
- **S5 Profiles.** Mean of each covariate at offsets -64..+64 vx (direction of travel) around hard switches vs the
  exposure-weighted mean.
- **S6 What a switch looks like.** Share of hard switches whose path lands on another annotated fiber (within 3 vx
  for >= 16 vx of path in the first 200 vx after onset) and that fiber's family; turn angle between the GT direction
  at the departure and the path chord from onset+16 to onset+48 vx; for hard switches that are the trace's first
  departure, the normal / in-sheet split already in `departures.csv`.
- **S7 Hotspot list.** Bins where both directions switched, with coordinates and covariates, for visual review.
- **S8 Mutual adjustment.** Poisson model for hard switches with all tertiles, family and scroll together.

## Reading the results (fixed now)

- Q1 positive: switches come from specific places (hard image spots or annotation problems), not chance. Q1 negative:
  they look random along the fiber, so place features cannot explain them.
- Q2-Q4 positive: that geometry raises the risk. A kink effect (S2) means some "switches" may be annotation errors.
- S1: if the adjusted ratio stays near the crude one, the unseen-scroll gap is not explained by the geometry of the
  annotated regions.

## Not changed after seeing results

Definitions, windows, bins, tertile rules and tests above. If a bug is found, the fix and its effect are reported.
