"""v3 stage 2b (ANALYSIS_PLAN_v3.md): the primary tests Q1-Q5 and the secondary S2-S4, S7, computed exactly as pre-registered.

Usage: python -I why_tests.py DATA FIBERS VILLA OUT [--2c]
  DATA, FIBERS, VILLA  as in why_errors.py (only the S7 hotspot list reloads the human fibers, for GT xyz)
  OUT                  reads OUT/tables/ (stage 2a); writes OUT/results/{q1_place,q2_q5_rates,primary_tests,sensitivity,
                       secondary,hotspots}.csv and results_2b.json
  --2c                 stage 2c instead of 2b (post-hoc; needs `why_errors.py ... --flags` first): reads OUT/tables/ and
                       OUT/results/{episode_flags.csv, strict_exposure.npz}; writes OUT/results/{flag_counts,end_censored,
                       strict_headline,s1_poisson,q1_expectation}.csv and results_2c.json
  --2d                 stage 2d instead of 2b (post-hoc; needs `why_errors.py ... --flags` and `--corrected` first): original vs episodes recomputed on
                       the region cut at the annotation end (OUT/tables_2d); writes OUT/results/{transitions_2d,corrected,strict_headline_2d,
                       s1_poisson_2d,hard_per_cm,q1_expectation_2d,detection_recheck,detection_changed}.csv and results_2d.json
  --2e                 stage 2e instead of 2b (post-hoc except S5, S6, S8, which were pre-registered and now run on the corrected episodes; needs --2d and
                       `why_errors.py ... --s6`): writes OUT/results/{fiber_stratified,s8_poisson,s6_summary,s5_profiles,rates_by_condition}.csv, results_2e.json
Nothing here interprets results.

Conventions (fixed by the stage-2b instructions): tertile cut points c1, c2 per scroll and covariate are the exposure-weighted
33.3 / 66.7 quantiles in sanity.json (bin 0: x <= c1, bin 2: x >= c2, else bin 1; NaN excluded from exposure and events).
Bootstrap: B = 4000, default_rng(7349); each replicate draws, for each scroll in order, multinomial fiber counts over that scroll's
fibers (every fiber has traces); one weight matrix is used for every analysis. CI: 2.5 / 97.5 percentiles of the defined
replicates (x/0 = +inf, 0/x = 0, 0/0 undefined and counted). p = 2 min(share R* <= 1, share R* >= 1) over defined replicates,
floor 1/B, cap 1. Mantel-Haenszel RR over scrolls: sum_s a E0/T / sum_s b E1/T; Q1 OR: sum_f n11 n00/N / sum_f n10 n01/N over fibers.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not put the script directory on the path
import stats as S
import why_errors as WE

B, SEED, CM, BIN = 4000, 7349, WE.CM, 32
SRC = S.SOURCES
POOLED = 'pooled'
PLAN_SHA = '61b47240219e6c58c5ded4b300f42ffc24178945aac99da257092de3cd96be54'
TERT_COVS = ('bend', 'kink', 'cross_d', 'par_d')
SEG_COVS = ('bend', 'kink', 'cross_d', 'par_d', 'junc_d', 'end_d')


# ------------------------------------------------------------------------------------------------------- statistics
def div(n, d):
    """n/d elementwise with x/0 = +inf (x > 0), 0/x = 0 and 0/0 = nan (undefined)."""
    n, d = np.asarray(n, float), np.asarray(d, float)
    out = np.full(np.broadcast(n, d).shape, np.nan)
    n, d = np.broadcast_to(n, out.shape), np.broadcast_to(d, out.shape)
    pos = d > 0
    out[pos] = n[pos]/d[pos]
    out[(d == 0) & (n > 0)] = np.inf
    return out


def pct(x, q):
    """Linear-interpolation percentile of a sorted array that may hold +inf (np.percentile would give nan for inf-inf)."""
    n = len(x)
    if not n:
        return np.nan
    pos = q/100*(n-1)
    i = int(np.floor(pos)); f = pos-i
    lo, hi = x[i], x[min(i+1, n-1)]
    return lo if (f == 0 or lo == hi) else (np.inf if np.isinf(hi) else lo+(hi-lo)*f)


def summarize(x):
    """x[0] = estimate on the data, x[1:] = bootstrap replicates -> estimate, CI, p, undefined count."""
    est, bs = x[0], np.asarray(x[1:], float)
    und = int(np.isnan(bs).sum())
    ok = np.sort(bs[~np.isnan(bs)])
    p = min(1.0, max(2*min((ok <= 1).mean(), (ok >= 1).mean()), 1/B)) if len(ok) else np.nan
    return dict(est=est, lo=pct(ok, 2.5), hi=pct(ok, 97.5), p=p, undefined=und)


def boot_weights(nf, rng):
    """(B, sum nf) multinomial fiber counts: replicate by replicate, scroll by scroll in order."""
    out = np.empty((B, sum(nf)))
    offs = np.r_[0, np.cumsum(nf)]
    pv = [np.full(n, 1.0/n) for n in nf]
    for r in range(B):
        for s, n in enumerate(nf):
            out[r, offs[s]:offs[s+1]] = rng.multinomial(n, pv[s])
    return out


def holm(p):
    """Holm step-down adjusted p-values (cumulative max of (m - i + 1) p_(i), capped at 1), in the input order."""
    p = np.asarray(p, float)
    o = np.argsort(p, kind='stable')
    adj = np.minimum(1.0, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[o]))
    out = np.empty(len(p)); out[o] = adj
    return out


def scroll_sums(M, Wall, blocks):
    """Weighted per-scroll sums of per-fiber table M (nfib, G) -> (R, S, G); row 0 of Wall is all ones (the data)."""
    return np.stack([Wall[:, sl] @ M[sl] for sl in blocks], 1)


def mh_rr(SA, SE, ex, ref):
    """Pooled Mantel-Haenszel RR over scrolls (R,) and crude per-scroll RR (R, S), exposed group ex vs reference ref."""
    a, E1, b, E0 = SA[:, :, ex], SE[:, :, ex], SA[:, :, ref], SE[:, :, ref]
    T = E0+E1
    t = np.where(T > 0, T, 1.0)
    num, den = np.where(T > 0, a*E0/t, 0.0).sum(1), np.where(T > 0, b*E1/t, 0.0).sum(1)
    return div(num, den), div(a*E0, b*E1)


# ------------------------------------------------------------------------------------------------------------ loading
def tertile(x, c1, c2):
    """0: x <= c1, 2: x >= c2, 1 otherwise, -1 for NaN."""
    x = np.asarray(x, float)
    return np.where(np.isnan(x), -1, np.where(x <= c1, 0, np.where(x >= c2, 2, 1))).astype(np.int8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Data:
    """Stage-2a tables as flat arrays. Segments (at-risk path pieces) and events carry the GT-point covariates, a global fiber
    index `gf` (scroll-major, fibers in covariates order) and the scroll index `si`."""

    def __init__(self, tab, plan):
        self.tab = tab
        self.san = json.loads((tab/'sanity.json').read_text())
        ex = np.load(tab/'exposure.npz')
        assert tuple(ex['sources']) == SRC
        self.cov = {s: dict(np.load(tab/f'covariates_{s}.npz')) for s in SRC}
        self.names = [list(self.cov[s]['fiber_names']) for s in SRC]
        self.nf = [len(n) for n in self.names]
        self.foff = np.r_[0, np.cumsum(self.nf)].astype(int)
        self.blocks = [slice(self.foff[i], self.foff[i+1]) for i in range(len(SRC))]
        self.fiber_name = np.array([n for ns in self.names for n in ns])
        ck = self.checks = dict(plan_sha256_ok=bool(sha(plan) == PLAN_SHA), table_sha256={f: sha(tab/f) for f in
                                ('exposure.npz', 'episodes.csv', 'traces.csv', 'sanity.json')})
        # ---- segments
        n = len(ex['src'])
        self.si, self.gf = ex['src'].astype(int), self.foff[ex['src']] + ex['fiber']
        self.len_vx, self.seed_d = ex['length'].astype(float), ex['seed_d'].astype(float)
        self.x = {k: np.full(n, np.nan) for k in SEG_COVS}
        self.fam = np.zeros(n, np.int8); self.tc = np.zeros(n, bool); self.tp = np.zeros(n, bool)
        ck['segments'] = n
        ck['exposure_cm_vs_sanity'] = {}
        for si, s in enumerate(SRC):
            m = np.flatnonzero(self.si == si)
            c = self.cov[s]
            assert ex['fiber'][m].min() >= 0 and ex['fiber'][m].max() < self.nf[si]
            row = WE.cov_rows(c, ex['fiber'][m], ex['pt'][m])
            assert (c['fiber'][row] == ex['fiber'][m]).all() and (c['pt'][row] == ex['pt'][m]).all(), s   # lookup keys match
            for k in SEG_COVS:
                self.x[k][m] = c[k][row]
            self.fam[m] = c['family'][row] == 'V'
            self.tc[m], self.tp[m] = c['trunc_cross'][row], c['trunc_par'][row]
            cm = self.len_vx[m].sum()*CM
            ck['exposure_cm_vs_sanity'][s] = [cm, self.san[s]['at_risk_cm'], abs(cm/self.san[s]['at_risk_cm']-1)]
        # ---- episodes
        eps = pd.read_csv(tab/'episodes.csv', float_precision='round_trip')
        for k in SEG_COVS:   # stage 2a wrote these float32 columns with float32 repr: casting back recovers the exact stored value
            eps[k] = eps[k].astype(np.float32).astype(float)
        eps['si'] = eps['src'].map({s: i for i, s in enumerate(SRC)})
        eps['gf'] = self.foff[eps['si']] + eps['fiber_idx']
        assert (self.fiber_name[eps['gf']] == eps['fiber_name']).all()
        assert (eps['gt_pt'] >= 0).all()
        bad = {}
        for si, s in enumerate(SRC):   # the CSV covariates are those of the npz at (fiber_idx, gt_pt)
            m = eps['si'].to_numpy() == si
            c = self.cov[s]
            row = WE.cov_rows(c, eps['fiber_idx'][m], eps['gt_pt'][m])
            assert (c['fiber'][row] == eps['fiber_idx'][m]).all() and (c['pt'][row] == eps['gt_pt'][m]).all()
            for k in SEG_COVS:
                bad[f'{s}/{k}'] = int((~np.isclose(c[k][row].astype(float), eps[k][m], rtol=0, atol=0, equal_nan=True)).sum())
            bad[f'{s}/family'] = int((c['family'][row] != eps['family'][m]).sum())
            bad[f'{s}/trunc'] = int((c['trunc_cross'][row] != eps['trunc_cross'][m]).sum()+(c['trunc_par'][row] != eps['trunc_par'][m]).sum())
        ck['episode_covariates_not_equal_to_npz_lookup'] = {k: v for k, v in bad.items() if v}
        self.eps = eps
        ck['episode_counts_match_stage2a'] = bool(all(int(((eps.src == s) & (eps.cls == c)).sum()) == WE.EXPECT[c][s]
                                                      for s in SRC for c in WE.EXPECT))
        # ---- traces
        tr = pd.read_csv(tab/'traces.csv', float_precision='round_trip')
        tr['si'] = tr['src'].map({s: i for i, s in enumerate(SRC)})
        idx = {(i, nm): j for i in range(len(SRC)) for j, nm in enumerate(self.names[i])}
        tr['gf'] = [self.foff[i] + idx[(int(i), nm)] for i, nm in zip(tr['si'], tr['fiber_name'])]   # KeyError if a fiber is not a covariate fiber
        ck['fibers_with_traces'] = {s: int(tr[tr.si == i].fiber_name.nunique()) for i, s in enumerate(SRC)}
        ck['fibers_in_covariates'] = dict(zip(SRC, self.nf))
        assert all(ck['fibers_with_traces'][s] == ck['fibers_in_covariates'][s] for s in SRC)
        # the exposure's trace ids agree with the traces table's fibers
        gtr = np.full((len(SRC), int(tr['trace'].max())+1), -1)
        gtr[tr['si'], tr['trace']] = tr['gf']
        got = gtr[self.si, ex['trace']]
        ck['exposure_trace_fiber_mismatch'] = int((got != self.gf).sum())
        self.tr = tr
        # ---- tertile bins
        self.cuts = {(s, k): (self.san[s]['covariates'][k]['quantiles']['33.3'], self.san[s]['covariates'][k]['quantiles']['66.7'])
                     for s in SRC for k in TERT_COVS}
        assert all(c1 < c2 for c1, c2 in self.cuts.values())
        ck['cut_points'] = {f'{s}/{k}': v for (s, k), v in self.cuts.items()}
        self.seg_t = {k: np.full(n, -1, np.int8) for k in TERT_COVS}
        self.ev_t = {k: np.full(len(eps), -1, np.int8) for k in TERT_COVS}
        for si, s in enumerate(SRC):
            ms, me = self.si == si, eps['si'].to_numpy() == si
            for k in TERT_COVS:
                self.seg_t[k][ms] = tertile(self.x[k][ms], *self.cuts[(s, k)])
                self.ev_t[k][me] = tertile(eps[k][me], *self.cuts[(s, k)])
        ck['nan_excluded'] = {
            k: dict(exposure_share={s: float(self.len_vx[(self.si == i) & (self.seg_t[k] < 0)].sum()/self.len_vx[self.si == i].sum()) for i, s in enumerate(SRC)},
                    hard_switch_events={s: int(((eps.src == s) & (eps.cls == 'hard_switch') & (self.ev_t[k] < 0)).sum()) for s in SRC})
            for k in TERT_COVS}
        h = eps[eps.cls == 'hard_switch']
        ck['hard_switch_at_annotation_end'] = {s: dict(n=int((h.src == s).sum()), end_d_eq_0=int(((h.src == s) & (h.end_d == 0)).sum()),
                                                        end_d_le_16=int(((h.src == s) & (h.end_d <= 16)).sum())) for s in SRC}


# --------------------------------------------------------------------------------------------- rate-ratio analyses
def fiber_table(gf, grp, nfib, G, w=None):
    """(nfib, G) sums of w (counts when None) by fiber and group; grp < 0 is excluded."""
    ok = grp >= 0
    return np.bincount(gf[ok]*G + grp[ok], weights=None if w is None else w[ok], minlength=nfib*G).reshape(nfib, G).astype(float)


class RateAnalysis:
    """Events A and exposure E per fiber and group; rate rows per group and contrast rows (MH RR) for the given contrasts."""

    def __init__(self, D, Wall, name, variant, labels, eg, vg, ev_mask):
        G = len(labels)
        self.D, self.Wall, self.name, self.variant, self.labels = D, Wall, name, variant, labels
        self.A = fiber_table(D.eps['gf'].to_numpy()[ev_mask], vg[ev_mask], len(D.fiber_name), G)
        self.E = fiber_table(D.gf, eg, len(D.fiber_name), G, D.len_vx)
        self.SA, self.SE = scroll_sums(self.A, Wall, D.blocks), scroll_sums(self.E, Wall, D.blocks)

    def rates(self):
        rows = []
        for sc, sl in [*zip(SRC, self.D.blocks), (POOLED, slice(None))]:
            for g, lab in enumerate(self.labels):
                ev, cm = self.A[sl, g].sum(), self.E[sl, g].sum()*CM
                rows.append(dict(row_type='rate', analysis=self.name, variant=self.variant, scroll=sc, group=lab, events=int(ev),
                                 exposure_cm=cm, rate_per_cm=float(div(ev, cm))))
        return rows

    def contrasts(self, cons):
        rows = []
        for label, ex, ref in cons:
            pooled, crude = mh_rr(self.SA, self.SE, ex, ref)
            for sc, x, s in [(POOLED, pooled, slice(None)), *[(SRC[i], crude[:, i], i) for i in range(len(SRC))]]:
                r = summarize(x)
                rows.append(dict(row_type='contrast', analysis=self.name, variant=self.variant, contrast=label, scroll=sc,
                                 estimator='MH RR over scrolls' if sc == POOLED else 'crude RR', est=r['est'], lo=r['lo'], hi=r['hi'], p_raw=r['p'],
                                 undefined=r['undefined'], events_exposed=self.SA[0, s, ex].sum(), events_ref=self.SA[0, s, ref].sum(),
                                 exposure_cm_exposed=self.SE[0, s, ex].sum()*CM, exposure_cm_ref=self.SE[0, s, ref].sum()*CM))
        return rows


# --------------------------------------------------------------------------------------------------------- Q1 place
def trace_bins(sign, t0, reach, term, dep, w, persistent, censor=False):
    """(k0, out): outcome of one trace in bins k0.. of width w. 1 switch, 0 pass, -1 not entered or censored.
    p(x) = (x - t0) sign; bin k is entered when 0 < p(entry) <= R = p(reach); done in arc space (same inequalities, no rounding).
    censor (stage 2c, post-hoc): the terminal switch is not counted; the trace is censored (-1) in the bin it would have switched in."""
    sw = term == 'hard_switch' or (persistent and term == 'drift')
    lo, hi = min(t0, reach), max(t0, reach)
    if sw:
        lo, hi = min(lo, dep), max(hi, dep)
    k = np.arange(int(np.floor(lo/w))-1, int(np.floor(hi/w))+2)
    a, b = w*k, w*(k+1)
    if sign > 0:
        entered, passed = (a > t0) & (a <= reach), b <= reach
    else:
        entered, passed = (b < t0) & (b >= reach), a >= reach
    isw = sw & (np.floor(dep/w) == k)
    return k[0], np.where(entered, np.where(isw, -1 if censor else 1, np.where(passed, 0, -1)), -1)


def q1_counts(D, w=BIN, persistent=False, same_dir=False, censor_past_end=False):
    """Per-fiber 2x2 cell counts. Returns dict of (nfib,) arrays n11 n10 n01 n00 pairs pairs_cells (+ per-bin detail for hotspots).
    Same-direction pairs are unordered, so discordant cells are split evenly: n10 = n01 = discordant / 2.
    censor_past_end (stage 2c, post-hoc): a trace whose terminal switch has D.tr.term_pe is censored in its switch bin."""
    nfib = len(D.fiber_name)
    c = {k: np.zeros(nfib) for k in ('n11', 'n10', 'n01', 'n00', 'pairs', 'pairs_cells')}
    detail, n_hard, n_hard_decided = {}, 0, 0
    tr = D.tr[D.tr['reach_arc'].notna()]
    for f, g in tr.groupby('gf'):
        outs = [trace_bins(r.sign, r.t0, r.reach_arc, r.terminal, r.terminal_dep_arc, w, persistent, bool(censor_past_end and r.term_pe))
                for r in g.itertuples()]
        k0 = min(o[0] for o in outs); k1 = max(o[0]+len(o[1]) for o in outs)
        O = np.full((len(outs), k1-k0), -1, np.int8)
        for i, (k, o) in enumerate(outs):
            O[i, k-k0:k-k0+len(o)] = o
        sg = g['sign'].to_numpy()
        if same_dir:
            pairs = [(i, j) for s_ in (1.0, -1.0) for i in np.flatnonzero(sg == s_) for j in np.flatnonzero(sg == s_) if i < j]
        else:
            pairs = [(i, j) for i in np.flatnonzero(sg > 0) for j in np.flatnonzero(sg < 0)]
        n11 = n10 = n01 = n00 = np.zeros(k1-k0)
        for i, j in pairs:
            a, b = O[i], O[j]
            both = (a >= 0) & (b >= 0)
            c['pairs_cells'][f] += both.any()
            n11 = n11 + (both & (a == 1) & (b == 1))
            n00 = n00 + (both & (a == 0) & (b == 0))
            if same_dir:
                d = (both & (a != b))/2.0
                n10, n01 = n10 + d, n01 + d
            else:
                n10 = n10 + (both & (a == 1) & (b == 0))
                n01 = n01 + (both & (a == 0) & (b == 1))
        c['pairs'][f] = len(pairs)
        for k, v in (('n11', n11), ('n10', n10), ('n01', n01), ('n00', n00)):
            c[k][f] = v.sum()
        if not same_dir and not persistent:
            hs = (g['terminal'] == 'hard_switch').to_numpy()
            n_hard += int(hs.sum()); n_hard_decided += int(((O[hs] == 1).any(1)).sum())
            if n11.sum() > 0:
                for kk in np.flatnonzero(n11 > 0):
                    detail[(f, int(k0+kk))] = dict(n11=int(n11[kk]), n_A_switch=int((O[sg > 0, kk] == 1).sum()), n_B_switch=int((O[sg < 0, kk] == 1).sum()))
    c['detail'] = detail
    c['hard_switch_traces'], c['hard_switch_traces_with_a_decided_switch_bin'] = n_hard, n_hard_decided
    return c


def q1_rows(D, Wall, c, variant, bin_vx):
    N = c['n11']+c['n10']+c['n01']+c['n00']
    t = np.where(N > 0, 1.0, 0.0)
    nz = np.where(N > 0, N, 1.0)
    tn, td = t*c['n11']*c['n00']/nz, t*c['n10']*c['n01']/nz
    rows = []
    for sc, sl in [(POOLED, slice(None)), *zip(SRC, D.blocks)]:
        r = summarize(div(Wall[:, sl] @ tn[sl], Wall[:, sl] @ td[sl]))
        rows.append(dict(row_type='q1', analysis='Q1', variant=variant, bin_vx=bin_vx, scroll=sc,
                         pairs=int(c['pairs'][sl].sum()), pairs_with_cells=int(c['pairs_cells'][sl].sum()), cells=float(N[sl].sum()),
                         n11=float(c['n11'][sl].sum()), n10=float(c['n10'][sl].sum()), n01=float(c['n01'][sl].sum()),
                         n00=float(c['n00'][sl].sum()), est=r['est'], lo=r['lo'], hi=r['hi'], p_raw=r['p'], undefined=r['undefined']))
    return rows


# --------------------------------------------------------------------------------------------------------- S7 hotspots
def hotspots(D, detail, data, fibers, villa):
    WE.init_villa(villa)
    rows = []
    for si, s in enumerate(SRC):
        mine = sorted((f, k, v) for (f, k), v in detail.items() if D.foff[si] <= f < D.foff[si+1])
        if not mine:
            continue
        cfg = json.loads((WE.run_dir(data, s)/'config.json').read_text())
        gs = float(cfg['grid_scale'])
        gt, _ = WE.load_gt(Path(fibers)/Path(cfg['fibers_dir']).name, gs)
        c = D.cov[s]
        for f, k, v in mine:
            i = f-D.foff[si]
            fb = gt[D.names[si][i]]
            lo_, hi_ = c['offsets'][i], c['offsets'][i+1]
            assert len(fb.s) == hi_-lo_ and np.array_equal(fb.s, c['s'][lo_:hi_]), (s, i)
            centre = BIN*(k+0.5)
            pt = int(WE.arc_to_idx(fb.s, np.array([centre]))[0])
            xyz = fb.points[pt]
            row = lo_+pt
            rows.append(dict(scroll=s, fiber=D.names[si][i], bin=k, arc_lo=BIN*k, arc_hi=BIN*(k+1), centre_arc=centre, gt_point=pt, gt_arc=float(fb.s[pt]),
                             x=xyz[0], y=xyz[1], z=xyz[2], x_full=xyz[0]*gs, y_full=xyz[1]*gs, z_full=xyz[2]*gs,
                             **{k_: float(c[k_][row]) for k_ in SEG_COVS}, n_cells_both_switched=v['n11'],
                             n_A_traces_switched=v['n_A_switch'], n_B_traces_switched=v['n_B_switch']))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------- main
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return None if np.isnan(o) else ('inf' if o == np.inf else ('-inf' if o == -np.inf else o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(data, fibers, villa, out):
    t_start = time.time()
    out = Path(out); res_dir = out/'results'; res_dir.mkdir(parents=True, exist_ok=True)
    D = Data(out/'tables', Path(__file__).resolve().parent.parent/'ANALYSIS_PLAN_v3.md')
    print(f'tables loaded ({time.time()-t_start:.0f}s); checks: plan hash ok {D.checks["plan_sha256_ok"]}, trace/fiber mismatch '
          f'{D.checks["exposure_trace_fiber_mismatch"]}, episode cov mismatch {D.checks["episode_covariates_not_equal_to_npz_lookup"]}', flush=True)
    rng = np.random.default_rng(SEED)
    Wall = np.vstack([np.ones((1, len(D.fiber_name))), boot_weights(D.nf, rng)])
    print(f'weights {Wall.shape} ({time.time()-t_start:.0f}s)', flush=True)
    eps, hard = D.eps, (D.eps['cls'] == 'hard_switch').to_numpy()
    ret = (D.eps['cls'] == 'returned').to_numpy()

    def tert_analysis(name, cov, variant='', mask=None, seg_mask=None, ev=hard):
        eg = D.seg_t[cov].copy(); vg = D.ev_t[cov].copy()
        if seg_mask is not None:
            eg[seg_mask] = -1
        if mask is not None:
            vg[mask] = -1
        return RateAnalysis(D, Wall, name, variant, ['bin0', 'bin1', 'bin2'], eg, vg, ev)

    # ---- Q1
    q1 = q1_counts(D)
    q1_primary = q1_rows(D, Wall, q1, 'primary: 32 vx, hard switches', BIN)
    sens_rows = []
    for w, pers, same, lab in [(16, False, False, '16 vx bins'), (64, False, False, '64 vx bins'),
                               (BIN, True, False, 'persistent (drift counts as switch), 32 vx'),
                               (BIN, False, True, 'same-direction pairs, 32 vx (n10 = n01 = discordant/2)')]:
        sens_rows += q1_rows(D, Wall, q1_counts(D, w, pers, same), lab, w)
    print(f'Q1 done ({time.time()-t_start:.0f}s)', flush=True)

    # ---- Q2-Q5
    fam_e = D.eps['family'].to_numpy() == 'V'
    qa = {'Q2': tert_analysis('Q2 bend', 'bend'), 'Q3': tert_analysis('Q3 crossing', 'cross_d'), 'Q4': tert_analysis('Q4 parallel', 'par_d'),
          'Q5': RateAnalysis(D, Wall, 'Q5 family', '', ['H', 'V'], D.fam.astype(int), fam_e.astype(int), hard)}
    qcon = {'Q2': [('bin2 vs bin0', 2, 0)], 'Q3': [('bin0 (nearest) vs bin2', 0, 2)], 'Q4': [('bin0 (nearest) vs bin2', 0, 2)], 'Q5': [('V vs H', 1, 0)]}
    prim = []
    for q in qa:
        prim += qa[q].contrasts(qcon[q])
    pooled = [r for r in prim if r['scroll'] == POOLED]
    q1_pool = next(r for r in q1_primary if r['scroll'] == POOLED)
    padj = holm([q1_pool['p_raw']] + [r['p_raw'] for r in pooled])   # Q1, Q2, Q3, Q4, Q5
    holm_of = {'Q1': padj[0], **{r['analysis'][:2]: padj[i+1] for i, r in enumerate(pooled)}}
    drop = ('row_type', 'variant')
    ptab = [dict(test='Q1', contrast='OR, 32 vx bins, hard switches', estimator='MH OR over fibers', **{k: v for k, v in r.items() if k not in drop + ('bin_vx',)})
            for r in q1_primary] + [dict(test=r['analysis'][:2], **{k: v for k, v in r.items() if k not in drop}) for r in prim]
    for r in ptab:
        r['p_holm'] = holm_of[r['test']] if r['scroll'] == POOLED else np.nan
        if r['scroll'] != POOLED:
            r['p_raw'] = np.nan
    primary = pd.DataFrame(ptab)
    primary = primary.sort_values('test', kind='stable', key=lambda c: c.map({'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4, 'Q5': 5})).reset_index(drop=True)
    print(f'Q2-Q5 done ({time.time()-t_start:.0f}s)', flush=True)

    # ---- sensitivity Q3 / Q4: drop GT points flagged as truncated neighbour search
    for q, cov, flag_s, col, nm, cname in (('Q3', 'cross_d', D.tc, 'trunc_cross', 'Q3 crossing', 'cross'), ('Q4', 'par_d', D.tp, 'trunc_par', 'Q4 parallel', 'par')):
        a = tert_analysis(nm, cov, variant=f'excluding {col} points', mask=eps[col].to_numpy(bool), seg_mask=flag_s)
        sens_rows += a.contrasts(qcon[q]) + a.rates()

    # ---- secondary
    sec = []
    a = tert_analysis('S2 kink, hard switches', 'kink', ev=hard); sec += a.rates() + a.contrasts([('bin2 vs bin0', 2, 0)])
    a = tert_analysis('S2 kink, returned excursions', 'kink', ev=ret); sec += a.rates() + a.contrasts([('bin2 vs bin0', 2, 0)])
    jg_s = np.where(D.x['junc_d'] <= 16, 0, 1); jg_e = np.where(eps['junc_d'].to_numpy() <= 16, 0, 1)
    a = RateAnalysis(D, Wall, 'S3 junction, hard switches', '', ['junc_d <= 16', 'junc_d > 16'], jg_s, jg_e, hard)
    sec += a.rates() + a.contrasts([('<= 16 vs > 16', 0, 1)])
    band = lambda x: np.where(x < 100, 0, np.where(x <= 400, 1, 2))
    a = RateAnalysis(D, Wall, 'S4 seed distance, hard switches', '', ['< 100', '100-400', '> 400'], band(D.seed_d), band(eps['seed_d'].to_numpy()), hard)
    sec += a.rates() + a.contrasts([('100-400 vs <100', 1, 0), ('>400 vs <100', 2, 0)])
    print(f'secondary done ({time.time()-t_start:.0f}s)', flush=True)

    # ---- outputs
    rates = pd.DataFrame([r for q in qa.values() for r in q.rates()]).drop(columns=['row_type'])
    q1_df = pd.DataFrame(q1_primary).drop(columns=['row_type'])
    sensitivity, secondary = (pd.DataFrame(r).drop(columns=['p_raw']) for r in (sens_rows, sec))   # estimates and CIs only, no p-values
    q1_df.to_csv(res_dir/'q1_place.csv', index=False)
    rates.to_csv(res_dir/'q2_q5_rates.csv', index=False)
    primary.to_csv(res_dir/'primary_tests.csv', index=False)
    sensitivity.to_csv(res_dir/'sensitivity.csv', index=False)
    secondary.to_csv(res_dir/'secondary.csv', index=False)
    hs = hotspots(D, q1['detail'], data, fibers, villa)
    hs.to_csv(res_dir/'hotspots.csv', index=False)
    print(f'hotspots {len(hs)} ({time.time()-t_start:.0f}s)', flush=True)
    D.checks['q1_hard_switch_traces'] = dict(total=q1['hard_switch_traces'], with_a_decided_switch_bin=q1['hard_switch_traces_with_a_decided_switch_bin'])
    full = dict(command=' '.join(sys.argv), script_sha256=sha(__file__), B=B, seed=SEED, checks=D.checks,
                tables={n: df.to_dict('records') for n, df in dict(q1_place=q1_df, q2_q5_rates=rates, primary_tests=primary, sensitivity=sensitivity,
                                                                  secondary=secondary).items()},
                n_hotspots=len(hs), runtime_s=time.time()-t_start, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (res_dir/'results_2b.json').write_text(json.dumps(clean(full), indent=1))
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
    print(primary.round(4).to_string(index=False))
    print(f'runtime {full["runtime_s"]:.0f}s peak RSS {full["peak_rss_mb"]:.0f} MB')


# ======================================================================================= stage 2c (post-hoc, see main_2c)
GROUPS_2C = ('unseen', 'bend', 'kink', 'fam', 'seed', 'end')
UNSEEN = (1, 2, 3)   # indices in SRC of PHerc0139, PHerc1451, PHerc0332; Paris4 = 0
CELL_SHAPE = (len(SRC), 4, 4, 2, 3, 2)   # scroll, bend (3 tertiles + NaN), kink, family V, seed band, end band


def load_flags(D, res_dir):
    """Merge episode_flags.csv into D.eps (order, past_end, end_tagged, crossing_len) and D.tr (term_pe: the terminal hard switch is past_end).
    Every hard_switch / drift / ended episode and every first episode must have a flag row; nothing else may."""
    fl = pd.read_csv(res_dir/'episode_flags.csv', float_precision='round_trip')
    eps = D.eps
    eps['order'] = eps.groupby(['src', 'trace']).cumcount()
    key = ['src', 'trace', 'order']
    m = eps[key+['cls', 'onset_idx', 'end_d']].merge(fl, on=key, how='left', suffixes=('', '_f'), indicator=True, validate='one_to_one')
    assert len(m) == len(eps) and (m[key].to_numpy() == eps[key].to_numpy()).all()
    need = (eps['cls'].isin(WE.TERMINAL) | (eps['order'] == 0)).to_numpy()
    has = (m['_merge'] == 'both').to_numpy()
    assert (has == need).all(), (int((need & ~has).sum()), int((has & ~need).sum()))
    assert len(fl) == need.sum() and (m.loc[has, 'cls'] == m.loc[has, 'cls_f']).all() and (m.loc[has, 'onset_idx'] == m.loc[has, 'onset_idx_f']).all()
    assert np.array_equal(m.loc[has, 'end_d'].to_numpy(np.float32), m.loc[has, 'end_d_f'].to_numpy(np.float32))   # D.eps holds float32-cast values
    eps['past_end'] = np.where(has, m['past_end'].fillna(False), False).astype(bool)
    eps['end_tagged'] = np.where(has & (eps['cls'] != 'returned').to_numpy(), m['end_tagged'].fillna(False), False).astype(bool)
    eps['crossing_len'] = m['crossing_len'].to_numpy()
    eps['is_first'] = (eps['order'] == 0).to_numpy()
    assert not (eps['past_end'] & (eps['cls'] == 'returned')).any()
    t = eps[(eps['cls'] == 'hard_switch').to_numpy()]
    pe = {(a, b): c for a, b, c in zip(t['src'], t['trace'], t['past_end'])}
    D.tr['term_pe'] = [pe.get((a, b), False) for a, b in zip(D.tr['src'], D.tr['trace'])]
    return fl


def flag_counts(eps):
    """Section 1 tables: episodes by scroll x class (terminal classes and first episodes), past_end, tagged ends, end_d = 0 hard switches."""
    rows = []
    for s in SRC:
        e = eps[eps['src'] == s]
        for c in WE.TERMINAL:
            g = e[e['cls'] == c]
            rows.append(dict(scroll=s, group=f'terminal {c}', n=len(g), past_end=int(g['past_end'].sum()), end_tagged=int(g['end_tagged'].sum()),
                             past_end_end_d_gt_16=int((g['past_end'] & (g['end_d'] > 16)).sum())))
        for c in ('returned', *WE.TERMINAL):
            g = e[e['is_first'] & (e['cls'] == c)]
            rows.append(dict(scroll=s, group=f'first episode {c}', n=len(g), past_end=int(g['past_end'].sum()), end_tagged=int(g['end_tagged'].sum()),
                             past_end_end_d_gt_16=int((g['past_end'] & (g['end_d'] > 16)).sum())))
        h = e[e['cls'] == 'hard_switch']
        z = h[h['end_d'] == 0]
        rows.append(dict(scroll=s, group='hard_switch with end_d = 0', n=len(z), past_end=int(z['past_end'].sum()), end_tagged=int(z['end_tagged'].sum()),
                         past_end_end_d_gt_16=0, not_past_end=int((~z['past_end']).sum())))
    out = pd.DataFrame(rows)
    tot = out.groupby('group', sort=False)[['n', 'past_end', 'end_tagged', 'past_end_end_d_gt_16', 'not_past_end']].sum(min_count=1).reset_index()
    tot.insert(0, 'scroll', 'all')
    return pd.concat([out, tot], ignore_index=True)


# -------------------------------------------------------------------------------------------- Poisson (exact MLE by IRLS)
def pdev(y, mu):
    """Poisson deviance 2 sum [y log(y/mu) - (y - mu)]."""
    return 2.0*float((np.where(y > 0, y*np.log(np.where(y > 0, y, 1.0)/mu), 0.0) - (y-mu)).sum())


def irls(X, y, off, maxit=200, tol=1e-10, cap=30.0):
    """Newton / IRLS for log E[y] = X b + off with step halving; returns (b, converged, iterations, max |X'(y - mu)|).
    No finite MLE exists when some coefficient runs off to -inf (separation, e.g. all events of one level also lie in one other level):
    IRLS then stops with converged = False when a coefficient passes `cap` in absolute value, the fit is singular or not finite."""
    with np.errstate(all='ignore'):
        b = np.zeros(X.shape[1])
        b[0] = np.log(y.sum()/np.exp(off).sum())   # intercept column first
        eta = X @ b + off
        mu = np.exp(eta)
        dev = pdev(y, mu)
        for it in range(maxit):
            XW = X.T*mu
            try:
                bn = np.linalg.solve(XW @ X, XW @ (eta-off+(y-mu)/mu))
            except np.linalg.LinAlgError:
                return b, False, it, np.nan
            if not np.isfinite(bn).all():
                return b, False, it, np.nan
            step = 1.0
            while True:
                bt = b+step*(bn-b)
                et = X @ bt + off
                mt = np.exp(et)
                dt = pdev(y, mt)
                if (np.isfinite(dt) and dt <= dev+1e-12*abs(dev)) or step < 1e-8:
                    break
                step *= 0.5
            if not np.isfinite(dt) or np.abs(bt).max() > cap:
                return b, False, it, np.nan
            db = float(np.abs(bt-b).max())
            b, eta, mu, dev = bt, et, mt, dt
            if db < tol:
                return b, True, it+1, float(np.abs(X.T @ (y-mu)).max())
        return b, False, maxit, float(np.abs(X.T @ (y-mu)).max())


def fit_cells(A, E, lab, factors, ref0=()):
    """Main-effects Poisson model on cells (events A, exposure E in cm, level codes lab[f]) with offset log E; exact MLE.

    Cells without exposure are left out. A level with no event has fitted rate 0 (coefficient -inf): its cells (all with y = 0) are
    left out, which leaves the likelihood and the other coefficients unchanged. The reference level of a factor is its lowest level
    present (always level 0 for the factors in ref0). Returns dict(coef {name: value}, converged, lost_events (events in cells
    without exposure), dropped_levels, score)."""
    keep = E > 0
    lost = float(A[~keep].sum())
    dropped = []
    for f in factors:
        for l in np.unique(lab[f][keep]):
            if A[keep & (lab[f] == l)].sum() == 0:
                keep &= lab[f] != l
                dropped.append(f'{f}={l}')
    cols, names = [np.ones(int(keep.sum()))], ['intercept']
    for f in factors:
        lv = np.unique(lab[f][keep])
        ref = 0 if (f in ref0 and 0 in lv) else lv[0]
        for l in lv:
            if l != ref:
                cols.append((lab[f][keep] == l).astype(float)); names.append(f'{f}={l}')
    X = np.column_stack(cols)
    b, ok, it, sc = irls(X, A[keep], np.log(E[keep]))
    return dict(coef=dict(zip(names, b)), converged=ok, lost_events=lost, dropped_levels=dropped, score=sc, X=X, y=A[keep], off=np.log(E[keep]), keep=keep)


def band_seed(x):
    return np.where(x < 100, 0, np.where(x <= 400, 1, 2))


def cell_codes(scroll, bend_t, kink_t, fam, seed_d, end_d):
    """Cell index of (scroll, bend tertile + NaN level, kink tertile + NaN level, family V, seed band, end band >= 32)."""
    t = lambda a: np.where(a < 0, 3, a)
    return np.ravel_multi_index((scroll, t(bend_t), t(kink_t), fam.astype(int), band_seed(seed_d), (end_d >= 32).astype(int)), CELL_SHAPE)


def cell_labels():
    sc, be, ki, fa, se, en = np.unravel_index(np.arange(int(np.prod(CELL_SHAPE))), CELL_SHAPE)
    return dict(scroll=sc, unseen=(sc > 0).astype(int), bend=be, kink=ki, fam=fa, seed=se, end=en)


def boot_poisson(Ar, Er, lab, factors, ref0, keys):
    """Fit the model on every replicate; returns {key: (R,) exp(coef), nan where the fit did not converge} plus counts of non-converged /
    dropped-level / lost-event fits."""
    R = len(Er)
    out = {k: np.full(R, np.nan) for k in keys}
    nc = nd = nl = 0
    for r in range(R):
        f = fit_cells(Ar[r], Er[r], lab, factors, ref0)
        nc += not f['converged']; nd += bool(f['dropped_levels']); nl += f['lost_events'] > 0
        for k in keys:
            if f['converged'] and k in f['coef']:   # no finite MLE -> undefined replicate (counted, as in stage 2b)
                out[k][r] = np.exp(f['coef'][k])
    return out, dict(not_converged=int(nc), with_dropped_levels=int(nd), with_lost_events=int(nl))


# ----------------------------------------------------------------------------------------- stage 2c: end-censored rerun
def analyses_2b(D, Wall, hard, q1):
    """Q1 (32 vx, hard), Q2-Q5, S2, S3, S4 exactly as in main(), with the hard-switch event mask `hard` and the Q1 cell counts `q1`.
    Returns a DataFrame of rate / contrast / q1 rows; p_raw and p_holm only on the pooled Q1-Q5 rows (as primary_tests.csv)."""
    eps = D.eps
    ret = (eps['cls'] == 'returned').to_numpy()
    tert = lambda name, cov, ev: RateAnalysis(D, Wall, name, '', ['bin0', 'bin1', 'bin2'], D.seg_t[cov].copy(), D.ev_t[cov].copy(), ev)
    fam_e = eps['family'].to_numpy() == 'V'
    qa = {'Q2': tert('Q2 bend', 'bend', hard), 'Q3': tert('Q3 crossing', 'cross_d', hard), 'Q4': tert('Q4 parallel', 'par_d', hard),
          'Q5': RateAnalysis(D, Wall, 'Q5 family', '', ['H', 'V'], D.fam.astype(int), fam_e.astype(int), hard)}
    qcon = {'Q2': [('bin2 vs bin0', 2, 0)], 'Q3': [('bin0 (nearest) vs bin2', 0, 2)], 'Q4': [('bin0 (nearest) vs bin2', 0, 2)], 'Q5': [('V vs H', 1, 0)]}
    rows = q1_rows(D, Wall, q1, 'primary: 32 vx, hard switches', BIN)
    prim = []
    for q in qa:
        c = qa[q].contrasts(qcon[q])
        prim += c
        rows += qa[q].rates() + c
    q1_pool = next(r for r in rows if r['row_type'] == 'q1' and r['scroll'] == POOLED)
    pooled = [r for r in prim if r['scroll'] == POOLED]
    padj = holm([q1_pool['p_raw']] + [r['p_raw'] for r in pooled])
    holm_of = {'Q1': padj[0], **{r['analysis'][:2]: padj[i+1] for i, r in enumerate(pooled)}}
    for r in rows:
        if r['row_type'] != 'rate' and r['scroll'] == POOLED:
            r['p_holm'] = holm_of[r['analysis'][:2]]
        else:
            r['p_raw'] = np.nan   # 2b kept p-values for the pooled primary tests only
    jg_s = np.where(D.x['junc_d'] <= 16, 0, 1); jg_e = np.where(eps['junc_d'].to_numpy() <= 16, 0, 1)
    sec = []
    for a, con in ((tert('S2 kink, hard switches', 'kink', hard), [('bin2 vs bin0', 2, 0)]),
                   (tert('S2 kink, returned excursions', 'kink', ret), [('bin2 vs bin0', 2, 0)]),
                   (RateAnalysis(D, Wall, 'S3 junction, hard switches', '', ['junc_d <= 16', 'junc_d > 16'], jg_s, jg_e, hard), [('<= 16 vs > 16', 0, 1)]),
                   (RateAnalysis(D, Wall, 'S4 seed distance, hard switches', '', ['< 100', '100-400', '> 400'], band_seed(D.seed_d), band_seed(eps['seed_d'].to_numpy()), hard),
                    [('100-400 vs <100', 1, 0), ('>400 vs <100', 2, 0)])):
        sec += a.rates() + a.contrasts(con)
    for r in sec:
        r['p_raw'] = np.nan
    df = pd.DataFrame(rows + sec)
    df['test'] = df['analysis'].str[:2]
    return df


KEYS_2C = ['row_type', 'analysis', 'variant', 'scroll', 'group', 'contrast', 'estimator']
VALS_2C = ['events', 'exposure_cm', 'rate_per_cm', 'est', 'lo', 'hi', 'p_raw', 'p_holm', 'undefined', 'events_exposed', 'events_ref',
           'exposure_cm_exposed', 'exposure_cm_ref', 'pairs', 'pairs_with_cells', 'cells', 'n11', 'n10', 'n01', 'n00']


def check_against_2b(df, res_dir):
    """The uncensored rerun must equal the stage-2b CSVs: rows compared, rows on one side only, max abs difference (inf if a nan meets a number)."""
    prim = pd.read_csv(res_dir/'primary_tests.csv', float_precision='round_trip')
    sec = pd.read_csv(res_dir/'secondary.csv', float_precision='round_trip')
    rat = pd.read_csv(res_dir/'q2_q5_rates.csv', float_precision='round_trip')

    def diff(a, b, cols):
        m = a.merge(b, on='k', suffixes=('_a', '_b'), how='outer', indicator=True, validate='one_to_one')
        both = m[m['_merge'] == 'both']
        dmax = 0.0
        for c in cols:
            x, y = both[c+'_a'].to_numpy(float), both[c+'_b'].to_numpy(float)
            d = np.where((x == y) | (np.isnan(x) & np.isnan(y)), 0.0, np.abs(x-y))
            dmax = max(dmax, np.inf if np.isnan(d).any() else (float(d.max()) if len(d) else 0.0))
        return dict(rows_compared=len(both), only_in_rerun=int((m['_merge'] == 'left_only').sum()), only_in_2b=int((m['_merge'] == 'right_only').sum()), max_abs_diff=dmax)
    out = {}
    cols = ['est', 'lo', 'hi', 'p_raw', 'p_holm', 'undefined']
    a = df[df['test'].isin(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']) & (df['row_type'] != 'rate')].copy()
    a['k'] = a['test']+'|'+a['scroll']+'|'+np.where(a['test'] == 'Q1', '', a['contrast'].fillna(''))
    b = prim.copy(); b['k'] = b['test']+'|'+b['scroll']+'|'+np.where(b['test'] == 'Q1', '', b['contrast'])
    out['primary_tests'] = diff(a[['k']+cols], b[['k']+cols], cols)
    cols = ['events', 'exposure_cm', 'rate_per_cm']
    a = df[(df['row_type'] == 'rate') & df['test'].isin(['Q2', 'Q3', 'Q4', 'Q5'])].copy(); a['k'] = a['analysis']+'|'+a['scroll']+'|'+a['group']
    b = rat.copy(); b['k'] = b['analysis']+'|'+b['scroll']+'|'+b['group']
    out['q2_q5_rates'] = diff(a[['k']+cols], b[['k']+cols], cols)
    cols = ['events', 'exposure_cm', 'rate_per_cm', 'est', 'lo', 'hi', 'undefined', 'events_exposed', 'events_ref', 'exposure_cm_exposed', 'exposure_cm_ref']
    key = lambda d: d[['row_type', 'analysis', 'scroll', 'group', 'contrast']].fillna('').astype(str).agg('|'.join, axis=1)
    a = df[df['test'].isin(['S2', 'S3', 'S4'])].copy(); a['k'] = key(a)
    b = sec.copy(); b['k'] = key(b)
    out['secondary'] = diff(a[['k']+cols], b[['k']+cols], cols)
    return out


def side_by_side(orig, cens):
    for d in (orig, cens):
        for k in KEYS_2C:
            d[k] = d[k].fillna('').astype(str)
    cols = KEYS_2C+VALS_2C
    m = orig.reindex(columns=cols).merge(cens.reindex(columns=cols), on=KEYS_2C, how='outer', suffixes=('_orig', '_cens'), sort=False, validate='one_to_one')
    return m


# ---------------------------------------------------------------------------------- stage 2c: strict first departures (v2 headline, S1)
def strict_tables(D, res_dir):
    """Strict exposure segments (stage 2c --flags) with the covariates at their GT point, and the first-episode events.
    Returns (seg dict, ev dict) with gf, scroll index, cell code, length (vx) / past_end."""
    ex = np.load(res_dir/'strict_exposure.npz')
    assert tuple(ex['sources']) == SRC
    si = ex['src'].astype(int)
    n = len(si)
    bend, kink, fam, endd = (np.full(n, np.nan) for _ in range(4))
    gf = D.foff[si]+ex['fiber']
    for k, s in enumerate(SRC):
        m = np.flatnonzero(si == k)
        c = D.cov[s]
        row = WE.cov_rows(c, ex['fiber'][m], ex['pt'][m])
        assert (c['fiber'][row] == ex['fiber'][m]).all() and (c['pt'][row] == ex['pt'][m]).all()
        bend[m], kink[m], endd[m] = c['bend'][row], c['kink'][row], c['end_d'][row]
        fam[m] = c['family'][row] == 'V'
    return dict(si=si, gf=gf, len_vx=ex['length'].astype(float), bend=bend, kink=kink, fam=fam, end_d=endd, seed_d=ex['seed_d'].astype(float),
                trace=ex['trace'])


def pooled_tertile(D, k, x):
    c1, c2 = D.san[POOLED]['covariates'][k]['quantiles']['33.3'], D.san[POOLED]['covariates'][k]['quantiles']['66.7']
    return tertile(x, c1, c2), (c1, c2)


def strict_headline(D, Wall, seg, first, variants):
    """Rate per cm per scroll and unseen pooled, crude RR vs Paris4 per scroll, and the unseen/Paris4 rate ratio, for events = rows of `first`
    selected by each variant mask (code, label, mask) over the exposure segments `seg` (fiber bootstrap with the weights Wall)."""
    nf = len(D.fiber_name)
    scr = lambda M: scroll_sums(M[:, None], Wall, D.blocks)[:, :, 0]   # (R, S)
    SE_ = scr(np.bincount(seg['gf'], weights=seg['len_vx'], minlength=nf)*CM)   # cm
    sh = []
    for var, lab, m in variants:
        SA = scr(np.bincount(first['gf'].to_numpy()[m], minlength=nf).astype(float))
        rate = div(SA, SE_)
        ru = div(SA[:, list(UNSEEN)].sum(1), SE_[:, list(UNSEEN)].sum(1))
        ratio = div(ru, rate[:, 0])
        for name, x, ev, ex in [(SRC[i], rate[:, i], SA[0, i], SE_[0, i]) for i in range(len(SRC))] + [('unseen pooled', ru, SA[0, 1:].sum(), SE_[0, 1:].sum())]:
            r = summarize(x)
            sh.append(dict(variant=var, label=lab, row='rate per cm', scroll=name, events=ev, exposure_cm=ex, est=r['est'], lo=r['lo'], hi=r['hi'], p=np.nan, undefined=r['undefined']))
        for i in range(1, len(SRC)):
            r = summarize(div(rate[:, i], rate[:, 0]))
            sh.append(dict(variant=var, label=lab, row='crude RR vs Paris4', scroll=SRC[i], events=SA[0, i], exposure_cm=SE_[0, i], est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'], undefined=r['undefined']))
        r = summarize(ratio)
        sh.append(dict(variant=var, label=lab, row='rate ratio unseen/Paris4', scroll='unseen pooled', events=SA[0, 1:].sum(), exposure_cm=SE_[0, 1:].sum(), est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'], undefined=r['undefined']))
    return pd.DataFrame(sh)


def s1_poisson(D, Wall, seg, first, variants, fe=()):
    """S1: Poisson model (log-exposure offset, exact MLE) for the events `first` selected by each variant mask, over exposure `seg`, cells = unseen x
    bend tertile (+NaN) x kink tertile (+NaN) x family x seed band x end band, pooled cut points, main effects. Crude and adjusted exp(beta_unseen)
    with bootstrap CI for every variant; adjusted per-scroll ratios vs Paris4 (scroll fixed effects) for the variants whose code is in `fe`.
    Returns (DataFrame, fits {code: ...}, setup)."""
    nf, R = len(D.fiber_name), Wall.shape[0]
    bend_s, cut_b = pooled_tertile(D, 'bend', seg['bend']); kink_s, cut_k = pooled_tertile(D, 'kink', seg['kink'])
    bend_e, _ = pooled_tertile(D, 'bend', first['bend'].to_numpy()); kink_e, _ = pooled_tertile(D, 'kink', first['kink'].to_numpy())
    cs = cell_codes(seg['si'], bend_s, kink_s, seg['fam'], seg['seed_d'], seg['end_d'])
    ce = cell_codes(first['si'].to_numpy(), bend_e, kink_e, (first['family'] == 'V').to_numpy(), first['seed_d'].to_numpy(), first['end_d'].to_numpy())
    nc = int(np.prod(CELL_SHAPE))
    fiber_cell = lambda gf, c, w=None: np.bincount(gf*nc+c, weights=w, minlength=nf*nc).reshape(nf, nc).astype(float)
    Efc = fiber_cell(seg['gf'], cs, seg['len_vx'])*CM
    gf_e = first['gf'].to_numpy()
    Afc = {v: fiber_cell(gf_e[m], ce[m]) for v, _, m in variants}
    Ar, Er = {v: Wall @ Afc[v] for v in Afc}, Wall @ Efc
    lab = cell_labels()
    setup = dict(pooled_cuts=dict(bend=cut_b, kink=cut_k), n_cells=nc, cells_with_exposure=int((Er[0] > 0).sum()),
                 cells_with_events=dict((v, int((Ar[v][0] > 0).sum())) for v in Ar), cell_exposure_cm=float(Er[0].sum()),
                 events_in_cells_without_exposure={v: float(Ar[v][0][Er[0] == 0].sum()) for v in Ar},
                 nan_bend_events=int((bend_e < 0).sum()), nan_bend_exposure_share=float(seg['len_vx'][bend_s < 0].sum()/seg['len_vx'].sum()))
    fits = {}
    for v, _, _ in variants:
        A0, E0 = Ar[v][0], Er[0]
        crude = (A0.reshape(len(SRC), -1)[list(UNSEEN)].sum()/E0.reshape(len(SRC), -1)[list(UNSEEN)].sum())/(A0.reshape(len(SRC), -1)[0].sum()/E0.reshape(len(SRC), -1)[0].sum())
        f0 = fit_cells(A0, E0, lab, ['unseen'], ref0=('unseen',))
        fa = fit_cells(A0, E0, lab, list(GROUPS_2C), ref0=('unseen',))
        ffe = fit_cells(A0, E0, lab, ['scroll', *GROUPS_2C[1:]], ref0=('scroll',))
        sk = None
        try:
            from sklearn.linear_model import PoissonRegressor
            m = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=5000, tol=1e-12, solver='newton-cholesky').fit(fa['X'], fa['y']/np.exp(fa['off']), sample_weight=np.exp(fa['off']))
            sk = float(np.abs(m.coef_-np.array(list(fa['coef'].values()))).max())
        except Exception as e:   # sklearn only cross-checks the IRLS fit
            sk = f'sklearn failed: {e!r}'
        fits[v] = dict(crude_ratio_closed_form=float(crude), crude_ratio_irls_no_covariates=float(np.exp(f0['coef']['unseen=1'])),
                       no_covariate_irls_minus_closed_form=float(np.exp(f0['coef']['unseen=1'])-crude),
                       adjusted_unseen=float(np.exp(fa['coef']['unseen=1'])), adjusted_converged=fa['converged'], adjusted_max_score=fa['score'],
                       adjusted_dropped_levels=fa['dropped_levels'], adjusted_lost_events=fa['lost_events'], fe_converged=ffe['converged'],
                       sklearn_alpha0_max_abs_coef_diff=sk, n_parameters=len(fa['coef']))
    s1 = []
    t4 = time.time()
    for v, _, _ in variants:
        # crude (closed form per replicate; equals the IRLS fit without covariates)
        a2 = lambda M: M.reshape(R, len(SRC), -1).sum(2)
        sa, se = a2(Ar[v]), a2(Er)
        crude = div(div(sa[:, 1:].sum(1), se[:, 1:].sum(1)), div(sa[:, 0], se[:, 0]))
        r = summarize(crude)
        s1.append(dict(variant=v, model='crude (unseen only)', term='unseen vs Paris4', est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'], undefined=r['undefined']))
        boot, cnt = boot_poisson(Ar[v], Er, lab, list(GROUPS_2C), ('unseen',), ['unseen=1'])
        r = summarize(boot['unseen=1'])
        s1.append(dict(variant=v, model='adjusted (bend, kink, family, seed band, end band)', term='unseen vs Paris4', est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'], undefined=r['undefined'], **cnt))
        if v in fe:
            boot, cnt = boot_poisson(Ar[v], Er, lab, ['scroll', *GROUPS_2C[1:]], ('scroll',), ['scroll=1', 'scroll=2', 'scroll=3'])
            for i in (1, 2, 3):
                r = summarize(boot[f'scroll={i}'])
                s1.append(dict(variant=v, model='adjusted, scroll fixed effects', term=f'{SRC[i]} vs Paris4', est=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'], undefined=r['undefined'], **cnt))
        print(f'S1 variant {v} bootstrap done ({time.time()-t4:.0f}s)', flush=True)
    return pd.DataFrame(s1), fits, setup


def main_2c(data, fibers, villa, out):
    t_start = time.time()
    out = Path(out); res_dir = out/'results'
    D = Data(out/'tables', Path(__file__).resolve().parent.parent/'ANALYSIS_PLAN_v3.md')
    load_flags(D, res_dir)
    rng = np.random.default_rng(SEED)
    Wall = np.vstack([np.ones((1, len(D.fiber_name))), boot_weights(D.nf, rng)])
    print(f'tables, flags, weights {Wall.shape} ({time.time()-t_start:.0f}s); plan hash ok {D.checks["plan_sha256_ok"]}', flush=True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
    eps = D.eps
    res = dict(command=' '.join(sys.argv), script_sha256=sha(__file__), B=B, seed=SEED, label='post-hoc (decided after stage 2b)',
               flags_sha256=sha(res_dir/'episode_flags.csv'), strict_exposure_sha256=sha(res_dir/'strict_exposure.npz'),
               flags_2c=json.loads((res_dir/'flags_2c.json').read_text()), checks=D.checks)

    # ---- 1. counts
    fc = flag_counts(eps)
    fc.to_csv(res_dir/'flag_counts.csv', index=False)
    print('\n== 1. past_end counts\n', fc.to_string(index=False), flush=True)
    res['flag_counts'] = fc.to_dict('records')

    # ---- 2. end-censored reruns (past_end hard switches removed from the events; exposure unchanged)
    hard = (eps['cls'] == 'hard_switch').to_numpy()
    hard_c = hard & ~eps['past_end'].to_numpy()
    q1_o, q1_c = q1_counts(D), q1_counts(D, censor_past_end=True)
    orig, cens = analyses_2b(D, Wall, hard, q1_o), analyses_2b(D, Wall, hard_c, q1_c)
    chk = check_against_2b(orig, res_dir)
    print('\n== 2. uncensored rerun vs stage-2b CSVs:', json.dumps(clean(chk)), flush=True)
    res['rerun_vs_2b'] = chk
    sbs = side_by_side(orig.copy(), cens.copy())
    sbs.to_csv(res_dir/'end_censored.csv', index=False)
    res['end_censored_events'] = dict(hard_switches=int(hard.sum()), removed_past_end=int((hard & ~hard_c).sum()), remaining=int(hard_c.sum()),
                                      q1_traces_terminal_hard_past_end=int(D.tr['term_pe'].sum()))
    show = sbs[(sbs['row_type'] != 'rate') & ((sbs['scroll'] == POOLED) | (sbs['row_type'] == 'contrast'))]
    show = show[['analysis', 'contrast', 'scroll', 'est_orig', 'lo_orig', 'hi_orig', 'p_raw_orig', 'p_holm_orig', 'est_cens', 'lo_cens', 'hi_cens', 'p_raw_cens', 'p_holm_cens',
                 'events_exposed_orig', 'events_ref_orig', 'events_exposed_cens', 'events_ref_cens', 'n11_orig', 'n11_cens']]
    print('\n== 2. end-censored (orig | cens)\n', show.round(4).to_string(index=False), flush=True)
    print(f'end-censored done ({time.time()-t_start:.0f}s)', flush=True)

    # ---- 3. strict first departures per cm
    seg = strict_tables(D, res_dir)
    first = eps[eps['is_first'].to_numpy()]
    variants = [('a', 'as scored', np.ones(len(first), bool)), ('b', 'past_end first episodes removed', ~first['past_end'].to_numpy())]
    sh = strict_headline(D, Wall, seg, first, variants)
    sh.to_csv(res_dir/'strict_headline.csv', index=False)
    dg = res['flags_2c']['scrolls']
    cmp = pd.DataFrame([dict(scroll=s, strict_exposure_cm=dg[s]['strict_exposure_cm'], sum_rows_correct_cm=dg[s]['sum_rows_correct_cm'],
                             stage2a_at_risk_cm=D.san[s]['at_risk_cm'], traces_with_first_episode=dg[s]['traces_with_first_episode'],
                             traces_diverged_in_rows=dg[s]['traces_diverged_in_rows'], n_traces=dg[s]['n_traces'],
                             max_trace_excess_over_correct_vx=dg[s]['max_trace_excess_over_correct_vx'], traces_excess_gt_0p05=dg[s]['traces_with_excess_gt_0p05_vx']) for s in SRC])
    print('\n== 3. strict exposure vs sum of rows.correct\n', cmp.round(3).to_string(index=False))
    print('\n== 3. strict first departures per cm\n', sh.round(4).to_string(index=False), flush=True)
    res['strict_exposure_vs_correct'] = cmp.to_dict('records'); res['strict_headline'] = sh.to_dict('records')

    # ---- 4. S1: Poisson with log-exposure offset
    s1, fits, setup = s1_poisson(D, Wall, seg, first, variants, fe=('a',))
    res['s1_setup'], res['s1_fits'] = setup, fits
    print('\n== 4. S1 setup', json.dumps(clean(setup)), flush=True)
    print('\n== 4. data fits', json.dumps(clean(fits)), flush=True)
    s1.to_csv(res_dir/'s1_poisson.csv', index=False)
    print('\n== 4. S1\n', s1.round(4).to_string(index=False), flush=True)
    res['s1'] = s1.to_dict('records')

    # ---- 5. Q1 expectation under within-fiber independence
    q = q1_o
    r_, c_, N_ = q['n11']+q['n10'], q['n11']+q['n01'], q['n11']+q['n10']+q['n01']+q['n00']
    ex_f = np.where(N_ > 0, r_*c_/np.where(N_ > 0, N_, 1.0), 0.0)
    rows5 = []
    for name, sl in [(POOLED, slice(None)), *zip(SRC, D.blocks)]:
        rows5.append(dict(scroll=name, observed_n11=float(q['n11'][sl].sum()), expected_n11=float(ex_f[sl].sum()), n_cells=float(N_[sl].sum()),
                          rows_n11_plus_n10=float(r_[sl].sum()), cols_n11_plus_n01=float(c_[sl].sum()),
                          places_n11_ge_1=int(sum(1 for (f, k) in q['detail'] if (sl == slice(None) or sl.start <= f < sl.stop)))))
    cq = q1_c
    rows5.append(dict(scroll='pooled, end-censored (for reference)', observed_n11=float(cq['n11'].sum()), places_n11_ge_1=len(cq['detail'])))
    q5 = pd.DataFrame(rows5)
    q5.to_csv(res_dir/'q1_expectation.csv', index=False)
    print('\n== 5. Q1 expectation\n', q5.round(4).to_string(index=False))
    res['q1_expectation'] = q5.to_dict('records')
    res['runtime_s'] = time.time()-t_start
    res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (res_dir/'results_2c.json').write_text(json.dumps(clean(res), indent=1))
    print(f'\nruntime {res["runtime_s"]:.0f}s peak RSS {res["peak_rss_mb"]:.0f} MB')


# ================================================================================ stage 2d (post-hoc, see main_2d)
def compare_tables(a, b, cols):
    """Rows of a and b (key column k) compared on cols: rows on both sides / one side only / max abs difference (inf if nan meets a number)."""
    m = a.merge(b, on='k', suffixes=('_a', '_b'), how='outer', indicator=True, validate='one_to_one')
    both = m[m['_merge'] == 'both']
    dmax = 0.0
    for c in cols:
        x, y = both[c+'_a'].to_numpy(float), both[c+'_b'].to_numpy(float)
        d = np.where((x == y) | (np.isnan(x) & np.isnan(y)), 0.0, np.abs(x-y))
        dmax = max(dmax, np.inf if np.isnan(d).any() else (float(d.max()) if len(d) else 0.0))
    return dict(rows_compared=len(both), only_in_rerun=int((m['_merge'] == 'left_only').sum()), only_in_2b=int((m['_merge'] == 'right_only').sum()), max_abs_diff=dmax)


def trunc_sensitivity(D, Wall, hard):
    """Q3 / Q4 sensitivity of main(): GT points flagged as truncated neighbour search are left out (exposure and events)."""
    rows = []
    for cov, flag_s, col, nm in (('cross_d', D.tc, 'trunc_cross', 'Q3 crossing'), ('par_d', D.tp, 'trunc_par', 'Q4 parallel')):
        eg, vg = D.seg_t[cov].copy(), D.ev_t[cov].copy()
        eg[flag_s] = -1
        vg[D.eps[col].to_numpy(bool)] = -1
        a = RateAnalysis(D, Wall, nm, f'excluding {col} points', ['bin0', 'bin1', 'bin2'], eg, vg, hard)
        rows += a.contrasts([('bin0 (nearest) vs bin2', 0, 2)]) + a.rates()
    df = pd.DataFrame(rows)
    df['p_raw'] = np.nan
    df['test'] = df['analysis'].str[:2]
    return df


def check_sens(df, res_dir):
    sens = pd.read_csv(res_dir/'sensitivity.csv', float_precision='round_trip')
    sens = sens[sens['variant'].fillna('').str.startswith('excluding')]
    key = lambda d: d[['row_type', 'analysis', 'variant', 'scroll', 'group', 'contrast']].fillna('').astype(str).agg('|'.join, axis=1)
    cols = ['events', 'exposure_cm', 'rate_per_cm', 'est', 'lo', 'hi', 'undefined', 'events_exposed', 'events_ref', 'exposure_cm_exposed', 'exposure_cm_ref']
    a, b = df.copy(), sens.copy()
    a['k'], b['k'] = key(a), key(b)
    return compare_tables(a[['k']+cols], b[['k']+cols], cols)


def q1_expectation(D, q):
    """Observed n11 and E[n11] = sum_f r_f c_f / N_f under within-fiber independence, with the number of (fiber, bin) places with n11 >= 1."""
    r_, c_, N_ = q['n11']+q['n10'], q['n11']+q['n01'], q['n11']+q['n10']+q['n01']+q['n00']
    ex_f = np.where(N_ > 0, r_*c_/np.where(N_ > 0, N_, 1.0), 0.0)
    rows = []
    for name, sl in [(POOLED, slice(None)), *zip(SRC, D.blocks)]:
        rows.append(dict(scroll=name, observed_n11=float(q['n11'][sl].sum()), expected_n11=float(ex_f[sl].sum()), n_cells=float(N_[sl].sum()),
                         rows_n11_plus_n10=float(r_[sl].sum()), cols_n11_plus_n01=float(c_[sl].sum()),
                         places_n11_ge_1=int(sum(1 for (f, k) in q['detail'] if (sl == slice(None) or sl.start <= f < sl.stop)))))
    return pd.DataFrame(rows)


def recall_rows(m, clscol, version):
    """Recall of the frozen v1 detectors (fb, conf) on the checked episodes whose class (column clscol) is hard_switch, with exact McNemar."""
    from scipy.stats import binomtest
    unseen = [SRC[i] for i in UNSEEN]
    rows = []
    for name in [*SRC, 'unseen pooled']:
        x = m[m['checked'] & (m['source'].isin(unseen) if name == 'unseen pooled' else m['source'] == name) & (m[clscol] == 'hard_switch')]
        n, fb, cf = len(x), int(x['caught_fb'].sum()), int(x['caught_conf'].sum())
        b, c = int((x['caught_fb'] & ~x['caught_conf']).sum()), int((~x['caught_fb'] & x['caught_conf']).sum())
        rows.append(dict(version=version, scroll=name, n=n, caught_fb=fb, caught_conf=cf, recall_fb=fb/n if n else np.nan, recall_conf=cf/n if n else np.nan,
                         fb_only=b, conf_only=c, mcnemar_p=float(binomtest(b, b+c, 0.5).pvalue) if b+c else 1.0))
    return pd.DataFrame(rows)


def detection_check(Do, Dc, events_csv, recall_csv):
    """Stage-2d re-check of the v2 detector recall. detection_events.csv rows are matched to the original and the corrected episodes on
    (source, trace, onset_idx). Returns (recall DataFrame orig + corrected, changed-episodes DataFrame, match diagnostics)."""
    ev = pd.read_csv(events_csv)
    key = ['source', 'trace', 'onset_idx']
    eo = Do.eps[['src', 'trace', 'onset_idx', 'cls']].rename(columns={'src': 'source', 'cls': 'cls_o'})
    m = ev.merge(eo, on=key, how='left', validate='one_to_one')
    assert len(m) == len(ev) == len(Do.eps) and m['cls_o'].notna().all() and (m['cls_o'] == m['cls']).all(), 'detection_events.csv != original episodes'
    ec = Dc.eps[['src', 'trace', 'onset_idx', 'cls', 'orig_cls']].rename(columns={'src': 'source', 'cls': 'cls_c'})
    m = m.merge(ec, on=key, how='left', validate='one_to_one')
    assert ((m['cls_c'].isna()) | (m['orig_cls'] == m['cls'])).all()
    m['cls_c'] = m['cls_c'].fillna('removed')
    assert (m['cls_c'] != 'removed').sum() == len(Dc.eps), 'a corrected episode has no original'
    rec = pd.concat([recall_rows(m, 'cls', 'original'), recall_rows(m, 'cls_c', 'corrected')], ignore_index=True)
    v2 = pd.read_csv(recall_csv, float_precision='round_trip')
    v2 = v2[v2['cls'] == 'hard_switch'].set_index('source')
    o = rec[rec['version'] == 'original'].set_index('scroll')
    diffs = {}
    for name in v2.index:
        diffs[name] = dict(n=int(o.loc[name, 'n'] - v2.loc[name, 'n']), fb=float(abs(o.loc[name, 'recall_fb'] - v2.loc[name, 'fb'])), conf=float(abs(o.loc[name, 'recall_conf'] - v2.loc[name, 'conf'])),
                           fb_only=int(o.loc[name, 'fb_only'] - v2.loc[name, 'fb_only']), conf_only=int(o.loc[name, 'conf_only'] - v2.loc[name, 'conf_only']),
                           mcnemar_p=float(abs(o.loc[name, 'mcnemar_p'] - v2.loc[name, 'mcnemar_p'])))
    unseen = [SRC[i] for i in UNSEEN]
    rows = []
    chg = m[m['checked'] & (m['cls'] != m['cls_c'])]
    for name in [*SRC, 'unseen pooled', 'all scrolls']:
        sub = chg if name == 'all scrolls' else chg[chg['source'].isin(unseen) if name == 'unseen pooled' else chg['source'] == name]
        for (oc, cc), g in sub.groupby(['cls', 'cls_c']):
            rows.append(dict(scroll=name, orig_cls=oc, corrected_cls=cc, n_checked=len(g), caught_fb=int(g['caught_fb'].sum()), caught_conf=int(g['caught_conf'].sum())))
        rows.append(dict(scroll=name, orig_cls='all', corrected_cls='all changed', n_checked=len(sub), caught_fb=int(sub['caught_fb'].sum()), caught_conf=int(sub['caught_conf'].sum())))
        hs = sub[sub['cls'] == 'hard_switch']
        rows.append(dict(scroll=name, orig_cls='hard_switch', corrected_cls='not hard_switch (removed or reclassified)', n_checked=len(hs),
                         caught_fb=int(hs['caught_fb'].sum()), caught_conf=int(hs['caught_conf'].sum())))
    diag = dict(v2_recall_reproduced=diffs, n_events=len(ev), checked=int(m['checked'].sum()),
                unchecked_changed=int((~m['checked'] & (m['cls'] != m['cls_c'])).sum()))
    return rec, pd.DataFrame(rows), diag


def main_2d(data, fibers, villa, out):
    t_start = time.time()
    out = Path(out); res_dir = out/'results'
    plan = Path(__file__).resolve().parent.parent/'ANALYSIS_PLAN_v3.md'
    Do, Dc = Data(out/'tables', plan), Data(out/'tables_2d', plan)
    assert Do.nf == Dc.nf and list(Do.fiber_name) == list(Dc.fiber_name)
    rng = np.random.default_rng(SEED)
    Wall = np.vstack([np.ones((1, len(Do.fiber_name))), boot_weights(Do.nf, rng)])
    print(f'tables, weights {Wall.shape} ({time.time()-t_start:.0f}s); plan hash ok {Do.checks["plan_sha256_ok"]} {Dc.checks["plan_sha256_ok"]}', flush=True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
    for D in (Do, Dc):
        D.eps['order'] = D.eps.groupby(['src', 'trace']).cumcount()
        D.eps['is_first'] = D.eps['order'] == 0
    res = dict(command=' '.join(sys.argv), script_sha256=sha(__file__), B=B, seed=SEED, label='post-hoc (stage 2d, decided after stage 2c)',
               episodes_corrected_sha256=sha(out/'tables_2d'/'episodes.csv'), exposure_corrected_sha256=sha(out/'tables_2d'/'exposure.npz'),
               corrected_2d=json.loads((res_dir/'corrected_2d.json').read_text()), checks_original=Do.checks, checks_corrected=Dc.checks)

    # ---- transitions
    tr = pd.read_csv(res_dir/'episode_transitions_2d.csv')
    cnt = tr.groupby(['src', 'orig_cls', 'cls']).size().rename('n').reset_index()
    rem = tr[tr['cls'] == 'removed'].groupby(['src', 'orig_cls', 'reason']).size().rename('n').reset_index()
    tot = cnt.groupby(['orig_cls', 'cls'])['n'].sum().reset_index().assign(src='all')
    cnt = pd.concat([cnt, tot[cnt.columns]], ignore_index=True)
    cnt.to_csv(res_dir/'transitions_2d.csv', index=False)
    e0 = tr[(tr['orig_cls'] == 'hard_switch') & (tr['end_d'] == 0)]
    end0 = {'n': len(e0), **{str(k): int(v) for k, v in e0['cls'].value_counts().items()}, 'removed_reasons': {str(k): int(v) for k, v in e0[e0['cls'] == 'removed']['reason'].value_counts().items()},
            'by_scroll_removed': {s: int(((e0['src'] == s) & (e0['cls'] == 'removed')).sum()) for s in SRC}}
    res['transitions'] = cnt.to_dict('records'); res['transitions_removed_reasons'] = rem.to_dict('records'); res['end_d_0_hard_switches'] = end0
    wide = cnt.pivot_table(index=['src', 'orig_cls'], columns='cls', values='n', fill_value=0, aggfunc='sum')
    print('\n== transitions (orig class x corrected class)\n', wide.to_string())
    print('\n== removed, by reason\n', rem.to_string(index=False))
    print('\n== end_d = 0 hard switches:', json.dumps(end0), flush=True)

    # ---- 1. primary tests and secondary, original vs corrected
    hard_o, hard_c = (Do.eps['cls'] == 'hard_switch').to_numpy(), (Dc.eps['cls'] == 'hard_switch').to_numpy()
    q1_o, q1_c = q1_counts(Do), q1_counts(Dc)
    orig, corr = analyses_2b(Do, Wall, hard_o, q1_o), analyses_2b(Dc, Wall, hard_c, q1_c)
    so, sc_ = trunc_sensitivity(Do, Wall, hard_o), trunc_sensitivity(Dc, Wall, hard_c)
    res['original_vs_2b'] = dict(**check_against_2b(orig, res_dir), q3_q4_truncation=check_sens(so, res_dir))
    print('\n== original rerun vs stage-2b CSVs:', json.dumps(clean(res['original_vs_2b'])), flush=True)
    sbs = side_by_side(pd.concat([orig, so], ignore_index=True), pd.concat([corr, sc_], ignore_index=True))
    sbs.rename(columns=lambda c: c[:-5]+'_corrected' if c.endswith('_cens') else c).to_csv(res_dir/'corrected.csv', index=False)
    sbs = sbs.rename(columns=lambda c: c[:-5]+'_corr' if c.endswith('_cens') else c)
    res['corrected_events'] = dict(hard_switches_original=int(hard_o.sum()), hard_switches_corrected=int(hard_c.sum()),
                                   exposure_cm_original=float(Do.len_vx.sum()*CM), exposure_cm_corrected=float(Dc.len_vx.sum()*CM),
                                   exposure_cm_by_scroll_original={s: float(Do.len_vx[Do.si == i].sum()*CM) for i, s in enumerate(SRC)},
                                   exposure_cm_by_scroll_corrected={s: float(Dc.len_vx[Dc.si == i].sum()*CM) for i, s in enumerate(SRC)},
                                   hard_by_scroll_original={s: int(((Do.eps.src == s) & hard_o).sum()) for s in SRC},
                                   hard_by_scroll_corrected={s: int(((Dc.eps.src == s) & hard_c).sum()) for s in SRC},
                                   q1_hard_switch_traces=dict(original=Do.checks['q1_hard_switch_traces'] if 'q1_hard_switch_traces' in Do.checks else None,
                                                              corrected_traces_with_terminal_hard=int((Dc.tr['terminal'] == 'hard_switch').sum())),
                                   tertile_exposure_share_fixed_cuts={f'{s}/{k}': [float(Dc.len_vx[(Dc.si == i) & (Dc.seg_t[k] == b)].sum()/Dc.len_vx[(Dc.si == i) & (Dc.seg_t[k] >= 0)].sum()) for b in range(3)]
                                                                      for i, s in enumerate(SRC) for k in TERT_COVS})
    show = sbs[(sbs['row_type'] != 'rate') & ((sbs['scroll'] == POOLED) | (sbs['row_type'] == 'contrast'))]
    show = show[['analysis', 'variant', 'contrast', 'scroll', 'est_orig', 'lo_orig', 'hi_orig', 'p_raw_orig', 'p_holm_orig', 'est_corr', 'lo_corr', 'hi_corr', 'p_raw_corr', 'p_holm_corr',
                 'events_exposed_orig', 'events_ref_orig', 'events_exposed_corr', 'events_ref_corr', 'n11_orig', 'n11_corr']]
    print('\n== 1. original | corrected\n', show.round(4).to_string(index=False), flush=True)
    q5 = pd.concat([q1_expectation(Do, q1_o).assign(version='original'), q1_expectation(Dc, q1_c).assign(version='corrected')], ignore_index=True)
    q5.to_csv(res_dir/'q1_expectation_2d.csv', index=False)
    print('\n== 1. Q1 expectation\n', q5.round(4).to_string(index=False), flush=True)
    res['q1_expectation'] = q5.to_dict('records')
    print(f'primary tests done ({time.time()-t_start:.0f}s)', flush=True)

    # ---- 2. strict first departures and S1 (this should barely change)
    seg_o, seg_c = strict_tables(Do, res_dir), strict_tables(Dc, out/'tables_2d')
    first_o, first_c = Do.eps[Do.eps['is_first'].to_numpy()], Dc.eps[Dc.eps['is_first'].to_numpy()]
    shs, s1s, fits_, setups = [], [], {}, {}
    for lab_, D, seg, first in (('original', Do, seg_o, first_o), ('corrected', Dc, seg_c, first_c)):
        v = [(lab_, lab_, np.ones(len(first), bool))]
        shs.append(strict_headline(D, Wall, seg, first, v))
        s1, fits, setup = s1_poisson(D, Wall, seg, first, v, fe=(lab_,))
        s1s.append(s1); fits_[lab_] = fits[lab_]; setups[lab_] = setup
    sh, s1 = pd.concat(shs, ignore_index=True), pd.concat(s1s, ignore_index=True)
    sh.to_csv(res_dir/'strict_headline_2d.csv', index=False)
    s1.to_csv(res_dir/'s1_poisson_2d.csv', index=False)
    res['strict_headline'], res['s1'], res['s1_fits'], res['s1_setup'] = sh.to_dict('records'), s1.to_dict('records'), fits_, setups
    res['strict_first_episodes'] = dict(original=len(first_o), corrected=len(first_c), segments_original=len(seg_o['gf']), segments_corrected=len(seg_c['gf']),
                                        exposure_cm_original=float(seg_o['len_vx'].sum()*CM), exposure_cm_corrected=float(seg_c['len_vx'].sum()*CM),
                                        first_episode_class_changed=int((first_c['cls'].to_numpy() != first_c['orig_cls'].to_numpy()).sum()))
    print('\n== 2. strict first departures\n', sh.round(4).to_string(index=False))
    print('\n== 2. S1\n', s1.round(4).to_string(index=False), flush=True)
    print('first episodes / strict exposure:', json.dumps(clean(res['strict_first_episodes'])), flush=True)

    # ---- 3. hard switches per cm of at-risk exposure
    hs = []
    for lab_, D, hard in (('original', Do, hard_o), ('corrected', Dc, hard_c)):
        h = D.eps[hard]
        hs.append(strict_headline(D, Wall, dict(gf=D.gf, len_vx=D.len_vx), h, [(lab_, lab_, np.ones(len(h), bool))]))
    hs = pd.concat(hs, ignore_index=True)
    hs.to_csv(res_dir/'hard_per_cm.csv', index=False)
    res['hard_per_cm'] = hs.to_dict('records')
    print('\n== 3. hard switches per cm of at-risk exposure\n', hs.round(4).to_string(index=False), flush=True)

    # ---- 4. detection recall re-check (v2 claim)
    rec, chg, dd = detection_check(Do, Dc, Path(__file__).resolve().parent.parent/'results'/'v2'/'detection_events.csv',
                                   Path(__file__).resolve().parent.parent/'results'/'v2'/'detection_recall.csv') \
        if (Path(__file__).resolve().parent.parent/'results'/'v2'/'detection_events.csv').exists() else (None, None, None)
    if rec is not None:
        rec.to_csv(res_dir/'detection_recheck.csv', index=False)
        chg.to_csv(res_dir/'detection_changed.csv', index=False)
        res['detection_recheck'], res['detection_changed'], res['detection_match'] = rec.to_dict('records'), chg.to_dict('records'), dd
        print('\n== 4. detector recall, hard switches (checked)\n', rec.round(4).to_string(index=False))
        print('\n== 4. checked episodes whose class changed or were removed\n', chg.to_string(index=False))
        print('\n== 4. match diagnostics', json.dumps(clean(dd)), flush=True)
    res['runtime_s'] = time.time()-t_start
    res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (res_dir/'results_2d.json').write_text(json.dumps(clean(res), indent=1))
    print(f'\nruntime {res["runtime_s"]:.0f}s peak RSS {res["peak_rss_mb"]:.0f} MB')


# ================================================================================ stage 2e (post-hoc, see main_2e)
S5_OFFS = np.arange(-64, 65, 4)
S5_COVS = ('bend', 'kink', 'cross_d', 'par_d')
S8_FACTORS = ('scroll', 'bend', 'kink', 'cross', 'par', 'fam', 'seed', 'end', 'junc')
S8_SHAPE = (len(SRC), 4, 4, 3, 3, 2, 3, 2, 2)   # level 0 of every factor is the reference level
S8_LABELS = {'scroll': ['Paris4'] + [f'{s} vs Paris4' for s in SRC[1:]], 'bend': ['bin0', 'bin1 vs bin0', 'bin2 vs bin0', 'NaN vs bin0'],
             'kink': ['bin0', 'bin1 vs bin0', 'bin2 vs bin0', 'NaN vs bin0'], 'cross': ['bin2 (farthest)', 'bin0 (nearest) vs bin2', 'bin1 vs bin2'],
             'par': ['bin2 (farthest)', 'bin0 (nearest) vs bin2', 'bin1 vs bin2'], 'fam': ['H', 'V vs H'], 'seed': ['< 100', '100-400 vs <100', '> 400 vs <100'],
             'end': ['end_d >= 32', 'end_d < 32 vs >= 32'], 'junc': ['junc_d > 16', 'junc_d <= 16 vs > 16']}


def group_specs(D):
    """Groupings behind Q2, Q3, Q4, S2 (hard), S3 and S4: key -> (name, labels, exposure group per segment, event group per episode, contrasts)."""
    eps = D.eps
    jg_s = np.where(D.x['junc_d'] <= 16, 0, 1); jg_e = np.where(eps['junc_d'].to_numpy() <= 16, 0, 1)
    bins3 = ['bin0', 'bin1', 'bin2']
    return {'Q2': ('Q2 bend', bins3, D.seg_t['bend'].copy(), D.ev_t['bend'].copy(), [('bin2 vs bin0', 2, 0)]),
            'Q3': ('Q3 crossing', bins3, D.seg_t['cross_d'].copy(), D.ev_t['cross_d'].copy(), [('bin0 (nearest) vs bin2', 0, 2)]),
            'Q4': ('Q4 parallel', bins3, D.seg_t['par_d'].copy(), D.ev_t['par_d'].copy(), [('bin0 (nearest) vs bin2', 0, 2)]),
            'S2': ('S2 kink, hard switches', bins3, D.seg_t['kink'].copy(), D.ev_t['kink'].copy(), [('bin2 vs bin0', 2, 0)]),
            'S3': ('S3 junction, hard switches', ['junc_d <= 16', 'junc_d > 16'], jg_s, jg_e, [('<= 16 vs > 16', 0, 1)]),
            'S4': ('S4 seed distance, hard switches', ['< 100', '100-400', '> 400'], band_seed(D.seed_d), band_seed(eps['seed_d'].to_numpy()),
                   [('100-400 vs <100', 1, 0), ('>400 vs <100', 2, 0)])}


def fiber_stratified(D, Wall, spec, hard):
    """MH rate ratios stratified by FIBER (every fiber a stratum; the bootstrap weights multiply each fiber's counts and exposure), next to the
    scroll-stratified values of the same contrasts. Returns rows (dicts)."""
    name, labels, eg, vg, cons = spec
    nfib, G = len(D.fiber_name), len(labels)
    A = fiber_table(D.eps['gf'].to_numpy()[hard], vg[hard], nfib, G)
    E = fiber_table(D.gf, eg, nfib, G, D.len_vx)
    SA, SE = Wall[:, :, None]*A[None], Wall[:, :, None]*E[None]
    ra = RateAnalysis(D, Wall, name, '', labels, eg, vg, hard)
    rows = []
    for label, ex, ref in cons:
        pooled, _ = mh_rr(SA, SE, ex, ref)
        r = summarize(pooled)
        T = E[:, ex]+E[:, ref]
        t = np.where(T > 0, T, 1.0)
        num_f, den_f = np.where(T > 0, A[:, ex]*E[:, ref]/t, 0.0), np.where(T > 0, A[:, ref]*E[:, ex]/t, 0.0)
        sc = next(x for x in ra.contrasts([(label, ex, ref)]) if x['scroll'] == POOLED)
        rows.append(dict(analysis=name, contrast=label, est_scroll=sc['est'], lo_scroll=sc['lo'], hi_scroll=sc['hi'], p_scroll=sc['p_raw'],
                         est_fiber=r['est'], lo_fiber=r['lo'], hi_fiber=r['hi'], p_fiber=r['p'], undefined_scroll=sc['undefined'], undefined_fiber=r['undefined'],
                         events_exposed=float(A[:, ex].sum()), events_ref=float(A[:, ref].sum()), exposure_cm_exposed=float(E[:, ex].sum()*CM),
                         exposure_cm_ref=float(E[:, ref].sum()*CM), fibers_contributing=int(((num_f > 0) | (den_f > 0)).sum()), n_fibers=nfib))
    return rows


def pooled_rates(D, Wall, spec, hard):
    """Crude pooled hard switches per cm per bin (all scrolls together, no stratification) with fiber-bootstrap percentile CIs."""
    name, labels, eg, vg, _ = spec
    nfib, G = len(D.fiber_name), len(labels)
    A = fiber_table(D.eps['gf'].to_numpy()[hard], vg[hard], nfib, G)
    E = fiber_table(D.gf, eg, nfib, G, D.len_vx)
    SA, SE = Wall @ A, (Wall @ E)*CM
    rate = div(SA, SE)
    rows = []
    for g, lab in enumerate(labels):
        r = summarize(rate[:, g])
        rows.append(dict(panel=name, bin=lab, events=float(SA[0, g]), exposure_cm=float(SE[0, g]), rate_per_cm=r['est'], lo=r['lo'], hi=r['hi']))
    return rows


def s8_codes(D, hard):
    """Cell codes (S8_SHAPE) of every at-risk segment and every corrected hard switch; level 0 of every factor is its reference level."""
    eps = D.eps
    def lv(seg_t, ev_t):
        return np.where(seg_t < 0, 3, seg_t), np.where(ev_t < 0, 3, ev_t)
    refmap = np.array([1, 2, 0])   # tertile bin -> code with bin2 as reference
    parts = {}
    for k, nm in (('bend', 'bend'), ('kink', 'kink')):
        parts[nm] = lv(D.seg_t[k], D.ev_t[k][hard])
    for k, nm in (('cross_d', 'cross'), ('par_d', 'par')):
        assert (D.seg_t[k] >= 0).all() and (D.ev_t[k][hard] >= 0).all()
        parts[nm] = (refmap[D.seg_t[k]], refmap[D.ev_t[k][hard]])
    parts['fam'] = (D.fam.astype(int), (eps['family'].to_numpy()[hard] == 'V').astype(int))
    parts['seed'] = (band_seed(D.seed_d), band_seed(eps['seed_d'].to_numpy()[hard]))
    parts['end'] = ((D.x['end_d'] < 32).astype(int), (eps['end_d'].to_numpy()[hard] < 32).astype(int))
    parts['junc'] = ((D.x['junc_d'] <= 16).astype(int), (eps['junc_d'].to_numpy()[hard] <= 16).astype(int))
    seg = np.ravel_multi_index((D.si, *[parts[f][0] for f in S8_FACTORS[1:]]), S8_SHAPE)
    ev = np.ravel_multi_index((eps['si'].to_numpy()[hard], *[parts[f][1] for f in S8_FACTORS[1:]]), S8_SHAPE)
    return seg, ev


def s8_rr(fit, A, E, lab, terms):
    """exp(coef) of every term. A level with exposure but no event has fitted rate 0 (RR 0, the boundary MLE); a level without exposure, or a factor whose
    reference level is gone, is not estimable (nan); a fit that did not converge (separation) gives nan everywhere. Returns (dict, flags dict)."""
    out, flags = {}, {}
    for fac, l in terms:
        key = f'{fac}={l}'
        if not fit['converged']:
            out[key], flags[key] = np.nan, 'not converged'
        elif not (lab[fac][fit['keep']] == 0).any():
            out[key], flags[key] = np.nan, 'reference level gone'
        elif key in fit['coef']:
            out[key], flags[key] = float(np.exp(fit['coef'][key])), ''
        elif E[lab[fac] == l].sum() > 0:
            out[key], flags[key] = 0.0, 'zero-event level (RR 0)'
        else:
            out[key], flags[key] = np.nan, 'no exposure'
    return out, flags


def s8_poisson(D, Wall, hard):
    """S8 mutual adjustment: one Poisson model (log-exposure offset, exact MLE by IRLS) for the hard switches `hard`, main effects only, reference levels
    as in S8_LABELS. Returns (DataFrame of terms, diagnostics)."""
    nfib, R = len(D.fiber_name), Wall.shape[0]
    seg, ev = s8_codes(D, hard)
    codes = np.unique(np.r_[seg, ev])
    ncell = len(codes)
    si_seg, si_ev = np.searchsorted(codes, seg), np.searchsorted(codes, ev)
    gf_ev = D.eps['gf'].to_numpy()[hard]
    fc = lambda gf, c, w=None: np.bincount(gf*ncell+c, weights=w, minlength=nfib*ncell).reshape(nfib, ncell).astype(float)
    Afc, Efc = fc(gf_ev, si_ev), fc(D.gf, si_seg, D.len_vx)*CM
    Ar, Er = Wall @ Afc, Wall @ Efc
    lv = np.unravel_index(codes, S8_SHAPE)
    lab = dict(zip(S8_FACTORS, lv))
    terms = [(f, l) for f in S8_FACTORS for l in range(1, S8_SHAPE[S8_FACTORS.index(f)]) if Er[0][lab[f] == l].sum() > 0]
    est = np.full((R, len(terms)), np.nan)
    flag_counts = {f'{f}={l}': dict() for f, l in terms}
    nonconv = 0
    t0 = time.time()
    for r in range(R):
        fit = fit_cells(Ar[r], Er[r], lab, list(S8_FACTORS), ref0=S8_FACTORS)
        out_, fl = s8_rr(fit, Ar[r], Er[r], lab, terms)
        nonconv += not fit['converged']
        for j, (f, l) in enumerate(terms):
            est[r, j] = out_[f'{f}={l}']
            if r > 0 and fl[f'{f}={l}']:
                flag_counts[f'{f}={l}'][fl[f'{f}={l}']] = flag_counts[f'{f}={l}'].get(fl[f'{f}={l}'], 0)+1
        if r == 0:
            fit0 = fit
            try:
                from sklearn.linear_model import PoissonRegressor
                m = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=5000, tol=1e-12, solver='newton-cholesky').fit(fit['X'], fit['y']/np.exp(fit['off']), sample_weight=np.exp(fit['off']))
                sk = float(np.abs(m.coef_-np.array(list(fit['coef'].values()))).max())
            except Exception as e:
                sk = f'sklearn failed: {e!r}'
            print(f'S8 data fit: {int(fit["keep"].sum())} cells with exposure of {ncell} occupied, {len(fit["coef"])} parameters, converged {fit["converged"]}, '
                  f'dropped {fit["dropped_levels"]}, lost events {fit["lost_events"]}, sklearn diff {sk}', flush=True)
        elif r % 1000 == 0:
            print(f'S8 bootstrap {r}/{R-1} ({time.time()-t0:.0f}s)', flush=True)
    rows = []
    for j, (f, l) in enumerate(terms):
        sm = summarize(est[:, j])
        m = lab[f] == l
        rows.append(dict(factor=f, level=S8_LABELS[f][l], rr=sm['est'], lo=sm['lo'], hi=sm['hi'], p=sm['p'], undefined=sm['undefined'],
                         replicates_zero_event_level=flag_counts[f'{f}={l}'].get('zero-event level (RR 0)', 0),
                         replicates_reference_gone=flag_counts[f'{f}={l}'].get('reference level gone', 0),
                         replicates_not_converged=flag_counts[f'{f}={l}'].get('not converged', 0),
                         replicates_no_exposure=flag_counts[f'{f}={l}'].get('no exposure', 0),
                         events=float(Ar[0][m].sum()), exposure_cm=float(Er[0][m].sum()), events_reference=float(Ar[0][lab[f] == 0].sum()),
                         exposure_cm_reference=float(Er[0][lab[f] == 0].sum())))
    diag = dict(occupied_cells=int(ncell), cells_with_exposure=int((Er[0] > 0).sum()), parameters=len(fit0['coef']), data_fit_converged=bool(fit0['converged']),
                data_fit_dropped_levels=fit0['dropped_levels'], data_fit_lost_events=float(fit0['lost_events']), data_fit_max_score=float(fit0['score']),
                replicates=int(R-1), replicates_not_converged=int(nonconv), reference_levels={f: S8_LABELS[f][0] for f in S8_FACTORS}, events=float(Ar[0].sum()),
                exposure_cm=float(Er[0].sum()))
    return pd.DataFrame(rows), diag


def s6_summary(df):
    """S6 shares per scroll and pooled from s6_switches.csv."""
    rows = []
    df = df.copy()
    tobool = lambda x: x.map({True: True, False: False, 'True': True, 'False': False})
    for c in ('landed', 'is_first'):
        df[c] = tobool(df[c]).astype(bool)
    for c in ('same_family', 'landing_crossing', 'departure_match'):
        df[c] = tobool(df[c])
    for name, g in [*[(s, df[df['src'] == s]) for s in SRC], ('pooled', df)]:
        n = len(g)
        land = g[g['landed']]
        nl = len(land)
        tb = g['turn_bin'].dropna()
        fs = g[g['is_first']]
        dm = fs['departure_match'].dropna().astype(bool)
        dirs = fs['direction'].dropna()
        gap = land['gap_vx'].dropna()
        rows.append(dict(scroll=name, switches=n, landed=nl, landed_share=nl/n if n else np.nan,
                         same_family=int(land['same_family'].astype(bool).sum()) if nl else 0, other_family=int((~land['same_family'].astype(bool)).sum()) if nl else 0,
                         crossing=int(land['landing_crossing'].astype(bool).sum()) if nl else 0, parallel=int((~land['landing_crossing'].astype(bool)).sum()) if nl else 0,
                         landing_len_median=float(land['landing_len'].median()) if nl else np.nan,
                         gap_median=float(gap.median()) if len(gap) else np.nan, gap_q1=float(gap.quantile(0.25)) if len(gap) else np.nan, gap_q3=float(gap.quantile(0.75)) if len(gap) else np.nan,
                         rank1=int((land['gap_rank'] == 1).sum()) if nl else 0, rank1_share=float((land['gap_rank'] == 1).mean()) if nl else np.nan,
                         turn_valid=len(tb), turn_0_15=int((tb == '0-15').sum()), turn_15_45=int((tb == '15-45').sum()), turn_45_90=int((tb == '45-90').sum()),
                         turn_0_15_share=float((tb == '0-15').mean()) if len(tb) else np.nan, turn_15_45_share=float((tb == '15-45').mean()) if len(tb) else np.nan,
                         turn_45_90_share=float((tb == '45-90').mean()) if len(tb) else np.nan,
                         first_episode_switches=len(fs), departure_rows_found=len(dirs), departure_match=int(dm.sum()), departure_match_rate=float(dm.mean()) if len(dm) else np.nan,
                         normal=int((dirs == 'normal').sum()), in_sheet=int((dirs == 'in_sheet').sum()),
                         normal_share=float((dirs == 'normal').mean()) if len(dirs) else np.nan, in_sheet_share=float((dirs == 'in_sheet').mean()) if len(dirs) else np.nan))
    return pd.DataFrame(rows)


def s5_values(D, hs):
    """Covariates at S5_OFFS vx from the departure GT point along the GT in the direction of travel (nan beyond the annotation)."""
    out = {k: np.full((len(hs), len(S5_OFFS)), np.nan) for k in S5_COVS}
    for i, e in enumerate(hs.itertuples()):
        c = D.cov[SRC[e.si]]
        lo, hi = c['offsets'][e.fiber_idx], c['offsets'][e.fiber_idx+1]
        sf = c['s'][lo:hi]
        target = sf[int(e.gt_pt)]+e.sign*S5_OFFS
        ok = (target >= sf[0]) & (target <= sf[-1])
        idx = WE.arc_to_idx(sf, target)
        for k in S5_COVS:
            out[k][i, ok] = c[k][lo+idx[ok]]
    return out


def s5_profiles(D, hs, Wall, B1=1000):
    """Event-weighted pooled mean of each covariate at each offset, baseline = sum_s (share of events in scroll s) x (exposure-weighted mean in scroll s),
    and 95% bands from the first B1 fiber-bootstrap replicates. Returns (DataFrame, checks)."""
    vals = s5_values(D, hs)
    W = Wall[:B1+1]
    nfib = len(D.fiber_name)
    we = W[:, hs['gf'].to_numpy()]
    si = hs['si'].to_numpy()
    n_s = np.stack([we[:, si == k].sum(1) for k in range(len(SRC))], 1)
    share = n_s/n_s.sum(1, keepdims=True)
    rows, chk = [], {}
    for k in S5_COVS:
        v = vals[k]
        ok = np.isfinite(v)
        prof = (we @ np.where(ok, v, 0.0))/(we @ ok.astype(float))
        x = D.x[k]
        okx = np.isfinite(x)
        Sx = np.bincount(D.gf[okx], weights=(D.len_vx*x)[okx], minlength=nfib)
        Sl = np.bincount(D.gf[okx], weights=D.len_vx[okx], minlength=nfib)
        mean_s = np.stack([(W[:, b] @ Sx[b])/(W[:, b] @ Sl[b]) for b in D.blocks], 1)
        base = (share*mean_s).sum(1)
        lo, hi = np.nanpercentile(prof[1:], [2.5, 97.5], axis=0)
        blo, bhi = np.percentile(base[1:], [2.5, 97.5])
        for j, off in enumerate(S5_OFFS):
            rows.append(dict(covariate=k, offset_vx=int(off), n_valid=int(ok[:, j].sum()), mean=float(prof[0, j]), lo=float(lo[j]), hi=float(hi[j]),
                             baseline=float(base[0]), baseline_lo=float(blo), baseline_hi=float(bhi)))
        j0 = int(np.flatnonzero(S5_OFFS == 0)[0])
        e0 = hs[k].to_numpy(float).astype(np.float32).astype(float)
        chk[k] = int((~np.isclose(v[:, j0], e0, rtol=0, atol=0, equal_nan=True)).sum())
    return pd.DataFrame(rows), dict(offset0_values_not_equal_to_event_covariate=chk, replicates=B1)


def main_2e(data, fibers, villa, out):
    t_start = time.time()
    out = Path(out); res_dir = out/'results'
    plan = Path(__file__).resolve().parent.parent/'ANALYSIS_PLAN_v3.md'
    D = Data(out/'tables_2d', plan)
    rng = np.random.default_rng(SEED)
    Wall = np.vstack([np.ones((1, len(D.fiber_name))), boot_weights(D.nf, rng)])
    print(f'corrected tables, weights {Wall.shape} ({time.time()-t_start:.0f}s); plan hash ok {D.checks["plan_sha256_ok"]}', flush=True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 500)
    D.eps['order'] = D.eps.groupby(['src', 'trace']).cumcount()
    hard = (D.eps['cls'] == 'hard_switch').to_numpy()
    res = dict(command=' '.join(sys.argv), script_sha256=sha(__file__), B=B, seed=SEED, label='post-hoc, except S5, S6 and S8 (pre-registered, now on corrected episodes)',
               episodes_corrected_sha256=sha(out/'tables_2d'/'episodes.csv'), n_hard_switches=int(hard.sum()))
    specs = group_specs(D)

    # ---- A. stratified by fiber instead of scroll (post-hoc)
    fs = pd.DataFrame([r for k in ('Q2', 'Q3', 'Q4', 'S2', 'S3', 'S4') for r in fiber_stratified(D, Wall, specs[k], hard)])
    prev = pd.read_csv(res_dir/'corrected.csv', float_precision='round_trip')
    prev = prev[(prev['row_type'] == 'contrast') & (prev['scroll'] == POOLED) & prev['variant'].isna()].set_index(['analysis', 'contrast'])
    def _d(x, y):
        x, y = float(x), float(y)
        return 0.0 if (x == y or (np.isnan(x) and np.isnan(y))) else abs(x-y)
    diff = max(_d(prev.loc[(r.analysis, r.contrast), c], getattr(r, a)) for r in fs.itertuples()
               for c, a in (('est_corrected', 'est_scroll'), ('lo_corrected', 'lo_scroll'), ('hi_corrected', 'hi_scroll')))
    fs.to_csv(res_dir/'fiber_stratified.csv', index=False)
    res['fiber_stratified'] = fs.to_dict('records'); res['scroll_values_vs_corrected_csv_max_abs_diff'] = diff
    print('\n== A. MH by scroll vs by fiber (corrected hard switches); scroll values equal corrected.csv to', diff, '\n', fs.round(4).to_string(index=False), flush=True)
    print(f'A done ({time.time()-t_start:.0f}s)', flush=True)

    # ---- B. S8 mutual adjustment (pre-registered, corrected episodes)
    s8, d8 = s8_poisson(D, Wall, hard)
    s8.to_csv(res_dir/'s8_poisson.csv', index=False)
    res['s8'], res['s8_diag'] = s8.to_dict('records'), d8
    print('\n== B. S8\n', s8.round(4).to_string(index=False), '\n', json.dumps(clean(d8)), flush=True)
    print(f'B done ({time.time()-t_start:.0f}s)', flush=True)

    # ---- C. S6 summary (pre-registered, corrected episodes)
    s6 = pd.read_csv(res_dir/'s6_switches.csv')
    assert len(s6) == int(hard.sum())
    s6s = s6_summary(s6)
    s6s.to_csv(res_dir/'s6_summary.csv', index=False)
    res['s6_summary'] = s6s.to_dict('records')
    print('\n== C. S6 summary\n', s6s.round(3).T.to_string(), flush=True)

    # ---- D. S5 profiles (pre-registered, corrected episodes)
    s5, c5 = s5_profiles(D, D.eps[hard], Wall)
    s5.to_csv(res_dir/'s5_profiles.csv', index=False)
    res['s5_checks'] = c5
    print('\n== D. S5 checks', json.dumps(clean(c5)), '\n', s5[s5.offset_vx.isin([-64, -32, -16, 0, 16, 32, 64])].round(3).to_string(index=False), flush=True)

    # ---- E. data for figure 2 (post-hoc): pooled rates by bin
    rc = pd.DataFrame([r for k in ('Q2', 'S2', 'Q3', 'Q4', 'S4', 'S3') for r in pooled_rates(D, Wall, specs[k], hard)])
    rc.to_csv(res_dir/'rates_by_condition.csv', index=False)
    res['rates_by_condition'] = rc.to_dict('records')
    print('\n== E. pooled hard switches per cm by bin\n', rc.round(4).to_string(index=False), flush=True)
    res['runtime_s'] = time.time()-t_start
    res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (res_dir/'results_2e.json').write_text(json.dumps(clean(res), indent=1))
    print(f'\nruntime {res["runtime_s"]:.0f}s peak RSS {res["peak_rss_mb"]:.0f} MB')


if __name__ == '__main__':
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 4 or flags - {'--2c', '--2d', '--2e'} or len(flags) > 1:
        sys.exit(__doc__)
    (main_2c if '--2c' in flags else main_2d if '--2d' in flags else main_2e if '--2e' in flags else main)(*args)
