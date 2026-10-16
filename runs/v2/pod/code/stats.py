"""Final statistics for the held-out evaluation of scrollprize/fiber_single_small (runs local, no GPU, no villa import).

Inputs : runs/{c050,c070}/<source>/{rows.csv,paths.jsonl,config.json}, runs/c050/<source>/fwdback_*.csv,
         results/baseline/decisions.csv.gz (per-decision signals from analyze.py).
Outputs: results/final/*.csv, *.json and figures/*.png.

Definitions (trace-grid voxels, 1 vx = 19.2 um; tol = 3 vx and patience = 3 vx exactly as villa's scorer):
  strict      villa score_trace: everything after the first sustained departure is wrong, even if the path returns.
  pointwise   a segment is wrong only while its end vertex is > tol from the matched GT (same scored region).
  episode     a sustained excursion (> tol over >= 3 vx of path). It has RETURNED when the path is back within tol for
              >= RETURN_RUN vx; otherwise it is PERSISTENT. Persistent episodes are split by peak distance:
              drift (<= HARD vx) and hard switch (> HARD vx); 'ended' = trace stopped < RETURN_RUN vx after leaving.
  exposure    strict hazards use on-fiber length before the first departure (rows.correct) and the decisions made up
              to and including the departing one; episode hazards use all pointwise on-fiber length.
CIs: percentile bootstrap over traces (one directed seed per fiber, so traces = fibers), stratified by source.
Rate comparisons: exact conditional binomial test on (events, exposure) pairs, Clopper-Pearson CI on the ratio.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'results'/'final'
FIG = OUT/'figures'
MM = 0.0192
TOL, PATIENCE, RETURN_RUN, HARD = 3.0, 3.0, 16.0, 15.0
SOURCES = ('paris4_heldout', 'PHerc0139', 'PHerc1451', 'PHerc0332')
LABEL = {'paris4_heldout': 'Paris4 (control)', 'PHerc0139': 'PHerc0139', 'PHerc1451': 'PHerc1451', 'PHerc0332': 'PHerc0332'}
B = 4000
RNG = np.random.default_rng(7349)


# ----------------------------------------------------------------------------------------------- per-trace geometry
def sustained_onsets(lengths, bad, patience=PATIENCE):
    """Copy of villa evaluation.scoring.sustained_onsets (same semantics)."""
    onsets, start, counted = [], None, False
    for i, v in enumerate(bad):
        if not v:
            start = None
            continue
        if start is None:
            start, counted = i, False
        if not counted and lengths[i]-lengths[start] >= patience:
            onsets.append(start)
            counted = True
    return onsets


def episodes(L, d, return_run=RETURN_RUN, hard=HARD):
    """Excursion episodes in a scored region (arrays L cumulative length, d GT distance)."""
    out, i0 = [], 0
    bad = d > TOL
    while True:
        on = sustained_onsets(L[i0:]-L[i0], bad[i0:])
        if not on:
            return out
        o = i0 + on[0]
        j, ret = o, None
        while j < len(d):  # first index from which the path stays within tol for RETURN_RUN vx
            if not bad[j]:
                k = j
                while k < len(d) and not bad[k]:
                    k += 1
                if L[k-1]-L[j] >= return_run:
                    ret = j
                    break
                j = k
            else:
                j += 1
        stop = ret if ret is not None else len(d)
        peak = float(d[o:stop].max())
        after = float(L[-1]-L[o])
        if ret is not None:
            cls = 'returned'
        elif after < return_run:
            cls = 'ended'
        else:
            cls = 'hard_switch' if peak > hard else 'drift'
        out.append(dict(onset_idx=o, onset_len=float(L[o]), peak=peak, returned=ret is not None, cls=cls,
                        off_len=float((L[stop-1] if stop > o else L[o])-L[o]), len_after=after))
        if ret is None:
            return out
        i0 = ret


def load(conf_dir):
    rows_all, eps_all, pts = [], [], {}
    for src in SOURCES:
        run = ROOT/'runs'/conf_dir/src
        cfg = json.loads((run/'config.json').read_text())
        rows = pd.read_csv(run/'rows.csv')
        paths = [json.loads(line) for line in open(run/'paths.jsonl')]
        assert len(rows) == len(paths)
        for t, (r, p) in enumerate(zip(rows.itertuples(), paths)):
            P = np.asarray(p['path'], float).reshape(-1, 3)
            d = np.asarray(p['gt_distance'], float)
            L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] if len(P) else np.zeros(0)
            # Check the stored profile reproduces villa's strict departure and length partition.
            on = sustained_onsets(L, d > TOL)
            end = on[0] if on else len(P)
            if not r.crossed_endpoint:
                stored = p['departure_index'] if p['departure_index'] is not None else len(P)
                # gt_distance is stored rounded to 1e-3: allow an off-by-one only at an exact tolerance tie.
                assert stored == end or (abs(stored-end) == 1 and abs(d[min(stored, end)]-TOL) < 1e-3), (src, t)
                end = stored
                c_expect = float(L[max(0, end-1)]) if end < len(P) else (float(L[-1]) if len(L) else 0.0)
                assert abs(c_expect - r.correct) < 0.05 and abs((L[-1] if len(L) else 0) - c_expect - r.offtrack) < 0.05, (src, t)
            # Scored region: the whole path, or up to the endpoint crossing (after it is unknown / overrun).
            scored = L <= r.correct + 1e-9 if r.crossed_endpoint else np.ones(len(L), bool)
            Ls, ds = L[scored], d[scored]
            seg = np.diff(Ls)
            off = ds[1:] > TOL
            on_len, off_len = float(seg[~off].sum()), float(seg[off].sum())
            if r.crossed_endpoint and r.endpoint_known:
                off_len += float(r.endpoint_overrun)
            ep = episodes(Ls, ds) if len(Ls) > 1 else []
            for e in ep:
                eps_all.append(dict(source=src, trace=t, fiber_name=r.fiber_name, annotator=r.annotator, **e))
            first = ep[0]['cls'] if ep else None
            rows_all.append(dict(source=src, trace=t, fiber_name=r.fiber_name, family=r.family, annotator=r.annotator,
                                 split=r.split, reviewed=r.reviewed, length=float(r.length), correct=float(r.correct),
                                 wrong=float(r.offtrack), followed=float(r.followed), avail=float(r.avail),
                                 diverged=bool(r.diverged), reason=r.reason, pw_on=on_len, pw_off=off_len,
                                 first_cls=first if r.diverged else None,
                                 n_returned=sum(e['cls'] == 'returned' for e in ep),
                                 n_hard=sum(e['cls'] == 'hard_switch' for e in ep),
                                 n_drift=sum(e['cls'] == 'drift' for e in ep),
                                 n_ended=sum(e['cls'] == 'ended' for e in ep),
                                 n_persistent=sum(e['cls'] in ('hard_switch', 'drift') for e in ep)))
            pts[(src, t)] = (Ls, ds)
    return pd.DataFrame(rows_all), pd.DataFrame(eps_all), pts


# ---------------------------------------------------------------------------------------------------- statistics
def boot_idx(n):
    return RNG.integers(0, n, size=(B, n))


def ci(samples):
    return np.nanpercentile(samples, [2.5, 97.5])


def ratio_ci(num, den, idx):
    s = num[idx].sum(1)/np.maximum(den[idx].sum(1), 1e-12)
    return ci(s)


def rate_ratio(k1, e1, k2, e2):
    """Exact conditional test of rate1 == rate2 (events k over exposure e). Returns RR, CI, p."""
    n = k1 + k2
    if n == 0:
        return np.nan, (np.nan, np.nan), 1.0
    p0 = e1/(e1+e2)
    p = binomtest(int(k1), int(n), p0).pvalue
    lo = beta.ppf(0.025, k1, n-k1+1) if k1 > 0 else 0.0
    hi = beta.ppf(0.975, k1+1, n-k1) if k1 < n else 1.0
    conv = lambda q: (q/(1-q))*(e2/e1) if q < 1 else np.inf
    return (k1/e1)/(k2/e2) if k2 > 0 else np.inf, (conv(lo), conv(hi)), p


def stratified_rate_ratio(strata):
    """Mantel-Haenszel rate ratio over strata [(k1, e1, k0, e0)] with an exact conditional (Poisson-binomial) p-value."""
    num = sum(k1*e0/(e1+e0) for k1, e1, k0, e0 in strata)
    den = sum(k0*e1/(e1+e0) for k1, e1, k0, e0 in strata)
    dist = np.array([1.0])
    for k1, e1, k0, e0 in strata:
        n, q = int(k1+k0), e1/(e1+e0)
        dist = np.convolve(dist, binom_pmf(n, q))
    obs = int(sum(k1 for k1, *_ in strata))
    p_two = min(1.0, 2*min(dist[:obs+1].sum(), dist[obs:].sum()))
    return num/den if den > 0 else np.inf, p_two


def binom_pmf(n, q):
    from scipy.stats import binom
    return binom.pmf(np.arange(n+1), n, q)


def decision_exposure(dec, rows):
    """Decisions at risk per trace: up to and including the first departing decision."""
    d = dec[dec['label'].isin(['clean_trace', 'before', 'at_departure'])]
    n = d.groupby('trace').size()
    return np.array([n.get(t, 0) for t in rows['trace']], float)


def summary_table(rows, dec, conf):
    out = []
    for src in SOURCES:
        g = rows[rows.source == src].reset_index(drop=True)
        dd = dec[(dec.source == src) & (dec.confidence == conf)]
        nd = decision_exposure(dd, g)
        idx = boot_idx(len(g))
        c, w, on, off = (g[k].to_numpy(float) for k in ('correct', 'wrong', 'pw_on', 'pw_off'))
        dep = g['diverged'].to_numpy(float)
        hard, ret, pers = (g[k].to_numpy(float) for k in ('n_hard', 'n_returned', 'n_persistent'))
        fol, av = g['followed'].to_numpy(float), g['avail'].to_numpy(float)
        first = g['first_cls']
        row = dict(source=src, confidence=conf, traces=len(g), departures=int(dep.sum()),
                   first_returned=int((first == 'returned').sum()), first_drift=int((first == 'drift').sum()),
                   first_hard=int((first == 'hard_switch').sum()), first_ended=int((first == 'ended').sum()),
                   traces_with_hard=int((hard > 0).sum()), traced_mm=g['length'].sum()*MM)
        metrics = {
            'strict_precision': (c, c+w), 'pointwise_precision': (on, on+off), 'coverage': (fol, av),
            'departures_per_100_decisions': (100*dep, nd), 'departures_per_cm': (dep, c*MM/10),
            'hard_switches_per_cm': (hard, on*MM/10), 'returned_excursions_per_cm': (ret, on*MM/10),
            'persistent_per_cm': (pers, on*MM/10), 'hard_switch_trace_rate': (hard > 0, np.ones(len(g))),
        }
        for name, (num, den) in metrics.items():
            num, den = np.asarray(num, float), np.asarray(den, float)
            row[name] = num.sum()/max(den.sum(), 1e-12)
            row[name+'_lo'], row[name+'_hi'] = ratio_ci(num, den, idx)
        row.update(hard_events=int(hard.sum()), correct_mm=c.sum()*MM, strict_wrong_mm=w.sum()*MM, pointwise_wrong_mm=off.sum()*MM,
                   decisions_at_risk=int(nd.sum()), onfiber_cm=on.sum()*MM/10, exposure_cm=c.sum()*MM/10)
        out.append(row)
    return pd.DataFrame(out)


def rate_tests(summ):
    out = []
    ref = summ[summ.source == 'paris4_heldout'].iloc[0]
    for _, r in summ.iterrows():
        if r.source == 'paris4_heldout':
            continue
        for ev, ex, lab in (('departures', 'exposure_cm', 'strict departures per cm'),
                            ('departures', 'decisions_at_risk', 'strict departures per decision'),
                            ('hard_events', 'onfiber_cm', 'hard switches per cm')):
            k1, k2 = r[ev], ref[ev]
            rr, (lo, hi), p = rate_ratio(k1, r[ex], k2, ref[ex])
            out.append(dict(confidence=r.confidence, source=r.source, vs='paris4_heldout', metric=lab,
                            events=int(round(k1)), exposure=r[ex], ref_events=int(round(k2)), ref_exposure=ref[ex],
                            rate_ratio=rr, lo=lo, hi=hi, p_exact=p))
    return pd.DataFrame(out)


def sensitivity(rows, pts):
    """Class counts of first departures and all excursions under other return-run / hard-switch cut-offs."""
    out = []
    for rr in (8.0, 16.0, 32.0, 64.0):
        for hd in (10.0, 15.0, 30.0):
            for src in SOURCES:
                g = rows[rows.source == src]
                first, every = [], []
                for r in g.itertuples():
                    L, d = pts[(src, r.trace)]
                    ep = episodes(L, d, rr, hd) if len(L) > 1 else []
                    every += [e['cls'] for e in ep]
                    if r.diverged and ep:
                        first.append(ep[0]['cls'])
                row = dict(return_run=rr, hard=hd, source=src, departures=int(g.diverged.sum()))
                for c in ('returned', 'drift', 'hard_switch', 'ended'):
                    row['first_'+c] = first.count(c)
                    row['all_'+c] = every.count(c)
                out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------- stop rules (offline, c050 traces)
CONF_GRID = np.round(np.arange(0.50, 0.9951, 0.0025), 4)


def kept(cut, L, d, c, w, crossed):
    """(strict correct, strict wrong, pointwise on, pointwise off) kept when a trace is cut at path length `cut`."""
    sc = min(cut, c)
    sw = max(0.0, min(cut, c+w) - c)
    if len(L) < 2:
        return sc, sw, 0.0, 0.0
    seg, end = np.diff(L), L[1:]
    keep = end <= cut
    off = d[1:] > TOL
    pon, poff = seg[keep & ~off].sum(), seg[keep & off].sum()
    if crossed:  # overrun beyond a known endpoint counts as wrong in both schemes
        poff += max(0.0, min(cut, c+w) - c)
    return sc, sw, pon, poff


def conf_cuts(dec, n):
    """Cut length per trace for every confidence threshold: first decision with conf16 < thr."""
    cut = np.full((n, len(CONF_GRID)), np.inf)
    for t, g in dec.groupby('trace'):
        g = g.sort_values('decision')
        cf, st = g['conf16'].to_numpy(), g['start_len'].to_numpy()
        for j, thr in enumerate(CONF_GRID):
            hit = np.flatnonzero(cf < thr)
            if len(hit):
                cut[t, j] = st[hit[0]]
    return cut


def fb_cuts(seg, n, min_back, on_frac, prespecified=False):
    cut = np.full(n, np.inf)
    if prespecified:
        fail = ~seg['ok'].astype(bool)
    else:
        fail = (seg['compared_len'] >= min_back) & (seg['frac_on'] < on_frac)
    for t, g in seg[fail].groupby('trace'):
        cut[t] = max(0.0, float(g['at_len'].min()) - 128.0)  # keep up to the previous (passing) check
    return cut


def kept_matrix(cuts, rows, pts, src):
    """Per-trace kept quantities for a (n,) or (n, m) array of cut lengths -> array (n, m, 4)."""
    g = rows[rows.source == src].reset_index(drop=True)
    cuts = cuts.reshape(len(g), -1)
    out = np.zeros(cuts.shape + (4,))
    for i, r in g.iterrows():
        L, d = pts[(src, r.trace)]
        crossed = bool(r.correct < r.length - 1e-6 and not r.diverged)
        for j in range(cuts.shape[1]):
            out[i, j] = kept(cuts[i, j], L, d, r.correct, r.wrong, crossed and r.wrong > 0)
    return out


def matched(curve_c, curve_w, target_c):
    """Wrong length of the confidence rule at the threshold that keeps `target_c` correct length (interpolated)."""
    order = np.argsort(curve_c)
    return np.interp(target_c, curve_c[order], curve_w[order])


def stop_rules(rows, dec, pts):
    grid = [(mb, of) for mb in (16, 32, 64, 128) for of in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)]
    res, cmp_rows, curves = [], [], {}
    sel = None
    for src in ['paris4_heldout'] + [s for s in SOURCES if s != 'paris4_heldout']:
        g = rows[rows.source == src].reset_index(drop=True)
        n = len(g)
        run = ROOT/'runs'/'c050'/src
        seg = pd.read_csv(run/'fwdback_segments.csv')
        full = pd.read_csv(run/'fwdback_full.csv')
        dd = dec[(dec.source == src) & (dec.confidence == 0.5)]
        K_conf = kept_matrix(conf_cuts(dd, n), rows, pts, src)  # (n, thr, 4)
        curves[src] = K_conf.sum(0)
        base = kept_matrix(np.full(n, np.inf), rows, pts, src)[:, 0]
        ret = np.zeros(n, bool)
        ret[full['trace'].to_numpy()] = full['returned'].to_numpy(bool)
        rules = {'baseline (conf 0.5)': np.full(n, np.inf), 'full round trip': np.where(ret, np.inf, 0.0),
                 'segmented, pre-specified': fb_cuts(seg, n, 0, 0, prespecified=True)}
        for mb, of in grid:
            rules[f'segmented min_back={mb} on_frac<{of}'] = fb_cuts(seg, n, mb, of)
        for name, cut in rules.items():
            k = kept_matrix(cut, rows, pts, src)[:, 0]
            s = k.sum(0)
            res.append(dict(source=src, rule=name, strict_correct_mm=s[0]*MM, strict_wrong_mm=s[1]*MM,
                            pw_on_mm=s[2]*MM, pw_off_mm=s[3]*MM, strict_precision=s[0]/max(s[0]+s[1], 1e-9),
                            pointwise_precision=s[2]/max(s[2]+s[3], 1e-9),
                            correct_lost_frac=1-s[0]/base[:, 0].sum(), wrong_removed_frac=1-s[1]/max(base[:, 1].sum(), 1e-9),
                            pw_wrong_removed_frac=1-s[3]/max(base[:, 3].sum(), 1e-9)))
        if src == 'paris4_heldout':  # select segmented parameters on the control only (pre-stated criterion)
            r = pd.DataFrame([x for x in res if x['source'] == src and x['rule'].startswith('segmented min_back')])
            r['score'] = r['wrong_removed_frac'] - r['correct_lost_frac']
            r = r.sort_values(['score', 'rule'], ascending=[False, True])
            sel = r.iloc[0]['rule']
        # Matched comparison against confidence-threshold stopping, with trace bootstrap.
        weights = np.stack([np.bincount(b, minlength=n) for b in boot_idx(n)]).astype(float)  # (B, n)
        KC = (weights @ K_conf.reshape(n, -1)).reshape(B, len(CONF_GRID), 4)
        for name in ('full round trip', 'segmented, pre-specified', sel):
            k = kept_matrix(rules[name], rows, pts, src)[:, 0]
            kb = weights @ k  # (B, 4)
            for scheme, (ci_, wi_) in (('strict', (0, 1)), ('pointwise', (2, 3))):
                tc, tw = k[:, ci_].sum(), k[:, wi_].sum()
                cw = matched(K_conf[:, :, ci_].sum(0), K_conf[:, :, wi_].sum(0), tc)
                bs = [1 - kb[b, wi_]/max(matched(KC[b, :, ci_], KC[b, :, wi_], kb[b, ci_]), 1e-9) for b in range(B)]
                lo, hi = ci(np.asarray(bs))
                cmp_rows.append(dict(source=src, rule=name, scheme=scheme, correct_kept_mm=tc*MM, wrong_kept_mm=tw*MM,
                                     conf_rule_wrong_at_same_correct_mm=cw*MM,
                                     wrong_reduction_vs_conf=1-tw/max(cw, 1e-9), lo=lo, hi=hi,
                                     p_boot_not_better=float(np.mean(np.asarray(bs) <= 0)),
                                     selected_on='paris4_heldout' if name == sel else 'fixed'))
    return pd.DataFrame(res), pd.DataFrame(cmp_rows), curves, sel


def auc_table(dec):
    out = []
    for (src, conf), g in dec.groupby(['source', 'confidence']):
        lab = g[g['label'].isin(['before', 'clean_trace', 'at_departure'])].reset_index(drop=True)
        y = (lab['label'] == 'at_departure').to_numpy()
        traces = lab['trace'].to_numpy()
        ut = np.unique(traces)
        by = {t: np.flatnonzero(traces == t) for t in ut}
        for sig in ('s_conf16', 's_overlap', 's_dheading', 's_dnormal', 's_turn'):
            v = lab[sig].to_numpy(float)
            ok = np.isfinite(v)
            a = roc_auc_score(y[ok], v[ok])
            bs = []
            for _ in range(1000):
                pick = np.concatenate([by[t] for t in RNG.choice(ut, len(ut))])
                pick = pick[ok[pick]]
                if 0 < y[pick].sum() < len(pick):
                    bs.append(roc_auc_score(y[pick], v[pick]))
            lo, hi = ci(np.asarray(bs))
            out.append(dict(source=src, confidence=conf, signal=sig, auc=a, lo=lo, hi=hi,
                            positive_decisions=int(y[ok].sum()), decisions=int(ok.sum())))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------------- figures
def figures(summ, curves, rules, cmp_, dec, eps):
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SOURCES))
    # 1. precision, strict vs pointwise, both confidences
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
    for k, (conf, metric, col, lab) in enumerate(((0.5, 'strict_precision', '#c0392b', 'conf 0.5 strict'),
                                                   (0.5, 'pointwise_precision', '#e8a29a', 'conf 0.5 pointwise'),
                                                   (0.7, 'strict_precision', '#1f5f99', 'conf 0.7 strict'),
                                                   (0.7, 'pointwise_precision', '#93b8dc', 'conf 0.7 pointwise'))):
        s = summ[summ.confidence == conf].set_index('source').loc[list(SOURCES)]
        ax.bar(x + (k-1.5)*0.2, s[metric], 0.2, color=col, label=lab,
               yerr=[s[metric]-s[metric+'_lo'], s[metric+'_hi']-s[metric]], capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(x, [LABEL[s] for s in SOURCES]); ax.set_ylim(0, 1.05); ax.set_ylabel('length precision')
    ax.legend(fontsize=7, ncol=2, loc='lower left'); ax.set_title('Length precision with 95% bootstrap CIs (by fiber)', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/'fig1_precision.png'); plt.close(fig)
    # 2. event hazards per cm of on-fiber tracing at conf 0.5
    s = summ[summ.confidence == 0.5].set_index('source').loc[list(SOURCES)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=130)
    for k, (m, col, lab) in enumerate((('departures_per_cm', '#7f8c8d', 'strict departures (villa scorer)'),
                                       ('returned_excursions_per_cm', '#f1c40f', 'excursions that return'),
                                       ('persistent_per_cm', '#e67e22', 'persistent (drift + switch)'),
                                       ('hard_switches_per_cm', '#c0392b', f'hard switches (> {HARD:.0f} vx)'))):
        ax.bar(x + (k-1.5)*0.2, s[m], 0.2, color=col, label=lab, yerr=[s[m]-s[m+'_lo'], s[m+'_hi']-s[m]], capsize=2,
               error_kw=dict(lw=0.8))
    ax.set_xticks(x, [LABEL[q] for q in SOURCES]); ax.set_ylabel('events per cm traced on-fiber')
    ax.legend(fontsize=7); ax.set_title('Error events per cm at confidence 0.5 (95% bootstrap CIs)', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/'fig2_hazards.png'); plt.close(fig)
    # 3. trade-off curves
    for scheme, (ci_, wi_) in (('strict', (0, 1)), ('pointwise', (2, 3))):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), dpi=130)
        for ax, src in zip(axes, SOURCES):
            cv = curves[src]
            ax.plot(cv[:, ci_]*MM, cv[:, wi_]*MM, '-', color='#1f5f99', lw=1.2, label='confidence threshold 0.50-0.995')
            for thr, mk in ((0.5, 'o'), (0.7, 's'), (0.9, '^')):
                j = int(np.argmin(np.abs(CONF_GRID-thr)))
                ax.plot(cv[j, ci_]*MM, cv[j, wi_]*MM, mk, color='#1f5f99', ms=4)
                ax.annotate(f'{thr}', (cv[j, ci_]*MM, cv[j, wi_]*MM), fontsize=6, xytext=(3, 2), textcoords='offset points')
            r = rules[rules.source == src].set_index('rule')
            col = 'strict_correct_mm' if scheme == 'strict' else 'pw_on_mm'
            wcol = 'strict_wrong_mm' if scheme == 'strict' else 'pw_off_mm'
            ax.plot(r.loc['full round trip', col], r.loc['full round trip', wcol], 'D', color='#8e44ad', ms=5, label='full round trip')
            ax.plot(r.loc['segmented, pre-specified', col], r.loc['segmented, pre-specified', wcol], '*', color='#c0392b',
                    ms=9, label='segmented fwd/back (pre-specified)')
            real = summ[(summ.source == src) & (summ.confidence == 0.7)].iloc[0]
            rc, rw = (real.correct_mm, real.strict_wrong_mm) if scheme == 'strict' else (real.onfiber_cm*10, real.pointwise_wrong_mm)
            ax.plot(rc, rw, 'x', color='k', ms=6, label='real run at conf 0.7 (check)')
            ax.set_title(LABEL[src], fontsize=9); ax.set_xlabel('correct length kept (mm)', fontsize=8)
            ax.tick_params(labelsize=7); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        axes[0].set_ylabel(f'wrong length kept (mm, {scheme})', fontsize=8); axes[0].legend(fontsize=6)
        fig.suptitle(f'Stop rules on the conf 0.5 traces ({scheme} scoring): lower-right is better', fontsize=9)
        fig.tight_layout(); fig.savefig(FIG/f'fig3_tradeoff_{scheme}.png'); plt.close(fig)
    # 4. confidence at departure vs all at-risk decisions
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8), dpi=130, sharey=False)
    for ax, src in zip(axes, SOURCES):
        g = dec[(dec.source == src) & (dec.confidence == 0.5)]
        risk = g[g['label'].isin(['before', 'clean_trace'])]['conf16']
        dep = g[g['label'] == 'at_departure']['conf16']
        bins = np.linspace(0.4, 1, 31)
        ax.hist(risk, bins, density=True, alpha=0.5, color='#1f5f99', label='on-fiber decisions')
        ax.hist(dep, bins, density=True, alpha=0.6, color='#c0392b', label='decisions at a departure')
        ax.axvline(0.7, color='k', lw=0.6, ls='--'); ax.set_title(LABEL[src], fontsize=9); ax.tick_params(labelsize=7)
        ax.set_xlabel('model confidence (16th point)', fontsize=8)
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG/'fig4_confidence.png'); plt.close(fig)
    # 5. departure episode peaks
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=130)
    for k, src in enumerate(SOURCES):
        e = eps[(eps.source == src)]
        y = np.full(len(e), k) + RNG.uniform(-0.25, 0.25, len(e))
        col = e['cls'].map({'returned': '#f1c40f', 'drift': '#e67e22', 'hard_switch': '#c0392b', 'ended': '#7f8c8d'})
        ax.scatter(np.clip(e['peak'], 1, 3000), y, c=col, s=10)
    ax.set_xscale('log'); ax.axvline(TOL, color='k', lw=0.6, ls=':'); ax.axvline(HARD, color='k', lw=0.6, ls='--')
    ax.set_yticks(range(len(SOURCES)), [LABEL[s] for s in SOURCES]); ax.set_xlabel('peak distance from GT fiber (vx, 1 vx = 19.2 um)')
    for c, lab in (('#f1c40f', 'returned'), ('#e67e22', 'persistent drift'), ('#c0392b', 'hard switch'), ('#7f8c8d', 'trace ended')):
        ax.scatter([], [], c=c, s=10, label=lab)
    ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.0, 0.5)); ax.set_title('Every excursion at conf 0.5 (dotted: 3 vx tolerance, dashed: 15 vx)', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/'fig5_excursions.png'); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dec = pd.read_csv(ROOT/'results'/'baseline'/'decisions.csv.gz')
    tables, eps_tables, pts05 = [], [], None
    for conf, d in ((0.5, 'c050'), (0.7, 'c070')):
        rows, eps, pts = load(d)
        rows['confidence'], eps['confidence'] = conf, conf
        tables.append(summary_table(rows, dec, conf))
        eps_tables.append(eps)
        rows.to_csv(OUT/f'traces_{d}.csv', index=False)
        if conf == 0.5:
            rows05, pts05 = rows, pts
            sensitivity(rows, pts).to_csv(OUT/'class_sensitivity.csv', index=False)
    summ = pd.concat(tables, ignore_index=True)
    eps = pd.concat(eps_tables, ignore_index=True)
    summ.to_csv(OUT/'summary_ci.csv', index=False)
    eps.to_csv(OUT/'excursions.csv', index=False)
    tests = pd.concat([rate_tests(summ[summ.confidence == c]) for c in (0.5, 0.7)], ignore_index=True)
    # Same scroll, conf 0.7 vs 0.5: does a higher threshold lower the per-length hazard?
    for src in SOURCES:
        a, b = summ[(summ.source == src) & (summ.confidence == 0.7)].iloc[0], summ[(summ.source == src) & (summ.confidence == 0.5)].iloc[0]
        for ev, ex, lab in (('departures', 'exposure_cm', 'strict departures per cm'),
                            ('departures', 'decisions_at_risk', 'strict departures per decision'),
                            ('hard_events', 'onfiber_cm', 'hard switches per cm')):
            rr, (lo, hi), p = rate_ratio(a[ev], a[ex], b[ev], b[ex])
            tests.loc[len(tests)] = dict(confidence='0.7 vs 0.5', source=src, vs=src+' @0.5', metric=lab, events=a[ev],
                                         exposure=a[ex], ref_events=b[ev], ref_exposure=b[ex], rate_ratio=rr, lo=lo, hi=hi, p_exact=p)
    for ev, ex, lab in (('departures', 'exposure_cm', 'strict departures per cm'),
                        ('departures', 'decisions_at_risk', 'strict departures per decision'),
                        ('hard_events', 'onfiber_cm', 'hard switches per cm')):
        st = []
        for src in SOURCES:
            a = summ[(summ.source == src) & (summ.confidence == 0.7)].iloc[0]
            b = summ[(summ.source == src) & (summ.confidence == 0.5)].iloc[0]
            st.append((a[ev], a[ex], b[ev], b[ex]))
        rr, p = stratified_rate_ratio(st)
        tests.loc[len(tests)] = dict(confidence='0.7 vs 0.5', source='all 4 (stratified MH)', vs='same @0.5', metric=lab,
                                     events=sum(x[0] for x in st), exposure=sum(x[1] for x in st), ref_events=sum(x[2] for x in st),
                                     ref_exposure=sum(x[3] for x in st), rate_ratio=rr, lo=np.nan, hi=np.nan, p_exact=p)
    for conf in (0.5, 0.7):  # three held-out scrolls pooled vs the Paris4 control
        h = summ[(summ.confidence == conf) & (summ.source != 'paris4_heldout')]
        r0 = summ[(summ.confidence == conf) & (summ.source == 'paris4_heldout')].iloc[0]
        for ev, ex, lab in (('departures', 'exposure_cm', 'strict departures per cm'),
                            ('departures', 'decisions_at_risk', 'strict departures per decision'),
                            ('hard_events', 'onfiber_cm', 'hard switches per cm')):
            rr, (lo, hi), p = rate_ratio(h[ev].sum(), h[ex].sum(), r0[ev], r0[ex])
            tests.loc[len(tests)] = dict(confidence=conf, source='held-out pooled', vs='paris4_heldout', metric=lab,
                                         events=h[ev].sum(), exposure=h[ex].sum(), ref_events=r0[ev], ref_exposure=r0[ex],
                                         rate_ratio=rr, lo=lo, hi=hi, p_exact=p)
    tests.to_csv(OUT/'rate_tests.csv', index=False)
    rules, cmp_, curves, sel = stop_rules(rows05, dec, pts05)
    rules.to_csv(OUT/'stop_rules.csv', index=False)
    cmp_.to_csv(OUT/'stop_rules_vs_confidence.csv', index=False)
    pd.DataFrame([dict(source=s, threshold=t, strict_correct_mm=c[0]*MM, strict_wrong_mm=c[1]*MM, pw_on_mm=c[2]*MM,
                       pw_off_mm=c[3]*MM) for s, cv in curves.items() for t, c in zip(CONF_GRID, cv)]).to_csv(OUT/'confidence_curves.csv', index=False)
    auc = auc_table(dec)
    auc.to_csv(OUT/'signal_auc_ci.csv', index=False)
    figures(summ, curves, rules, cmp_, dec, eps[eps.confidence == 0.5])
    (OUT/'run_info.json').write_text(json.dumps(dict(tol=TOL, patience=PATIENCE, return_run=RETURN_RUN, hard=HARD,
                                                     bootstrap=B, seed=7349, selected_segmented_rule=sel), indent=1))
    pd.set_option('display.width', 250)
    cols = ['source', 'confidence', 'traces', 'departures', 'first_returned', 'first_drift', 'first_hard', 'first_ended',
            'strict_precision', 'pointwise_precision', 'coverage', 'departures_per_100_decisions', 'departures_per_cm',
            'hard_switches_per_cm']
    print(summ[cols].round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print('selected on Paris4:', sel)
    print(cmp_.round(3).to_string(index=False))
    print(auc.pivot_table(index='signal', columns=['source', 'confidence'], values='auc').round(3).to_string())


if __name__ == '__main__':
    main()
