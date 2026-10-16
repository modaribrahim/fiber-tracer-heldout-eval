"""Error detection from the saved runs (conf 0.5): can forward/backward checks and confidence flag departures?

Unit = one segmented fwd/back check at forward length A; it inspects the forward stretch [A-192, A].
  y = 1  the stretch has >= 3 vx of path farther than 3 vx from the GT fiber (pointwise, scored region only)
  y = 0  the stretch is entirely on the fiber
Scores: fb = 1 - frac_on of the back trace; conf = 1 - min conf16 over decisions in the stretch; combo = logistic
regression on both plus max deviation and an early-stop flag, fit leave-one-scroll-out (never on the scored scroll).
Thresholds for event recall are fixed on Paris4 clean checks (5% false-alarm rate) and applied unchanged to the
unseen scrolls. Writes results/final/detection_*.csv and figures/fig6_detection.png. CPU only.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import stats as S

BACK, MINBAD, FPR = 192.0, 3.0, 0.05
FEATS = ['fb', 'logdev', 'conf', 'confmean', 'short']


def checks_table(rows, pts, dec, conf_dir='c050'):
    out = []
    for src in S.SOURCES:
        seg = pd.read_csv(S.ROOT/'runs'/conf_dir/src/'fwdback_segments.csv')
        d = dec[(dec.source == src) & (dec.confidence == 0.5)]
        dg = {t: g.sort_values('start_len') for t, g in d.groupby('trace')}
        for c in seg.itertuples():
            L, dist = pts[(src, c.trace)]
            lo, hi = c.at_len - BACK, c.at_len
            m = (L[1:] > lo) & (L[1:] <= hi)
            if not m.any() or L[-1] < hi - 1e-6:  # stretch past the scored region (endpoint crossing): unscored
                continue
            off = float(np.diff(L)[m & (dist[1:] > S.TOL)].sum())
            y = 1 if off >= MINBAD else (0 if off == 0 else -1)
            g = dg.get(c.trace)
            cf = g[(g.start_len > lo) & (g.start_len <= hi)]['conf16'].to_numpy() if g is not None else np.zeros(0)
            known = c.compared_len >= 16
            out.append(dict(source=src, trace=c.trace, at_len=c.at_len, y=y, off_len=off,
                            fb=1 - c.frac_on if known else 0.0,
                            logdev=np.log1p(c.max_dev) if known else 0.0,
                            conf=1 - cf.min() if len(cf) else 0.0, confmean=1 - cf.mean() if len(cf) else 0.0,
                            short=float(c.back_len < min(BACK, c.at_len - 4.0) * 0.5)))
    return pd.DataFrame(out)


def loso_combo(ch):
    ch = ch.copy()
    ch['combo'] = np.nan
    lab = ch[ch.y >= 0]
    for src in S.SOURCES:
        tr = lab[lab.source != src]
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000).fit(tr[FEATS], tr.y)
        ch.loc[ch.source == src, 'combo'] = clf.predict_proba(ch.loc[ch.source == src, FEATS])[:, 1]
    return ch


def freeze(ch):
    """Detector frozen on all of run v1 (before run v2 existed): combo fit on every labeled v1 check, and each
    signal's threshold at a 5% false-alarm rate on v1 Paris4 clean checks. Applied unchanged to run v2."""
    import json
    lab = ch[ch.y >= 0]
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000).fit(lab[FEATS], lab.y)
    ref = lab[(lab.source == 'paris4_heldout') & (lab.y == 0)]
    combo = clf.predict_proba(ref[FEATS])[:, 1]
    frozen = dict(features=FEATS, coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                  thresholds=dict(fb=float(np.quantile(ref.fb, 1 - FPR)), conf=float(np.quantile(ref.conf, 1 - FPR)),
                                  combo=float(np.quantile(combo, 1 - FPR))),
                  back=BACK, min_bad=MINBAD, fpr=FPR, fitted_on='run v1 (results/final), all 4 sources, %d checks' % len(lab))
    (S.OUT/'detector_frozen.json').write_text(json.dumps(frozen, indent=1))
    return frozen


def apply_frozen(ch, frozen):
    z = ch[frozen['features']].to_numpy() @ np.asarray(frozen['coef']) + frozen['intercept']
    ch = ch.copy()
    ch['combo'] = 1 / (1 + np.exp(-z))
    return ch


def auc_ci(g, sig):
    y, v, tr = g.y.to_numpy(), g[sig].to_numpy(), g.trace.to_numpy()
    ut = np.unique(tr)
    by = {t: np.flatnonzero(tr == t) for t in ut}
    bs = []
    for _ in range(2000):
        p = np.concatenate([by[t] for t in S.RNG.choice(ut, len(ut))])
        if 0 < y[p].sum() < len(p):
            bs.append(roc_auc_score(y[p], v[p]))
    return roc_auc_score(y, v), *S.ci(np.asarray(bs))


def main():
    rows, eps, pts = S.load('c050')
    dec = pd.read_csv(S.ROOT/'results'/'baseline'/'decisions.csv.gz')
    ch = loso_combo(checks_table(rows, pts, dec))
    print('frozen detector:', freeze(ch))
    lab = ch[ch.y >= 0]
    ref = lab[(lab.source == 'paris4_heldout') & (lab.y == 0)]
    thr = {s: float(np.quantile(ref[s], 1 - FPR)) for s in ('fb', 'conf', 'combo')}
    aucs, ev = [], []
    for src in S.SOURCES:
        g = lab[lab.source == src]
        clean_cm = (g.y == 0).sum() * 128 * S.MM / 10  # each check adds ~128 vx of newly checked path
        for s in ('fb', 'conf', 'combo'):
            a, lo, hi = auc_ci(g, s)
            fa = int(((g.y == 0) & (g[s] > thr[s])).sum())
            aucs.append(dict(source=src, signal=s, auc=a, lo=lo, hi=hi, pos=int(g.y.sum()), neg=int((g.y == 0).sum()),
                             threshold=thr[s], tpr=float((g[g.y == 1][s] > thr[s]).mean()),
                             fpr=float((g[g.y == 0][s] > thr[s]).mean()), false_alarms_per_cm=fa / clean_cm))
        # Event recall: a departure is caught if a check within BACK after its onset exceeds the threshold.
        cs = ch[ch.source == src]
        for e in eps[eps.source == src].itertuples():
            c = cs[(cs.trace == e.trace) & (cs.at_len >= e.onset_len) & (cs.at_len <= e.onset_len + BACK)]
            ev.append(dict(source=src, trace=e.trace, onset_len=e.onset_len, cls=e.cls, peak=e.peak, checked=len(c) > 0,
                           **{f'caught_{s}': bool(len(c) and (c[s] > thr[s]).any()) for s in ('fb', 'conf', 'combo')}))
    aucs, ev = pd.DataFrame(aucs), pd.DataFrame(ev)
    ch.to_csv(S.OUT/'detection_checks.csv', index=False)
    aucs.to_csv(S.OUT/'detection_auc.csv', index=False)
    ev.to_csv(S.OUT/'detection_events.csv', index=False)
    pd.set_option('display.width', 220)
    print('thresholds (Paris4 clean checks, 5% FPR):', {k: round(v, 3) for k, v in thr.items()})
    print(aucs.round(3).to_string(index=False))
    e = ev[ev.checked]
    print('\nevent recall (departures with a check within 192 vx after onset):')
    print(e.groupby(['source', 'cls'])[['caught_fb', 'caught_conf', 'caught_combo']].agg(['mean', 'size']).round(2).to_string())
    print(e.groupby('source')[['caught_fb', 'caught_conf', 'caught_combo']].mean().round(2).to_string())
    print('unchecked departures:', int((~ev.checked).sum()), 'of', len(ev))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for a, src in zip(ax, S.SOURCES):
        g = lab[lab.source == src]
        for s, col in (('fb', 'C0'), ('conf', 'C1'), ('combo', 'C3')):
            f, t, _ = roc_curve(g.y, g[s])
            r = aucs[(aucs.source == src) & (aucs.signal == s)].iloc[0]
            a.plot(f, t, col, label=f'{s} AUC {r.auc:.2f} [{r.lo:.2f},{r.hi:.2f}]')
        a.plot([0, 1], [0, 1], 'k:', lw=.8)
        a.set_title(f'{src}  (bad {int(g.y.sum())} / clean {int((g.y == 0).sum())})')
        a.set_xlabel('false-alarm rate (clean stretches)')
        a.legend(fontsize=8, loc='lower right')
    ax[0].set_ylabel('detection rate (stretches with off-fiber path)')
    fig.suptitle('Flagging off-fiber stretches: fwd/back check vs confidence vs combination (leave-one-scroll-out)')
    fig.tight_layout()
    fig.savefig(S.OUT/'figures'/'fig6_detection.png', dpi=120)


if __name__ == '__main__':
    main()
