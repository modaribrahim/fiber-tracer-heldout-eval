"""Confirmatory analysis of run v2, exactly as fixed in ANALYSIS_PLAN_v2.md (CPU only, no villa import).

Inputs: runs/v2/<source>/ (rows.csv, paths.jsonl, fwdback_*.csv, img/image_features_*.csv), results/v2/decisions.csv.gz
(analyze.py decision table), results/final/detector_frozen.json (fitted on run v1 only; never refit here).
Outputs: results/v2/*.csv, results/v2/confirm_stdout.txt, results/v2/figures/*.png. Bootstrap unit: fiber.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

import stats as S
import detect as D

RUN = os.environ.get('CONFIRM_RUN', 'v2')  # override only for a dry run on v1 data
S.RUN_DIR = RUN
OUT = Path(os.environ.get('CONFIRM_OUT', S.ROOT/'results'/'v2'))
SIGS = ('fb', 'conf', 'combo')
UNSEEN = ('PHerc0139', 'PHerc1451', 'PHerc0332')


def boot_ratio(groups, num, den, b=S.B):
    W = S.boot_weights(groups, b)
    return (W @ num)/np.maximum(W @ den, 1e-12)


def recall_ci(e, col):
    """Recall with a fiber-cluster bootstrap CI."""
    if not len(e):
        return np.nan, np.nan, np.nan
    y = e[col].to_numpy(float)
    s = boot_ratio(e.fiber_name, y, np.ones(len(y)))
    return y.mean(), *S.ci(s)


def figures(summ, rec, img):
    import matplotlib.pyplot as plt
    names = list(S.SOURCES)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, m, title in ((ax[0], 'departures_per_cm', 'strict departures per cm'), (ax[1], 'hard_switches_per_cm', 'hard switches per cm')):
        s = summ.set_index('source').loc[names]
        a.bar(names, s[m], yerr=[s[m]-s[m+'_lo'], s[m+'_hi']-s[m]], capsize=4, color=['grey', 'C0', 'C1', 'C2'])
        a.set_title(f'{title} (95% fiber-bootstrap CI)'); a.tick_params(axis='x', labelsize=8)
    fig.suptitle('Run v2: error rates, Paris4 hash-held-out control vs three never-trained scrolls (conf 0.5)')
    fig.tight_layout(); fig.savefig(OUT/'figures'/'fig7_v2_rates.png', dpi=120); plt.close(fig)

    groups = names + ['unseen pooled']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for a, cls in zip(ax, ('hard_switch', 'all persistent', 'returned')):
        r = rec[rec.cls == cls].set_index('source').loc[groups]
        x = np.arange(len(groups))
        for k, (s, col, lab) in enumerate((('fb', 'C0', 'fwd/back (fb>0)'), ('conf', 'C1', 'confidence (min conf16<0.566)'),
                                           ('combo', 'C3', 'combined (frozen v1 model)'))):
            a.bar(x + (k-1)*0.27, r[s], 0.27, yerr=[(r[s]-r[s+'_lo']).clip(0), (r[s+'_hi']-r[s]).clip(0)], capsize=3, color=col, label=lab)
        a.set_xticks(x, [f'{g}\nn={int(n)}' for g, n in zip(groups, r['n'])], fontsize=7)
        a.set_title(f'{cls}: fraction caught within 192 vx'); a.set_ylim(0, 1.05)
    ax[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Run v2: departures flagged at thresholds frozen on run v1 (95% fiber-bootstrap CI)')
    fig.tight_layout(); fig.savefig(OUT/'figures'/'fig8_v2_detection.png', dpi=120); plt.close(fig)

    if img is not None and len(img):
        feats = ('layer_gap_vx', 'fiber_gap_vx', 'contrast', 'gt_neighbor_vx')
        fig, ax = plt.subplots(1, len(feats), figsize=(16, 4))
        for a, feat in zip(ax, feats):
            sets = [('control', img[img.kind == 'control']), ('returned', img[img.kind == 'returned']),
                    ('switch caught', img[(img.kind == 'hard_switch') & (img.caught_fb == True)]),  # noqa: E712
                    ('switch missed', img[(img.kind == 'hard_switch') & (img.caught_fb == False)])]  # noqa: E712
            data = [g[feat].dropna().to_numpy() for _, g in sets]
            a.boxplot(data, showfliers=False)
            a.set_xticks(range(1, len(sets)+1), [f'{n}\nn={len(d)}' for (n, _), d in zip(sets, data)], fontsize=7)
            a.set_title(feat, fontsize=9)
        fig.suptitle('Run v2, exploratory: CT features at excursion onsets vs on-fiber controls (all scrolls)')
        fig.tight_layout(); fig.savefig(OUT/'figures'/'fig9_v2_image_features.png', dpi=120); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    frozen = json.loads((S.ROOT/'results'/'final'/'detector_frozen.json').read_text())
    thr = frozen['thresholds']
    rows, eps, pts = S.load(RUN)  # asserts villa's departure index / length split are reproduced for every trace
    dec = pd.read_csv(OUT/'decisions.csv.gz')
    print(f'traces: {rows.groupby("source").size().to_dict()}  fibers: {rows.groupby("source").fiber_name.nunique().to_dict()}')

    # Q1 generalization ------------------------------------------------------------------------------------------
    summ = S.summary_table(rows, dec, 0.5)
    summ.to_csv(OUT/'summary_ci.csv', index=False)
    tests = S.rate_tests(summ)
    h, r0 = summ[summ.source.isin(UNSEEN)], summ[summ.source == 'paris4_heldout'].iloc[0]
    for ev, ex, lab in (('departures', 'exposure_cm', 'strict departures per cm'),
                        ('departures', 'decisions_at_risk', 'strict departures per decision'),
                        ('hard_events', 'onfiber_cm', 'hard switches per cm')):
        rr, (lo, hi), p = S.rate_ratio(h[ev].sum(), h[ex].sum(), r0[ev], r0[ex])
        tests.loc[len(tests)] = dict(confidence=0.5, source='unseen pooled', vs='paris4_heldout', metric=lab, events=h[ev].sum(),
                                     exposure=h[ex].sum(), ref_events=r0[ev], ref_exposure=r0[ex], rate_ratio=rr, lo=lo, hi=hi, p_exact=p)
    # Fiber-bootstrap CI of the pooled-unseen / Paris4 departures-per-cm ratio (groups resampled independently).
    u, p4 = rows[rows.source.isin(UNSEEN)], rows[rows.source == 'paris4_heldout']
    ru = boot_ratio(u.source + '/' + u.fiber_name, u.diverged.to_numpy(float), u.correct.to_numpy(float)*S.MM/10)
    rp = boot_ratio(p4.fiber_name, p4.diverged.to_numpy(float), p4.correct.to_numpy(float)*S.MM/10)
    q1 = (ru/rp)
    q1_lo, q1_hi = S.ci(q1)
    tests.to_csv(OUT/'rate_tests.csv', index=False)

    # Q2-Q4 detection with the frozen v1 detector ----------------------------------------------------------------
    ch = D.apply_frozen(D.checks_table(rows, pts, dec, RUN), frozen)
    ch.to_csv(OUT/'detection_checks.csv', index=False)
    lab = ch[ch.y >= 0]
    aucs, ev = [], []
    for src in S.SOURCES:
        g = lab[lab.source == src]
        clean = g[g.y == 0]
        for s in SIGS:
            a, lo, hi = D.auc_ci(g, s)
            fa = (clean[s] > thr[s]).to_numpy(float)
            fa_cm = boot_ratio(clean.fiber_name, fa, np.full(len(clean), 128*S.MM/10))
            aucs.append(dict(source=src, signal=s, auc=a, lo=lo, hi=hi, pos=int(g.y.sum()), neg=len(clean), threshold=thr[s],
                             stretch_tpr=float((g[g.y == 1][s] > thr[s]).mean()), stretch_fpr=float(fa.mean()),
                             false_alarms_per_cm=fa.sum()/(len(clean)*128*S.MM/10), fa_cm_lo=S.ci(fa_cm)[0], fa_cm_hi=S.ci(fa_cm)[1]))
        cs = ch[ch.source == src]
        fname = dict(zip(rows[rows.source == src].trace, rows[rows.source == src].fiber_name))
        for e in eps[eps.source == src].itertuples():
            c = cs[(cs.trace == e.trace) & (cs.at_len >= e.onset_len) & (cs.at_len <= e.onset_len + frozen['back'])]
            ev.append(dict(source=src, trace=e.trace, fiber_name=fname[e.trace], onset_idx=e.onset_idx, onset_len=e.onset_len,
                           cls=e.cls, peak=e.peak, checked=len(c) > 0, **{f'caught_{s}': bool(len(c) and (c[s] > thr[s]).any()) for s in SIGS}))
    aucs, ev = pd.DataFrame(aucs), pd.DataFrame(ev)
    aucs.to_csv(OUT/'detection_auc.csv', index=False)
    ev.to_csv(OUT/'detection_events.csv', index=False)

    rec = []
    for src in list(S.SOURCES) + ['unseen pooled']:
        e = ev[ev.checked & (ev.source.isin(UNSEEN) if src == 'unseen pooled' else ev.source == src)].copy()
        e['fiber_name'] = e.source + '/' + e.fiber_name
        for cls in ('hard_switch', 'drift', 'returned', 'all persistent', 'all'):
            x = e if cls == 'all' else e[e.cls.isin(('hard_switch', 'drift'))] if cls == 'all persistent' else e[e.cls == cls]
            row = dict(source=src, cls=cls, n=len(x))
            for s in SIGS:
                row[s], row[s+'_lo'], row[s+'_hi'] = recall_ci(x, 'caught_'+s)
            b = int((x.caught_fb & ~x.caught_conf).sum()); c = int((~x.caught_fb & x.caught_conf).sum())
            row.update(fb_only=b, conf_only=c, mcnemar_p=binomtest(b, b+c, 0.5).pvalue if b+c else 1.0)
            rec.append(row)
    rec = pd.DataFrame(rec)
    rec.to_csv(OUT/'detection_recall.csv', index=False)

    # Q5 stop rule vs confidence at equal correct length --------------------------------------------------------
    rules, cmp_, curves, sel = S.stop_rules(rows, dec, pts)
    rules.to_csv(OUT/'stop_rules.csv', index=False)
    cmp_.to_csv(OUT/'stop_rules_vs_confidence.csv', index=False)

    # Exploratory: 0139 fibers never seeded in v1; image features ------------------------------------------------
    v1 = set(pd.read_csv(S.ROOT/'runs'/'c050'/'PHerc0139'/'rows.csv').fiber_name)
    new = rows[(rows.source == 'PHerc0139') & ~rows.fiber_name.isin(v1)]
    new_rate = new.diverged.sum()/(new.correct.sum()*S.MM/10)
    nb = boot_ratio(new.fiber_name, new.diverged.to_numpy(float), new.correct.to_numpy(float)*S.MM/10)
    img = []
    for src in S.SOURCES:
        f = S.ROOT/'runs'/RUN/src/'img'/f'image_features_{src}.csv'
        if f.exists():
            img.append(pd.read_csv(f))
    img_out = None
    if img:
        img = pd.concat(img, ignore_index=True)
        img = img.merge(ev[['source', 'trace', 'onset_idx', 'checked', 'caught_fb', 'caught_conf']],
                        left_on=['source', 'trace', 'idx'], right_on=['source', 'trace', 'onset_idx'], how='left')
        img.to_csv(OUT/'image_features.csv', index=False)
        res = []
        for src in list(S.SOURCES) + ['unseen pooled', 'all']:
            g = img if src == 'all' else img[img.source.isin(UNSEEN)] if src == 'unseen pooled' else img[img.source == src]
            hs, ctl = g[g.kind == 'hard_switch'], g[g.kind == 'control']
            caught, missed = hs[hs.checked == True], hs[hs.checked == True]  # noqa: E712 (NaN-safe)
            caught, missed = caught[caught.caught_fb == True], missed[missed.caught_fb == False]  # noqa: E712
            for feat in ('layer_gap_vx', 'fiber_gap_vx', 'contrast', 'center_rel', 'gt_neighbor_vx'):
                for name, a, b in (('hard_switch vs control', hs, ctl), ('fwd/back missed vs caught', missed, caught)):
                    x, y = a[feat].dropna(), b[feat].dropna()
                    p = mannwhitneyu(x, y).pvalue if len(x) >= 3 and len(y) >= 3 else np.nan
                    res.append(dict(source=src, feature=feat, comparison=name, n_a=len(x), n_b=len(y),
                                    median_a=x.median() if len(x) else np.nan, median_b=y.median() if len(y) else np.nan, p_mwu=p))
        img_out = pd.DataFrame(res)
        img_out.to_csv(OUT/'image_feature_tests.csv', index=False)

    figures(summ, rec, img if img_out is not None else None)

    # Report -----------------------------------------------------------------------------------------------------
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    cols = ['source', 'traces', 'departures', 'first_hard', 'strict_precision', 'strict_precision_lo', 'strict_precision_hi',
            'pointwise_precision', 'coverage', 'departures_per_cm', 'departures_per_cm_lo', 'departures_per_cm_hi',
            'hard_switches_per_cm', 'hard_switches_per_cm_lo', 'hard_switches_per_cm_hi']
    print('\n== Q1 generalization (v2)\n' + summ[cols].round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(f'pooled unseen / Paris4 departures per cm: fiber-bootstrap 95% CI [{q1_lo:.2f}, {q1_hi:.2f}]')
    print('\n== Q2 recall at frozen v1 thresholds (checked departures)\n' + rec.round(3).to_string(index=False))
    print('\n== Q3/Q4 stretch-level AUROC and false alarms\n' + aucs.round(3).to_string(index=False))
    print('\n== Q5 stop rule vs confidence at equal correct length\n' + cmp_.round(3).to_string(index=False))
    print(f'\n== Exploratory: 0139 fibers not seeded in v1: {new.fiber_name.nunique()} fibers, {len(new)} traces, '
          f'departures/cm {new_rate:.3f} [{S.ci(nb)[0]:.3f}, {S.ci(nb)[1]:.3f}]')
    if img_out is not None:
        print('\n== Exploratory: image features\n' + img_out[img_out.source.isin(['unseen pooled', 'all', 'paris4_heldout'])].round(4).to_string(index=False))
    print('unchecked departures:', int((~ev.checked).sum()), 'of', len(ev))


if __name__ == '__main__':
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    (OUT/'confirm_stdout.txt').write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
