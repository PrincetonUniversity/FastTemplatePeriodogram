"""WP E5: build the stratified 24-star subset for the sparse-regime link.

E5 measures recovery vs N_epochs (per-band random subsampling) on REAL ZTF
light curves.  The FTP methods cost ~5.5-6.4 ms/freq INDEPENDENT of N, so the
sample is scoped by design to 24 stars (x 5 N x 3 seeds x 6 methods).

Selection rule (deterministic given E5_SEED; every input committed)
-------------------------------------------------------------------
Inputs: ../e4_recovery/star_table.json (73 unique stars),
        ../e4_recovery/e4_scores.csv  (full-N E4 outcomes = saturation
        anchors; also provides median r mag per star),
        ../e4_recovery/e4_rates.json  (the E4 science mag-tercile edges,
        reused verbatim so E5 strata line up with E4's).

1. ELIGIBILITY -- the sparse-N curve must measure sparsity, not catalog
   truth errors, so a star is eligible only if its full-N ftp_mb E4 verdict
   vs its role truth (Chen for science / Gaia for controls) was
   exact-or-harmonic ('exact','2f','f/2').  This excludes the 7 science +
   1 antijoin stars adjudicated 'lawful' in E4 (catalog truth wrong or too
   imprecise -- they would fail at EVERY N by construction) and the 1
   antijoin photometric-depth failure.  Eligible: 51 science, 10 antijoin.
   (All 73 stars have >= 44 epochs in both bands, so N=40/band is feasible
   for every candidate; asserted below.)

2. SCIENCE (21 stars): 7 per field (486, 686, 786).  Within each field the
   RRab/RRc split is (4,3), except the field with the most eligible RRc
   (tie: smaller field id) gets (3,4) -> global 11 RRab + 10 RRc.
   Within each (field, type) cell the quota is spread over the E4 median-r
   mag terciles (edges from e4_rates.json: bright <= 14.43 < mid <= 15.20
   < faint): first 1 seat to every non-empty tercile (descending
   availability, tie: bright->faint), remaining seats D'Hondt-style to the
   tercile maximizing avail/(assigned+1) (tie: bright->faint), capped at
   availability.  Within each (field, type, tercile) micro-cell, stars are
   drawn WITHOUT replacement by a per-cell RandomState seeded from
   crc32('E5subset|<field>|<type>|<tercile>|<E5_SEED>').

3. CONTROLS (3 anti-join): one seeded pick per field with any eligible
   antijoin control (ascending field id), then the remainder from the field
   with the most eligible left (tie: smaller id).  Field 786's only
   antijoin control was the E4 depth failure, so controls come from
   486 (x1) and 686 (x2); field 786 is still covered by its 7 science
   stars.  Per-field RandomState seeded from crc32('E5ctrl|<field>|<seed>').

Output: e5_star_subset.json -- spec + strata table + per-star records
(strata, per-band epochs, E4 full-N class for ALL 6 methods = the
saturation anchors, truth periods).
"""
import csv
import json
import os
import zlib
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.join(HERE, '..', 'e4_recovery')
OUT = os.path.join(HERE, 'e5_star_subset.json')

E5_SEED = 20260711
N_SCIENCE = 21
N_CONTROL = 3
PER_FIELD = 7
FIELDS = (486, 686, 786)
TYPES = ('RRab', 'RRc')
TERC_LABELS = ('bright', 'mid', 'faint')
OK_CLASSES = ('exact', '2f', 'f/2')
METHODS = ('gls_1band', 'ftp_1band', 'mbls_h1', 'mhls_h8', 'ce', 'ftp_mb')
N_MAX_PER_BAND = 40                       # largest E5 subsample size


def cell_rs(*parts):
    key = '|'.join(str(p) for p in parts)
    return np.random.RandomState(zlib.crc32(key.encode()) & 0xffffffff)


def dhondt_alloc(avail, q):
    """1 seat to each non-empty bin first (descending avail, tie low idx),
    then remaining seats to argmax avail/(assigned+1) (tie low idx)."""
    assigned = [0] * len(avail)
    order = sorted((b for b in range(len(avail)) if avail[b] > 0),
                   key=lambda b: (-avail[b], b))
    for b in order:
        if sum(assigned) < q:
            assigned[b] = 1
    while sum(assigned) < q:
        cand = [b for b in range(len(avail)) if assigned[b] < avail[b]]
        if not cand:
            raise SystemExit('cell availability %r < quota %d' % (avail, q))
        b = max(cand, key=lambda b: (avail[b] / (assigned[b] + 1.0), -b))
        assigned[b] += 1
    return assigned


def main():
    with open(os.path.join(E4, 'star_table.json')) as fh:
        stars = json.load(fh)['stars']
    with open(os.path.join(E4, 'e4_rates.json')) as fh:
        edges = json.load(fh)['strata_science']['mag_edges']
    anchors, mags = {}, {}
    with open(os.path.join(E4, 'e4_scores.csv')) as fh:
        for r in csv.DictReader(fh):
            anchors.setdefault(r['uid'], {})[r['method']] = r['class']
            mags[r['uid']] = float(r['r_med_mag'])

    def tercile(uid):
        m = mags[uid]
        return 0 if m <= edges[0] else (1 if m <= edges[1] else 2)

    for s in stars:                                   # N=40/band feasibility
        assert min(s['bands'][b]['n_epochs'] for b in 'gr') >= N_MAX_PER_BAND

    eligible = {s['uid']: s for s in stars
                if anchors[s['uid']]['ftp_mb'] in OK_CLASSES}
    sci = [s for s in eligible.values() if s['role'] == 'science']
    ctl = [s for s in eligible.values() if s['role'] == 'control_antijoin']
    print('eligible: %d science, %d antijoin (of 58/12)'
          % (len(sci), len(ctl)))

    # -------- science type quotas per field: (4,3) except max-RRc field
    rrc_counts = {f: sum(1 for s in sci
                         if s['field'] == f and s['type'] == 'RRc')
                  for f in FIELDS}
    rrc_field = min(FIELDS, key=lambda f: (-rrc_counts[f], f))
    quotas = {f: ({'RRab': 3, 'RRc': 4} if f == rrc_field
                  else {'RRab': 4, 'RRc': 3}) for f in FIELDS}
    print('max-eligible-RRc field = %d -> quotas %s' % (rrc_field, quotas))

    picked = []
    for f in FIELDS:
        for ty in TYPES:
            pool = sorted((s for s in sci
                           if s['field'] == f and s['type'] == ty),
                          key=lambda s: s['uid'])
            avail = [sum(1 for s in pool if tercile(s['uid']) == b)
                     for b in range(3)]
            assign = dhondt_alloc(avail, quotas[f][ty])
            for b in range(3):
                sub = [s for s in pool if tercile(s['uid']) == b]
                rs = cell_rs('E5subset', f, ty, b, E5_SEED)
                idx = rs.choice(len(sub), size=assign[b], replace=False)
                picked += [sub[i] for i in sorted(idx)]
    assert len(picked) == N_SCIENCE

    # -------- controls: 1 per field with eligible antijoin, rest from max
    ctl_pool = {f: sorted((s for s in ctl if s['field'] == f),
                          key=lambda s: s['uid']) for f in FIELDS}
    controls = []
    for f in FIELDS:
        if ctl_pool[f]:
            rs = cell_rs('E5ctrl', f, E5_SEED)
            controls.append(ctl_pool[f].pop(
                rs.randint(len(ctl_pool[f]))))
    while len(controls) < N_CONTROL:
        f = min(FIELDS, key=lambda f: (-len(ctl_pool[f]), f))
        rs = cell_rs('E5ctrl', f, E5_SEED, len(controls))
        controls.append(ctl_pool[f].pop(rs.randint(len(ctl_pool[f]))))
    assert len(controls) == N_CONTROL

    subset = picked + controls
    recs = []
    for s in subset:
        uid = s['uid']
        recs.append({
            'uid': uid, 'role': s['role'], 'type': s['type'],
            'field': s['field'], 'r_med_mag': mags[uid],
            'mag_tercile': TERC_LABELS[tercile(uid)],
            'n_epochs_g': s['bands']['g']['n_epochs'],
            'n_epochs_r': s['bands']['r']['n_epochs'],
            'primary_band': s['primary_band'],
            'chen_period': s['chen_period'], 'gaia_period': s['gaia_period'],
            'dual_truth': s['dual_truth'],
            'e4_fullN_class': anchors[uid],       # saturation anchors, all 6
        })

    strata = {
        'by_type': dict(Counter(r['type'] for r in recs)),
        'by_field': dict(Counter(r['field'] for r in recs)),
        'by_tercile': dict(Counter(r['mag_tercile'] for r in recs)),
        'by_role': dict(Counter(r['role'] for r in recs)),
        'science_field_x_type': {
            '%d_%s' % (f, ty): sum(1 for r in recs if r['role'] == 'science'
                                   and r['field'] == f and r['type'] == ty)
            for f in FIELDS for ty in TYPES},
        'science_type_x_tercile': {
            '%s_%s' % (ty, tl): sum(1 for r in recs
                                    if r['role'] == 'science'
                                    and r['type'] == ty
                                    and r['mag_tercile'] == tl)
            for ty in TYPES for tl in TERC_LABELS},
    }
    out = {
        'spec': {
            'seed': E5_SEED,
            'n_total': len(recs), 'n_science': N_SCIENCE,
            'n_control_antijoin': N_CONTROL,
            'eligibility': 'E4 full-N ftp_mb class in %r vs role truth '
                           '(Chen science / Gaia controls); excludes the 8 '
                           'lawful-adjudicated truth-error stars + 1 depth '
                           'failure so the sparse curve measures sparsity, '
                           'not catalog truth error' % (OK_CLASSES,),
            'mag_edges_r_med': list(edges),
            'mag_edges_source': 'e4_rates.json strata_science (reused '
                                'verbatim, E4 science terciles)',
            'science_rule': '7/field; RRab/RRc (4,3) except max-eligible-'
                            'RRc field %d gets (3,4); terciles D\'Hondt '
                            'within (field,type); per-cell RandomState('
                            'crc32(E5subset|field|type|tercile|seed))'
                            % rrc_field,
            'control_rule': '1/field with eligible antijoin then remainder '
                            'from largest remaining pool (786 excluded: its '
                            'only antijoin was the E4 depth failure)',
            'n_max_per_band_checked': N_MAX_PER_BAND,
        },
        'strata_table': strata,
        'stars': recs,
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=False)
    print(json.dumps(strata, indent=1))
    print('wrote %s (%d stars)' % (OUT, len(recs)))


if __name__ == '__main__':
    main()
