#!/usr/bin/env python3
"""
describe_corpus.py — data description for the COMP2017 student-LLM corpus.

Describes the analysis file (data/processed/transcripts_analysis.csv, after the D-53 repair and
the D-60 exclusion), including every figure of thesis Table 3.1. --corpus selects another file, for
example the baseline transcripts_all.csv. The output goes to data/processed/description/ (private,
not in git); the September snapshots that used to be written to docs/corpus/ described the corpus
before the D-53 repair and are archived (D-205).

Reads   data/processed/transcripts_analysis.csv (or --corpus)
        data/processed/transcripts_excluded_unpaired.csv
        data/processed/conversations_master.csv
        data/processed/extraction_manifest.csv
        data/Raw data from COMP2017/participant_map.csv
Writes  data/processed/description/corpus_stats.json
        data/processed/description/agg_conversation_lengths.csv
        data/processed/description/agg_student_volumes.csv
        data/processed/description/agg_turn_lengths.csv
        data/processed/description/agg_exclusion_funnel.csv

Aggregates only: no transcript text is written out. Deterministic: no sampling, no RNG.

Usage:  python analysis/describe_corpus.py [--root .] [--corpus data/processed/transcripts_analysis.csv]
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codebook_schema import is_non_english   # noqa: E402  the D-12 test, defined once
CODE = re.compile(r'(#include|int\s+main|printf\(|malloc\(|->\w|;\s*$|\bstruct\s+\w+|pthread_'
                  r'|\bvoid\s+\w+\s*\(|```|def\s+\w+\(|}\s*$)', re.M)
WORD_BINS   = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 10**9]
WORD_LABELS = ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100',
               '101-200', '201-500', '500+']


def stats(s) -> dict:
    s = pd.Series(s).astype(float)
    return dict(n=int(s.size), min=float(s.min()), max=float(s.max()),
                mean=round(float(s.mean()), 2), median=float(s.median()),
                sd=round(float(s.std()), 2), p25=float(s.quantile(.25)),
                p75=float(s.quantile(.75)), p90=float(s.quantile(.90)),
                p95=float(s.quantile(.95)))


def codey(s: str) -> bool:
    return '```' in s or len(CODE.findall(s)) >= 3


def main(root: Path, corpus: str = 'data/processed/transcripts_analysis.csv') -> int:
    proc = root / 'data' / 'processed'
    out_dir = proc / 'description'
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)

    df = rd(root / corpus)
    ex = rd(proc / 'transcripts_excluded_unpaired.csv')
    cm = rd(proc / 'conversations_master.csv')
    mf = rd(proc / 'extraction_manifest.csv')
    pm_path = root / 'data' / 'Raw data from COMP2017' / 'participant_map.csv'
    pm = rd(pm_path) if pm_path.exists() else pd.DataFrame({'pid': []})

    df['Segment Order'] = df['Segment Order'].astype(int)
    df['sw'] = df['Student Prompt'].str.split().str.len()
    df['aw'] = df['LLM Response'].str.split().str.len()
    df['asst'] = df['Conversation ID'].str.extract(r'_P([12])_', expand=False).fillna('?')
    df['route'] = df['Segmentation Method'].map(
        lambda m: 'Deterministic' if m == 'deterministic'
        else 'LLM-segmented' if m.startswith('llm')
        else 'MHTML' if m.startswith('mhtml') else 'Rule / manual')
    df['cjk'] = df.apply(lambda r: is_non_english(r['Student Prompt'], r['LLM Response']), axis=1)
    df['scode'] = df['Student Prompt'].map(codey)
    df['acode'] = df['LLM Response'].map(codey)

    pc = df.groupby('Conversation ID').size()
    ps = df.groupby('Student ID').size()
    cps = df.groupby('Student ID')['Conversation ID'].nunique()

    o: dict = {}
    o['corpus'] = dict(
        file=corpus, interactions=len(df), conversations=int(pc.size), students=int(ps.size),
        with_student_text=int(df['Student Prompt'].str.strip().ne('').sum()),
        excluded_rows=len(ex), conversation_records=len(cm),
        manifest_entries=len(mf), manifest_pids=int(mf['pid'].nunique()),
        mapped_participants=int(pm['pid'].nunique()) if len(pm) else None)

    for name, series, unit in (('per_conversation', pc, 'conversations'),
                               ('per_student', ps, 'students'),
                               ('conversations_per_student', cps, 'students')):
        o[name] = stats(series)
        o[name]['hist'] = {int(k): int(v) for k, v in sorted(collections.Counter(series.values).items())}
    o['per_conversation'].update(ge10=int((pc >= 10).sum()), ge20=int((pc >= 20).sum()),
                                 eq1=int((pc == 1).sum()), le2=int((pc <= 2).sum()),
                                 rows_ge10=int(df['Conversation ID'].isin(pc[pc >= 10].index).sum()))
    o['per_student'].update(ge20=int((ps >= 20).sum()), le5=int((ps <= 5).sum()),
                            rows_ge20=int(df['Student ID'].isin(ps[ps >= 20].index).sum()))
    srt = np.sort(ps.values)[::-1]
    o['per_student']['top10pct_share'] = round(float(srt[:max(1, len(srt)//10)].sum()/srt.sum()), 3)
    o['per_student']['top25pct_share'] = round(float(srt[:max(1, len(srt)//4)].sum()/srt.sum()), 3)
    o['cum_rows_top_students'] = {int(n): int(srt[:n].sum()) for n in (10, 20, 30, 50, 75, 100)}

    for name, col in (('student_words', 'sw'), ('llm_words', 'aw')):
        o[name] = stats(df[col])
        o[name]['bins'] = {l: int(v) for l, v in zip(WORD_LABELS, pd.cut(
            df[col], bins=WORD_BINS, labels=WORD_LABELS, right=True)
            .value_counts().reindex(WORD_LABELS).fillna(0))}
        o[name].update(eq1=int((df[col] == 1).sum()), le3=int((df[col] <= 3).sum()),
                       le10=int((df[col] <= 10).sum()), ge1000=int((df[col] >= 1000).sum()))
    w = df['sw'].sort_values()
    o['student_words']['mean_trimmed_1pct'] = round(float(w.iloc[:int(len(w)*.99)].mean()), 1)

    o['code'] = dict(student=int(df['scode'].sum()), llm=int(df['acode'].sum()),
                     student_fenced=int(df['Student Prompt'].str.contains('```', regex=False).sum()),
                     llm_fenced=int(df['LLM Response'].str.contains('```', regex=False).sum()))
    o['language'] = dict(
        rows_cjk=int(df['cjk'].sum()),
        pct=round(float(df['cjk'].mean()*100), 1),
        students=int(df[df['cjk']]['Student ID'].nunique()),
        conversations=int(df[df['cjk']]['Conversation ID'].nunique()),
        students_majority=int((df.groupby('Student ID')['cjk'].mean() >= .5).sum()),
        expected_in_400=round(400*float(df['cjk'].mean()), 1))

    o['by_assignment'] = {a: dict(interactions=len(g), conversations=g['Conversation ID'].nunique(),
                                  students=g['Student ID'].nunique())
                          for a, g in df.groupby('asst')}
    o['by_route'] = {r: dict(interactions=len(g), conversations=g['Conversation ID'].nunique(),
                             students=g['Student ID'].nunique())
                     for r, g in df.groupby('route')}

    pool = pc[(pc >= 8) & (pc <= 30)].index
    o['freeze_pool'] = dict(conversations=len(pool),
                            interactions=int(df['Conversation ID'].isin(pool).sum()),
                            students=int(df[df['Conversation ID'].isin(pool)]['Student ID'].nunique()),
                            conversations_with_cjk=int(df[df['Conversation ID'].isin(pool)]
                                                       .groupby('Conversation ID')['cjk'].any().sum()))

    mid, late = [], []
    for cid, g in df.groupby('Conversation ID'):
        so = sorted(g['Segment Order'])
        if so[0] != 1: late.append(cid)
        if len(so) != so[-1] - so[0] + 1: mid.append(cid)
    o['sequence_gaps'] = dict(mid_sequence=len(mid), starts_late=len(late),
                              mid_ids=sorted(mid), late_ids=sorted(late))

    (out_dir / 'corpus_stats.json').write_text(json.dumps(o, indent=2), encoding='utf-8')
    pd.DataFrame(sorted(o['per_conversation']['hist'].items()),
                 columns=['interactions_in_conversation', 'conversations']
                 ).to_csv(out_dir / 'agg_conversation_lengths.csv', index=False)
    pd.DataFrame(sorted(o['per_student']['hist'].items()),
                 columns=['interactions_contributed', 'students']
                 ).to_csv(out_dir / 'agg_student_volumes.csv', index=False)
    pd.DataFrame({'words': WORD_LABELS,
                  'student_turns': [o['student_words']['bins'][l] for l in WORD_LABELS],
                  'llm_turns': [o['llm_words']['bins'][l] for l in WORD_LABELS]}
                 ).to_csv(out_dir / 'agg_turn_lengths.csv', index=False)

    # ---- coding-purpose exclusion funnel (Maher, 1 Sep 2026) -------------------
    # Two stages must be documented separately: data-science cleaning (already done
    # upstream) and coding-purpose exclusion, below. Each criterion is reported with
    # its cost in conversations AND in interactions - the two differ sharply here.
    import re as _re
    # The researcher took COMP2017 in S1 2025 and his own submission is in the corpus. It stays
    # in the corpus and in every descriptive figure, but is excluded from the coded set: he is
    # the primary coder and has access to his own intent, which he cannot have for any other
    # student. Applied FIRST so its cost is reported on its own terms and does not depend on the
    # turn-count floor happening to remove the same conversations.
    # The ID itself is kept out of the published code, in a private file beside the data
    # (data/private-ids.json, key "researcher_student_id").
    RESEARCHER_ID = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'private-ids.json')
                               .read_text(encoding='utf-8'))['researcher_student_id']
    df['_cjk'] = df['cjk']          # same test, computed once above
    pc = df.groupby('Conversation ID').size()
    steps, keep = [], set(pc.index)

    def _step(label, new_keep):
        sub = df[df['Conversation ID'].isin(new_keep)]
        steps.append(dict(step=label, conversations=len(new_keep), interactions=len(sub),
                          students=int(sub['Student ID'].nunique())))
        return new_keep

    keep = _step('0. all complete interactions', keep)
    mine = set(df[df['Student ID'].str.upper().str.startswith(RESEARCHER_ID)]['Conversation ID'])
    keep = _step(f"1. drop the researcher's own submissions ({RESEARCHER_ID})", keep - mine)
    keep = _step('2. drop conversations of 3 interactions or fewer', {x for x in keep if pc[x] > 3})
    cjk_convs = set(df[df['_cjk']]['Conversation ID'].unique())
    keep = _step('3. drop conversations containing Chinese', keep - cjk_convs)
    CAP = 100
    keep = _step(f'4. cap at {CAP} interactions', {x for x in keep if pc[x] <= CAP})

    fn = pd.DataFrame(steps)
    fn['conversations_removed'] = fn['conversations'].shift(1) - fn['conversations']
    fn['interactions_removed'] = fn['interactions'].shift(1) - fn['interactions']
    fn.to_csv(out_dir / 'agg_exclusion_funnel.csv', index=False)
    o['exclusion_funnel'] = steps
    o['exclusion_funnel_note'] = (
        'Stage 3 removes 10 per cent of conversations but a larger share of interactions, '
        'because conversations containing Chinese are longer than average. Report both. '
        "Stage 1 currently costs nothing beyond what stage 2 would remove anyway - all of the "
        'researcher\'s conversations are short - but it is applied first and stated separately '
        'so the exclusion does not silently depend on the turn-count floor. If the floor is '
        'lowered, or if the opening prompt of short conversations is coded on a separate track, '
        'stage 1 is what keeps the researcher out of the coded set.')

    c = o['corpus']
    print(f"{c['file']}: {c['interactions']:,} interactions · {c['conversations']:,} conversations · "
          f"{c['students']} students")
    print("\nTable 3.1 (composition of the analysis file):")
    print(f"  records {c['interactions']:,}; containing student text {c['with_student_text']:,}; "
          f"conversations {c['conversations']:,}; students {c['students']}")
    for a, g in sorted(o['by_assignment'].items()):
        print(f"  Assignment {a}: {g['interactions']:,} records / {g['conversations']:,} conversations")
    print(f"median per conversation {o['per_conversation']['median']:.0f} · "
          f"per student {o['per_student']['median']:.0f}")
    print(f"conversations >=10: {o['per_conversation']['ge10']} · "
          f"students >=20: {o['per_student']['ge20']}")
    print(f"non-English rows: {o['language']['rows_cjk']:,} "
          f"({o['language']['pct']}%) across {o['language']['students']} students")
    print("\nexclusion funnel (coding purpose):")
    for st in o['exclusion_funnel']:
        print(f"  {st['step']:48} {st['conversations']:5} conv  {st['interactions']:6,} int  "
              f"{st['students']:4} students")
    print(f"-> {out_dir}")
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='repository root (default: cwd)')
    ap.add_argument('--corpus', default='data/processed/transcripts_analysis.csv',
                    help='corpus file relative to the root (default: the analysis file)')
    a = ap.parse_args()
    sys.exit(main(Path(a.root).resolve(), a.corpus))
