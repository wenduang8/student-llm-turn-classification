"""D-86 deterministic, source-reviewed allocation; never creates human labels.

The state machine persists pending candidates and cursors before a reviewer acts.
Every selected/fill/replacement record needs an explicit source-review pass.
Evaluation is locked before any development/training candidates are proposed.
Student content and manifests are written only under data/.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, html, itertools, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'data/maintenance/20261005-allocation'
OUT = ROOT/'data/coding/direct-20260924/allocation-20261005'
STRATA = ['20+', '5-19', '1-4']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rank(seed, purpose, cid): return hashlib.sha256(f'{seed}|{purpose}|{cid}'.encode()).hexdigest()
def band(n): return '1-4' if n <= 4 else '5-19' if n <= 19 else '20+'
def researcher_id():
    """The researcher's own pseudonymous student ID, kept outside the published code (data/private-ids.json)."""
    return read_json(ROOT/'data/private-ids.json')['researcher_student_id']
def csv_rows(path):
    csv.field_size_limit(100_000_000)
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def build_pool():
    invpath = ROOT/'data/coding/direct-20260924/agreement-200-inventory-20260929/conversation-inventory.jsonl'
    exppath = ROOT/'data/coding/direct-20260924/agreement-200-prep/exposure-evidence.csv'
    scpath = ROOT/'data/maintenance/20260930-structure-check/structure-check.json'
    duppath = ROOT/'data/maintenance/20260930-review-fixes/repeated-pairs-fulltext.json'
    inv=[json.loads(x) for x in invpath.read_text(encoding='utf-8').splitlines() if x.strip()]
    exposure={r['对话编号']:r['接触类别'][0] for r in csv_rows(exppath)}
    incomplete={r['对话编号'] for r in read_json(scpath)['structure'] if not r['可见捕获'].startswith('与原文一致')}
    kinds={'all turns of the shorter conversation near-identical','all turns near-identical or mostly overlapping',
           'some turns near-identical or mostly overlapping; both have other turns'}
    overlap={cid for p in read_json(duppath)['pairs'] if p['fulltext_kind'] in kinds for cid in [p['shorter'],p['longer']]}
    pool=[]; excluded=[]
    for r in inv:
        cid=r['conversation_id']; student=r['student_id']; reasons=[]
        if student==researcher_id(): reasons.append('E0 researcher submission')
        if r['language_rule_hit_interactions']: reasons.append('E1 language rule')
        if r['current_development_use']!='未列入当前材料': reasons.append('E2 current development')
        if student in {'P024','P065','P107','P344'} and r['current_development_use']=='未列入当前材料': reasons.append('E3 development student')
        if exposure.get(cid)=='一': reasons.append('E4 coding/discussion exposure')
        if exposure.get(cid)=='三': reasons.append('E5 exposure uncertain')
        if cid in incomplete: reasons.append('E6 incomplete capture')
        if cid in overlap: reasons.append('E7b pending overlap ruling')
        item={'cid':cid,'student':student,'n':r['interaction_count'],'stratum':band(r['interaction_count']),
              'assignment':r['assignment'],'normalized_path':r['source_text_location'],'source_files':r['source_files'],
              'association_groups':r['diagnostics']['association_group_ids'],'diagnostics':r['diagnostics']}
        if reasons: excluded.append({'cid':cid,'reasons':reasons})
        else: pool.append(item)
    paths=[invpath,exppath,scpath,duppath,ROOT/'data/processed/transcripts_analysis.csv',ROOT/'data/processed/conversations_master.csv',
           ROOT/'codebook_schema.json',ROOT/'docs/codebook/guide-direct-20260924.md',ROOT/'docs/decisions.md']
    return pool,excluded,{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}

def first_batch(roster, seed, cap=150):
    """Exact subset sums of whole conversations, not labels; one best hash-order tie per sum."""
    by_s={s:sorted([p for p in roster if p['stratum']==s],key=lambda p:rank(seed,'eval-first',p['cid'])) for s in STRATA}
    choices={}
    for s, rows in by_s.items():
        dp={0:()}
        for i,p in enumerate(rows):
            for count,ids in list(dp.items())[::-1]:
                new=count+p['n']; val=ids+(i,)
                if new<=cap and (new not in dp or val<dp[new]): dp[new]=val
        choices[s]=dp
    totals={s:sum(p['n'] for p in by_s[s]) for s in STRATA}; full=sum(totals.values())
    best=None
    for numbers in itertools.product(*(choices[s] for s in STRATA)):
        total=sum(numbers)
        if not 0<total<=cap or any(totals[s] and n==0 for s,n in zip(STRATA,numbers)): continue
        distance=sum((n*full-total*totals[s])**2 for s,n in zip(STRATA,numbers))
        ids=tuple(choices[s][n] for s,n in zip(STRATA,numbers))
        key=(-total,distance,ids)
        if best is None or key<best[0]: best=(key,numbers,ids)
    if best is None: raise ValueError('No first batch covering all nonempty strata under the cap')
    selected={by_s[s][i]['cid'] for s,ids in zip(STRATA,best[2]) for i in ids}
    return selected

class Engine:
    def __init__(self, config, pool, state=None):
        self.c=config; self.pool={p['cid']:p for p in pool}
        self.state=state or {'phase':'evaluation','selected':{k:[] for k in ['evaluation','development','training_initial','training_long']},
            'cursors':{},'held_students':[],'events':[],'locked':[],'completed':False}
    def event(self,action,**kw): self.state['events'].append({'seq':len(self.state['events'])+1,'action':action,**kw})
    def used(self): return set(self.state['held_students'])|{self.pool[cid]['student'] for ids in self.state['selected'].values() for cid in ids}
    def total(self,phase,s=None): return sum(self.pool[c]['n'] for c in self.state['selected'][phase] if s is None or self.pool[c]['stratum']==s)
    def next_candidate(self,phase,s,min_n,max_n,cap,stratum_cap=None):
        qkey=f'{phase}|{s or "all"}'
        queue=sorted([p for p in self.pool.values() if (s is None or p['stratum']==s) and min_n<=p['n']<=max_n],
                     key=lambda p:rank(self.c['seed'],phase,p['cid']))
        cursor=self.state['cursors'].get(qkey,0)
        while cursor<len(queue):
            p=queue[cursor]; cursor+=1; self.state['cursors'][qkey]=cursor
            reason=None
            if p['student'] in self.used(): reason='student already selected or held'
            elif self.total(phase)+p['n']>cap: reason='overall cap'
            elif stratum_cap is not None and self.total(phase,s)+p['n']>stratum_cap: reason='stratum cap'
            if reason:
                self.event('skip',phase=phase,cid=p['cid'],reason=reason); continue
            self.state['selected'][phase].append(p['cid'])
            self.event('propose',phase=phase,cid=p['cid'],queue=qkey,rank=rank(self.c['seed'],phase,p['cid']))
            return True
        return False
    def advance(self,reviews):
        for phase in self.state['locked']:
            for cid in self.state['selected'][phase]:
                if reviews.get(cid,{}).get('status')!='pass':
                    raise ValueError(f'Locked source review changed: {cid}; stop for explicit correction, do not silently redraw')
        if self.state['completed']: return []
        for phase in ['evaluation','development','training_initial','training_long']:
            if phase in self.state['locked']: continue
            self.state['phase']=phase
            unresolved=[self.pool[cid] for cid in self.state['selected'][phase]
                        if reviews.get(cid,{}).get('status') not in {'pass','hold'}]
            if unresolved:
                # Finish the saved review round before replacing any held item;
                # asynchronous reviewer timing must not change the allocation.
                return unresolved
            for cid in list(self.state['selected'][phase]):
                rev=reviews.get(cid)
                if rev and rev.get('status')=='hold':
                    self.state['selected'][phase].remove(cid)
                    self.state['held_students']=sorted(set(self.state['held_students'])|{self.pool[cid]['student']})
                    self.event('source_hold',phase=phase,cid=cid,reason=rev.get('reason',''))
            if phase=='evaluation':
                cfg=self.c['evaluation']
                for s in STRATA:
                    while self.total(phase,s)<cfg['lower_per_stratum']:
                        if not self.next_candidate(phase,s,1,10**9,cfg['maximum'],cfg['upper_per_stratum']): break
                while self.total(phase)<cfg['minimum']:
                    if not self.next_candidate(phase,'1-4',1,4,cfg['maximum'],cfg['upper_per_stratum']): break
            elif phase=='development':
                cfg=self.c[phase]
                while self.total(phase)<cfg['minimum'] or len(self.state['selected'][phase])<cfg['minimum_students']:
                    if not self.next_candidate(phase,None,1,cfg['conversation_maximum'],cfg['maximum']): break
            elif phase=='training_initial':
                cfg=self.c[phase]
                while self.total(phase)<cfg['minimum']:
                    if not self.next_candidate(phase,None,1,cfg['conversation_maximum'],cfg['maximum']): break
            else:
                cfg=self.c[phase]
                remaining=self.c['training_development_combined_maximum']-self.total('development')-self.total('training_initial')
                if not self.state['selected'][phase]:
                    self.next_candidate(phase,None,cfg['conversation_minimum'],cfg['conversation_maximum'],remaining)
            pending=[self.pool[cid] for cid in self.state['selected'][phase] if reviews.get(cid,{}).get('status')!='pass']
            if pending: return pending
            if not self.state['selected'][phase]:
                raise ValueError(f'No source-reviewed material can be selected for {phase}')
            if phase=='evaluation':
                self.state['first_batch']=sorted(first_batch([self.pool[c] for c in self.state['selected'][phase]],self.c['seed'],self.c['first_batch']['maximum']))
            self.state['locked'].append(phase)
            self.event('lock',phase=phase,interactions=self.total(phase),conversations=len(self.state['selected'][phase]))
        self.state['completed']=True
        return []

def validate_reviews(reviews,pool=None):
    lookup={p['cid']:p for p in pool} if pool is not None else None
    for cid,r in reviews.items():
        if lookup is not None and cid not in lookup: raise ValueError(f'Review outside allocation pool: {cid}')
        if r.get('status') not in {'pass','hold'}: raise ValueError(f'Bad review status {cid}')
        if not r.get('reason') or r.get('reviewer_kind')!='AI_source_review': raise ValueError(f'Missing review provenance {cid}')
        if r['status']=='pass':
            for key in ['normalized_sha256','analysis_rows_checked','source_paths','scope']:
                if not r.get(key): raise ValueError(f'Missing review field {cid}: {key}')
            for src in r['source_paths']:
                if sha(ROOT/src['path'])!=src['sha256']: raise ValueError(f'Source changed: {src["path"]}')
            if lookup is not None:
                p=lookup[cid]; normalized=ROOT/p['normalized_path']
                if r['normalized_sha256']!=sha(normalized): raise ValueError(f'Normalized source changed: {cid}')
                if type(r['analysis_rows_checked']) is not int or r['analysis_rows_checked']!=p['n']:
                    raise ValueError(f'Reviewed row count differs from inventory: {cid}')
                paths={(ROOT/src['path']).resolve() for src in r['source_paths']}
                if normalized.resolve() not in paths or len(paths)<2:
                    raise ValueError(f'Review needs normalized and original source receipts: {cid}')

def export_materials(engine,reviews):
    state=engine.state
    required={'evaluation','development','training_initial','training_long'}
    if not state['completed'] or set(state['locked'])!=required:
        raise ValueError('Cannot export an incomplete or unlocked allocation')
    for phase,ids in state['selected'].items():
        for cid in ids:
            if reviews.get(cid,{}).get('status')!='pass':
                raise ValueError(f'Cannot export unreviewed source: {cid}')
    OUT.mkdir(parents=True,exist_ok=True)
    csvsource=ROOT/'data/processed/transcripts_analysis.csv'; source=csv_rows(csvsource)
    rows_by_c=collections.defaultdict(list)
    for row in source: rows_by_c[row['Conversation ID']].append(row)
    all_cids={c for ids in state['selected'].values() for c in ids}
    summaries=[]; keysets={}; students={}
    eval_first=set(state['first_batch'])
    groups={'evaluation_first':[c for c in state['selected']['evaluation'] if c in eval_first],
            'evaluation_optional':[c for c in state['selected']['evaluation'] if c not in eval_first],
            'development':state['selected']['development'], 'training_initial':state['selected']['training_initial'],
            'training_long':state['selected']['training_long']}
    target_records=[]
    for group,cids in groups.items():
        keys=[]; ids=set(); packed=[]
        for cid in cids:
            p=engine.pool[cid]; ids.add(p['student'])
            rows=sorted(rows_by_c[cid],key=lambda r:int(r['Segment Order']))
            if len(rows)!=p['n']: raise ValueError(f'Inventory/corpus count mismatch: {cid}')
            for row in rows:
                key={'conversation_id':cid,'segment_order':int(row['Segment Order'])}
                keys.append(key)
                packed.append({'group':group,'student_id':p['student'],**key,'student_prompt':row['Student Prompt'],'ai_response':row['LLM Response'],
                               'source_file':row['Source File'],'source_review':'AI_source_review'})
            target_records.append({'group':group,**p,'source_review':reviews[cid]['reason']})
        keysets[group]=keys;students[group]=sorted(ids)
        summaries.append({'group':group,'interactions':len(keys),'conversations':len(cids),'students':len(ids),
            'strata':{s:sum(engine.pool[c]['n'] for c in cids if engine.pool[c]['stratum']==s) for s in STRATA},
            'assignments':{a:{'conversations':sum(engine.pool[c]['assignment']==a for c in cids),
                              'interactions':sum(engine.pool[c]['n'] for c in cids if engine.pool[c]['assignment']==a)}
                           for a in sorted({engine.pool[c]['assignment'] for c in cids})}})
        folder=OUT/group;folder.mkdir(exist_ok=True)
        write_json(folder/'manifest.json',{'protocol':engine.c['protocol'],'schema_version':engine.c['schema_version'],
             'phase':'selected_unlabelled','group':group,'target_count':len(keys),'conversation_count':len(cids),
             'student_count':len(ids),'targets':keys,'students':sorted(ids),'conversation_ids':cids,
             'coding_release':'Evaluation waits for stable post-calibration guide; this selection is not a coded return.'})
        (folder/'interactions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in packed),encoding='utf-8')
    student_sets={k:set(v) for k,v in students.items()}
    for a,b in itertools.combinations(student_sets,2):
        assert not student_sets[a]&student_sets[b], (a,b)
    # One conversation per student was selected across every material group.
    assert len({engine.pool[c]['student'] for c in all_cids})==len(all_cids)
    ev=sum(len(keysets[k]) for k in ['evaluation_first','evaluation_optional'])
    train_total=sum(len(keysets[k]) for k in ['development','training_initial','training_long'])
    assert ev<=engine.c['evaluation']['maximum'] and len(keysets['evaluation_first'])<=engine.c['first_batch']['maximum']
    assert train_total<=engine.c['training_development_combined_maximum']
    result={'status':'source_reviewed_selection','protocol':engine.c['protocol'],'seed':engine.c['seed'],'groups':summaries,
            'evaluation_total':ev,'development_training_total':train_total,'remaining_training_budget':engine.c['training_development_combined_maximum']-train_total,
            'held_students':state['held_students'],'checks':{'student_overlap':False,'all_source_reviewed':True,'all_targets_whole_conversations':True,'human_labels_created':False},
            'limits':['AI review of local captured sources only, not original live-chat completeness.','Outside-repository exposure remains for DW to report.',
                      'Only records actually independently double-coded can support human agreement; optional completion is not presumed.'],
            'shortfalls':{'evaluation':max(0,engine.c['evaluation']['minimum']-ev),
                          'first_batch':max(0,engine.c['first_batch']['minimum_goal']-len(keysets['evaluation_first'])),
                          'development_students':max(0,engine.c['development']['minimum_students']-len(students['development'])),
                          'development_interactions':max(0,engine.c['development']['minimum']-len(keysets['development'])),
                          'training_initial_interactions':max(0,engine.c['training_initial']['minimum']-len(keysets['training_initial'])),
                          **{f'evaluation_stratum_{s}':max(0,engine.c['evaluation']['lower_per_stratum']-engine.total('evaluation',s)) for s in STRATA}},
            'source_checks':{cid:reviews[cid] for cid in sorted(all_cids)}}
    write_json(OUT/'selection-summary.json',result)
    write_json(OUT/'conversation-roster.json',target_records)
    with (OUT/'conversation-roster.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['group','cid','student','n','stratum','assignment','normalized_path','source_files','source_review']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(target_records)
    write_json(OUT/'source-checks.json',result['source_checks'])
    render_readers(groups,rows_by_c,engine,reviews,result)
    return result

def render_readers(groups,rows_by_c,engine,reviews,result):
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#203048}h1,h2{color:#123b49}nav,a{color:#176777}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:16px;border-radius:7px}.note{padding:16px;background:#fff4cf}small{color:#5a6570}'
    titles={'evaluation_first':'评估首批｜第二编码员 最多150条','evaluation_optional':'评估后续｜第二编码员 可选补标','development':'开发材料｜DW','training_initial':'训练首批｜DW','training_long':'训练长对话｜DW'}
    def save(p,title,body):
        p.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
    review_dir=OUT/'conversations';review_dir.mkdir(exist_ok=True)
    for group,cids in groups.items():
        trs=[]
        for cid in cids:
            p=engine.pool[cid]
            trs.append(f'<tr><td><a href="../conversations/{cid}.html">{cid}</a></td><td>{p["student"]}</td><td>{p["n"]}</td><td>{p["stratum"]}</td></tr>')
            body=f'<p><a href="../{group}/index.html">返回清单</a></p><h1>{cid}</h1><p>{html.escape(titles[group])} · {p["n"]}条 · direct-20260924</p><p class="note">这是未编码的阅读材料，无AI建议或人工标签。评估编码须等待校准后的稳定手册。原文核查属于AI辅助核查。</p>'
            body+=f'<details><summary>来源核查与范围（不作为额外编码证据）</summary><p>{html.escape(reviews[cid]["reason"])}</p><p>{html.escape(str(reviews[cid]["scope"]))}</p>'
            body+=f'<p><a href="../../../../{html.escape(p["normalized_path"].removeprefix("data/"))}">规范化源文件，仅供来源核查</a>。源文件可能另含提交说明或事后使用反思；它们不是对话正文。本页及源链接不是正式编码包。</p></details>'
            if reviews[cid].get('sensitive_content_present'):
                body+='<p class="note">来源核查发现潜在身份信息。本页仅用于本地研究核查；正式向编码员发包前，须在派生副本上处理去标识并保留原件。</p>'
            for row in sorted(rows_by_c[cid],key=lambda r:int(r['Segment Order'])):
                body+=f'<article><h2>#{row["Segment Order"]}</h2><h3>Student</h3><pre>{html.escape(row["Student Prompt"])}</pre><h3>AI response</h3><pre>{html.escape(row["LLM Response"])}</pre></article>'
            save(review_dir/f'{cid}.html',cid,body)
        save(OUT/group/'index.html',titles[group],'<p><a href="../START-HERE.html">返回总览</a></p><h1>'+titles[group]+'</h1><table><tr><th>对话</th><th>学生</th><th>互动</th><th>长度层</th></tr>'+''.join(trs)+'</table>')
    trs=''.join(f'<tr><td><a href="{s["group"]}/index.html">{titles[s["group"]]}</a></td><td>{s["interactions"]}</td><td>{s["conversations"]}</td><td>{s["students"]}</td></tr>' for s in result['groups'])
    body='<h1>10月5日材料分配</h1><p>按完整对话选取 · 不同用途的学生互不重叠 · 未生成任何标签</p><p class="note">这是一份已完成AI来源核查的选样结果。评估编码须等待校准后的稳定手册；第二编码员只先安排首批，后续补标不作完成承诺。仓库外历史接触仍需DW补充。</p><table><tr><th>材料</th><th>互动数</th><th>对话数</th><th>学生数</th></tr>'+trs+'</table>'
    body+=f'<p>评估合计：{result["evaluation_total"]}条。训练与开发已安排：{result["development_training_total"]}条；200条预算内还剩{result["remaining_training_budget"]}条，留待有训练证据后决定是否补标。</p><p><a href="conversation-roster.csv">下载完整对话清单 CSV</a> · <a href="README.md">方法与核查范围</a></p><p>首批最多150条，由已锁定整批中的完整对话组成，分配过程不使用标签。仅双方实际独立完成的记录才能用于人工一致性；DW独自完成的余量应单独说明。</p>'
    save(OUT/'START-HERE.html','10月5日材料分配',body)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--advance',action='store_true');parser.add_argument('--export',action='store_true');args=parser.parse_args()
    config=read_json(TASK/'selection-config.json'); pool,excluded,hashes=build_pool()
    statepath=TASK/'selection-state.json';reviews_path=TASK/'source-reviews.json'
    reviews=read_json(reviews_path) if reviews_path.exists() else {};validate_reviews(reviews,pool)
    state=read_json(statepath) if statepath.exists() else None
    contract_sha=sha(TASK/'selection-config.json')
    if state:
        assert state['config_sha256']==contract_sha,'Contract changed after first proposal'
        assert state['input_hashes']==hashes,'Allocation inputs changed; do not redraw silently'
    else:
        write_json(TASK/'pool.json',pool);write_json(TASK/'pool-exclusions.json',excluded)
        queues={s:sorted([p['cid'] for p in pool if p['stratum']==s],key=lambda c:rank(config['seed'],'evaluation',c)) for s in STRATA}
        write_json(TASK/'evaluation-order.json',queues)
    engine=Engine(config,pool,state)
    engine.state.update(config_sha256=contract_sha,input_hashes=hashes)
    pending=engine.advance(reviews)
    write_json(statepath,engine.state);write_json(TASK/'pending-source-review.json',pending)
    if engine.state['completed']:
        result=export_materials(engine,reviews)
        print(json.dumps({k:v for k,v in result.items() if k not in ['source_checks']},ensure_ascii=False,indent=2))
    else:
        print(json.dumps({'phase':engine.state['phase'],'pending_conversations':len(pending),'pending_interactions':sum(p['n'] for p in pending),
            'current_totals':{phase:engine.total(phase) for phase in engine.state['selected']},'pending_file':str((TASK/'pending-source-review.json').relative_to(ROOT))},ensure_ascii=False))

if __name__=='__main__': main()
