#!/usr/bin/env python3
"""R1.4-R1.6 paired re-evaluation on frozen R1.2 candidate pool.
No measured emissions or calibrated flow model are claimed.
"""
import argparse, io, json, zipfile, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES=['distance_km','co2_proxy','traffic_rank','hotspot','pollutant','top10']
AGENTS=['prediction','hotspot','pollutant','efficiency','policy','top10']
BASE_W=np.array([.25,.18,.07,.20,.25,.05])

def load_csv(path,member):
    path=Path(path)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('/'+member) or n==member]
            if len(names)!=1: raise ValueError(f'{member}: expected one archive entry, got {names}')
            return pd.read_csv(io.BytesIO(z.read(names[0])))
    return pd.read_csv(path/member)

def module(path):
    spec=importlib.util.spec_from_file_location('gate_r12',str(path)); m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def rank_score(x):
    x=np.asarray(x,float); lo,hi=x.min(),x.max()
    return np.ones(len(x)) if hi-lo<1e-12 else (hi-x)/(hi-lo)

def score_matrix(f,eps):
    d,c,r,h,p,t=f.T
    shortest=d.min(); violation=np.maximum(0,100*(d/(shortest*(1+eps))-1))
    policy=np.maximum(0,1-violation/20)
    return np.column_stack([rank_score(r),rank_score(h),rank_score(p),rank_score(d),policy,rank_score(t)])

def pick(f,s,eps,weights,penalty,active):
    # A policy agent's absence removes its score, not the common physical detour constraint.
    eligible=np.flatnonzero(f[:,0] <= (1+eps)*f[:,0].min()+1e-9)
    a=np.array(active,dtype=int);w=weights[a]/weights[a].sum(); sub=s[np.ix_(eligible,a)]
    disagree=sub.std(axis=1,ddof=1) if len(a)>1 else np.zeros(len(eligible))
    utility=sub@w-penalty*disagree
    return int(eligible[np.argmax(utility)])

def output_metric(f,i,eps):
    j=int(np.argmin(f[:,0])); d,c,_,h,p,_=f[i];d0,c0,_,h0,_,_=f[j]
    return {'selected_candidate':i,'distance_km':d,'co2_proxy':c,'hotspot_cost':h,
      'co2_proxy_reduction_pct':100*(c0-c)/c0 if c0 else np.nan,
      'hotspot_reduction_pct':100*(h0-h)/h0 if h0 else np.nan,
      'distance_penalty_pct':100*(d/d0-1),'detour_compliant':int(d <= (1+eps)*d0+1e-9)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',required=True,help='R1.2 results folder or ZIP containing route_candidates.csv')
    ap.add_argument('--e5',required=True,help='E5_feedback_summary.csv')
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--epsilon',type=float,default=.20)
    ap.add_argument('--code-dir',default='.')
    args=ap.parse_args(); out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    pool=load_csv(args.results,'route_candidates.csv')
    for c in ['city','od_id','candidate']+FEATURES:
        if c not in pool: raise ValueError('Missing candidate column '+c)
    if pool.duplicated(['city','od_id','candidate']).any():raise ValueError('Duplicate candidates')
    e5=pd.read_csv(args.e5)
    # R1.4: reuse actual E5 experimental summary, preserving original units.
    r14=e5[e5.step==e5.step.max()].copy()
    r14['measurement_status']='synthetic_traffic_load_and_proxy_not_measured_CO2'
    r14.to_csv(out/'R14_feedback_metrics.csv',index=False)
    paired=[]
    for city,grp in r14.groupby('city'):
        if set(grp['mode'])=={'open','closed'}:
            o=grp[grp['mode']=='open'].iloc[0];cl=grp[grp['mode']=='closed'].iloc[0]
            paired.append({'city':city,'iteration':int(cl.step),'open_route_traffic_proxy':o.selected_route_traffic_cost,
             'closed_route_traffic_proxy':cl.selected_route_traffic_cost,
             'change_pct':100*(cl.selected_route_traffic_cost/o.selected_route_traffic_cost-1),
             'open_hotspot_cost':o.selected_route_hotspot_cost,'closed_hotspot_cost':cl.selected_route_hotspot_cost,
             'closed_relative_state_change':cl.relative_state_change})
    pd.DataFrame(paired).to_csv(out/'R14_feedback_comparison.csv',index=False)
    scenarios=[('base',BASE_W,.08)]
    for j,name in enumerate(AGENTS):
        for factor in [.5,1.5]:
            w=BASE_W.copy();w[j]*=factor;w/=w.sum()
            scenarios.append((f'{name}_x{factor:g}',w,.08))
    for pen in [0,.04,.16,.32]:scenarios.append((f'penalty_{pen:g}',BASE_W,pen))
    # Ablations are corrected: score component removed AND weights renormalized.
    ablations={'full_all_six':list(range(6)),
       'without_prediction':[1,2,3,4,5], 'without_hotspot':[0,2,3,4,5],
       'without_pollutant':[0,1,3,4,5], 'without_efficiency':[0,1,2,4,5],
       'without_policy':[0,1,2,3,5], 'without_top10':[0,1,2,3,4],
       'without_negotiation_direct':[0,1,2,3,4,5]}
    # Direct weighted score is a non-negotiating control, distinct from the full protocol.
    model_path=Path(args.code_dir)/'GATE_DT_R1_2_V2.py'
    m=module(model_path)
    sens=[]; abl=[]
    for (city,od),group in pool.groupby(['city','od_id'],sort=True):
        group=group.sort_values('candidate').reset_index(drop=True)
        f=group[FEATURES].to_numpy(float)
        if len(f)<2 or not np.isfinite(f).all():continue
        s=m.component_scores(f,args.epsilon,float(group['rho'].iloc[0]))[0]
        for label,w,pen in scenarios:
            i=pick(f,s,args.epsilon,w,pen,range(6))
            sens.append({'city':city,'od_id':od,'scenario':label,'penalty':pen,
              **{f'w_{a}':w[j] for j,a in enumerate(AGENTS)},**output_metric(f,i,args.epsilon)})
        mask=m.eligible(f,args.epsilon)
        # Same negotiation algorithm and candidate pool, with active agent set modified.
        for label,active in ablations.items():
            if label=='without_negotiation_direct':
                i=pick(f,s,args.epsilon,BASE_W,.08,active)
                rounds=0;agreed=np.nan
            else:
                i,info,_,_=m.negotiate(s,mask,active=active)
                rounds=info['rounds'];agreed=int(info['agreed'])
            abl.append({'city':city,'od_id':od,'ablation':label,'active_agents':'+'.join(AGENTS[j] for j in active),
              'rounds':rounds,'agreed':agreed,**output_metric(f,i,args.epsilon)})
    sens=pd.DataFrame(sens);abl=pd.DataFrame(abl)
    sens.to_csv(out/'R15_sensitivity_raw.csv',index=False)
    abl.to_csv(out/'R16_corrected_ablation_raw.csv',index=False)
    base=sens[sens.scenario=='base'][['city','od_id','selected_candidate']].rename(columns={'selected_candidate':'base_candidate'})
    sens=sens.merge(base,on=['city','od_id']);sens['changed_vs_base']=(sens.selected_candidate!=sens.base_candidate).astype(int)
    ss=sens.groupby(['city','scenario'],as_index=False).agg(n_od=('od_id','nunique'),route_change_pct=('changed_vs_base',lambda x:100*x.mean()),co2_proxy_reduction_pct=('co2_proxy_reduction_pct','mean'),hotspot_reduction_pct=('hotspot_reduction_pct','mean'),distance_penalty_pct=('distance_penalty_pct','mean'),detour_compliance_pct=('detour_compliant',lambda x:100*x.mean()))
    ss.to_csv(out/'R15_sensitivity_summary.csv',index=False)
    ref=abl[abl.ablation=='full_all_six'][['city','od_id','selected_candidate','co2_proxy_reduction_pct']].rename(columns={'selected_candidate':'full_candidate','co2_proxy_reduction_pct':'full_co2_reduction'})
    abl=abl.merge(ref,on=['city','od_id']);abl['changed_vs_full']=(abl.selected_candidate!=abl.full_candidate).astype(int)
    abl['delta_co2_reduction_pp']=abl.co2_proxy_reduction_pct-abl.full_co2_reduction
    abl.to_csv(out/'R16_corrected_ablation_raw.csv',index=False)
    aa=abl.groupby(['city','ablation'],as_index=False).agg(n_od=('od_id','nunique'),route_change_pct=('changed_vs_full',lambda x:100*x.mean()),delta_co2_reduction_pp=('delta_co2_reduction_pp','mean'),co2_proxy_reduction_pct=('co2_proxy_reduction_pct','mean'),hotspot_reduction_pct=('hotspot_reduction_pct','mean'),distance_penalty_pct=('distance_penalty_pct','mean'),detour_compliance_pct=('detour_compliant',lambda x:100*x.mean()),agreement_pct=('agreed',lambda x:100*x.mean()))
    aa.to_csv(out/'R16_corrected_ablation_summary.csv',index=False)
    # Three-panel LaTeX table, numeric results only, no unsupported emissions units.
    def tex(x):return str(x).replace('_',r'\_')
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{R1.4--R1.6: simulated congestion feedback, parameter sensitivity, and corrected agent ablations. Traffic-related CO$_2$ values are dimensionless proxies, not measured emissions.}',r'\label{tab:r14_r15_r16}',r'\scriptsize',r'\begin{tabular}{llrrrr}',r'\hline',r'\multicolumn{6}{l}{\textbf{(a) R1.4: Open-loop versus closed-loop simulated traffic-cost proxy (iteration 8)}}\\',r'City & Condition & Open & Closed & Change (\%) & State change\\',r'\hline']
    for r in paired:lines.append(f"{tex(r['city'])} & synthetic feedback & {r['open_route_traffic_proxy']:.2f} & {r['closed_route_traffic_proxy']:.2f} & {r['change_pct']:+.3f} & {r['closed_relative_state_change']:.6f}\\\\")
    lines.extend([r'\hline',r'\multicolumn{6}{l}{\textbf{(b) R1.5: Weight and disagreement-penalty sensitivity (city-averaged)}}\\',r'City & Parameter setting & Route changes (\%) & CO$_2$ proxy reduction (\%) & Hotspot reduction (\%) & Detour (\%)\\',r'\hline'])
    for r in ss.itertuples(index=False):lines.append(f'{tex(r.city)} & {tex(r.scenario)} & {r.route_change_pct:.1f} & {r.co2_proxy_reduction_pct:.2f} & {r.hotspot_reduction_pct:.2f} & {r.distance_penalty_pct:.2f}\\\\')
    lines.extend([r'\hline',r'\multicolumn{6}{l}{\textbf{(c) R1.6: Corrected ablations, same candidate pool and negotiation protocol}}\\',r'City & Agent configuration & Route changes (\%) & $\Delta$CO$_2$ reduction (pp) & Hotspot reduction (\%) & Detour (\%)\\',r'\hline'])
    for r in aa.itertuples(index=False):lines.append(f'{tex(r.city)} & {tex(r.ablation)} & {r.route_change_pct:.1f} & {r.delta_co2_reduction_pp:+.2f} & {r.hotspot_reduction_pct:.2f} & {r.distance_penalty_pct:.2f}\\\\')
    lines.extend([r'\hline',r'\end{tabular}',r'\end{table*}'])
    (out/'R14_R15_R16_three_panel_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    manifest={'epsilon':args.epsilon,'n_city_od':int(pool[['city','od_id']].drop_duplicates().shape[0]),'n_sensitivity_scenarios':len(scenarios),'n_ablation_conditions':len(ablations),'co2_status':'proxy only; no calibrated grams CO2','r14_source':'E5 summary from prior simulated feedback','r16_note':'full_all_six is legacy negotiation; it is not CARA_Guarded; direct control is a weighted score, not an agent-removal ablation','candidate_scores':'original R1.2 component_scores with exported rho, same frozen pool','comparison':'same frozen candidate pool and OD pairs'}
    (out/'R14_R15_R16_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Saved to',out,'\n',json.dumps(manifest,indent=2))

def guarded_active(f,s,eps,active,base):
    """Rebuild guarded anchor, regret and votes for ACTIVE agents only.
    Common detour rule and protected CO2/operational utility constraints are unchanged.
    """
    a=np.array(active,int); w=base.WEIGHTS[a].astype(float)
    w=w/w.sum()
    feasible=base.eligible(f,eps)
    ids=np.flatnonzero(feasible)
    if not len(ids): raise ValueError("No feasible candidates")
    sub=s[:,a]
    sd=sub.std(axis=1,ddof=1) if len(a)>1 else np.zeros(len(f))
    consensus=sub@w-base.DISAGREEMENT_WEIGHT*sd
    anchor=int(ids[np.argmax(consensus[ids])])
    regrets=sub[ids].max(axis=0)[None,:]-sub
    worst=regrets.max(axis=1)
    utility=base.utility_vector(f)
    guard=feasible & (f[:,1]<=f[anchor,1]+base.TOL) & (utility>=utility[anchor]-base.TOL) & (worst<=worst[anchor]+base.TOL)
    menu=np.flatnonzero(guard)
    assert anchor in menu
    weighted=regrets@w
    proposal=anchor
    agreed=False
    rounds=0
    for t,tau in enumerate(base.TAUS,1):
        rounds=t
        votes=regrets[proposal]<=tau+base.TOL
        if votes.all():
            agreed=True
            break
        if t<len(base.TAUS):
            support=(regrets[menu]<=tau+base.TOL)@w
            proposal=min(map(int,menu),key=lambda i:(-round(float(support[np.searchsorted(menu,i)]),12),worst[i],-utility[i],weighted[i],i))
    if not agreed:
        proposal=min(map(int,menu),key=lambda i:(worst[i],-utility[i],weighted[i],i))
    assert guard[proposal]
    return int(proposal),int(agreed),rounds,len(menu),anchor

def guarded_ablation(pool,model,epsilon,out):
    cases={
        'full_CARA_Guarded':tuple(range(6)),
        'without_prediction':(1,2,3,4,5),
        'without_hotspot':(0,2,3,4,5),
        'without_pollutant':(0,1,3,4,5),
        'without_efficiency':(0,1,2,4,5),
        'without_policy':(0,1,2,3,5),
        'without_top10':(0,1,2,3,4),
        'without_hotspot_and_top10':(0,2,3,4)
    }
    rows=[]
    for (city,od),g in pool.groupby(['city','od_id'],sort=True):
        g=g.sort_values('candidate').reset_index(drop=True)
        f=g[FEATURES].to_numpy(float)
        if len(f)<2 or not np.isfinite(f).all(): continue
        s=model.component_scores(f,epsilon,float(g.rho.iloc[0]))[0]
        # Full condition MUST reproduce the source's actual CARA_Guarded selector.
        full,_=model.select(f,s,epsilon,only=[model.FULL])
        reference=int(full[model.FULL][0])
        for name,active in cases.items():
            idx,agreed,rounds,menu,anchor=guarded_active(f,s,epsilon,active,model)
            if name=='full_CARA_Guarded' and idx!=reference:
                raise AssertionError(f"Guarded implementation mismatch {city} {od}: {idx} != {reference}")
            rows.append(dict(city=city,od_id=od,ablation=name,active_agents='+'.join(AGENTS[j] for j in active),
                agreed=agreed,rounds=rounds,guarded_pool_size=menu,anchor_candidate=anchor,
                changed_vs_full=int(idx!=reference),**output_metric(f,idx,epsilon)))
    raw=pd.DataFrame(rows)
    raw.to_csv(out/'R16_guarded_ablation_raw.csv',index=False)
    full=raw[raw.ablation=='full_CARA_Guarded'][['city','od_id','co2_proxy_reduction_pct']].rename(columns={'co2_proxy_reduction_pct':'full_co2_reduction_pct'})
    raw=raw.merge(full,on=['city','od_id'],validate='many_to_one')
    raw['delta_co2_reduction_pp']=raw.co2_proxy_reduction_pct-raw.full_co2_reduction_pct
    raw.to_csv(out/'R16_guarded_ablation_raw.csv',index=False)
    summ=raw.groupby(['city','ablation'],as_index=False).agg(
        n_od=('od_id','nunique'),route_change_pct=('changed_vs_full',lambda v:100*v.mean()),
        co2_proxy_reduction_pct=('co2_proxy_reduction_pct','mean'),
        delta_co2_reduction_pp=('delta_co2_reduction_pp','mean'),
        hotspot_reduction_pct=('hotspot_reduction_pct','mean'),
        distance_penalty_pct=('distance_penalty_pct','mean'),
        agreement_pct=('agreed',lambda v:100*v.mean()),
        detour_compliance_pct=('detour_compliant',lambda v:100*v.mean()),
        mean_guarded_pool=('guarded_pool_size','mean'))
    summ.to_csv(out/'R16_guarded_ablation_summary.csv',index=False)
    return raw,summ

def combined_table(out):
    a=pd.read_csv(out/'R14_feedback_comparison.csv')
    b=pd.read_csv(out/'R15_sensitivity_summary.csv')
    c=pd.read_csv(out/'R16_guarded_ablation_summary.csv')
    # Full sensitivity CSV retains 17 scenarios; concise paper table shows baseline,
    # ranges over all scenarios, and guarded agent-removal conditions.
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Paired evidence for simulated load feedback (a), preference sensitivity (b), and CARA-Guarded agent ablation (c). Traffic-related CO$_2$ quantities are proxies, not calibrated emissions.}',r'\label{tab:r14_r15_r16_v2}',r'\scriptsize',r'\begin{tabular}{llrrrr}',r'\hline',
      r'\multicolumn{6}{l}{\textbf{(a) Simulated open-loop and closed-loop traffic-related route cost, final iteration}}\\',
      r'City & Comparison & Open & Closed & Change (\%) & State change\\',r'\hline']
    for x in a.itertuples(index=False):
        lines.append(f'{x.city} & synthetic feedback & {x.open_route_traffic_proxy:.2f} & {x.closed_route_traffic_proxy:.2f} & {x.change_pct:+.3f} & {x.closed_relative_state_change:.6f}'+r'\\')
    lines.extend([r'\hline',r'\multicolumn{6}{l}{\textbf{(b) Parameter sensitivity: 17 configurations on fixed candidate pool}}\\',
        r'City & Scenario & Route changes (\%) & CO$_2$ reduction (\%) & Hotspot reduction (\%) & Detour (\%)\\',r'\hline'])
    for city,g in b.groupby('city'):
        for x in g.itertuples(index=False):
            scenario_tex=x.scenario.replace("_",r"\_")
            lines.append(f'{city} & {scenario_tex} & {x.route_change_pct:.1f} & {x.co2_proxy_reduction_pct:.2f} & {x.hotspot_reduction_pct:.2f} & {x.distance_penalty_pct:.2f}'+r'\\')
    lines.extend([r'\hline',r'\multicolumn{6}{l}{\textbf{(c) CARA-Guarded ablation: all safeguards and agent votes recomputed}}\\',
        r'City & Configuration & Route changes (\%) & $\Delta$CO$_2$ (pp) & Hotspot reduction (\%) & Detour (\%)\\',r'\hline'])
    for x in c.itertuples(index=False):
        ablation_tex=x.ablation.replace("_",r"\_")
        lines.append(f'{x.city} & {ablation_tex} & {x.route_change_pct:.1f} & {x.delta_co2_reduction_pp:+.2f} & {x.hotspot_reduction_pct:.2f} & {x.distance_penalty_pct:.2f}'+r'\\')
    lines.extend([r'\hline',r'\end{tabular}',r'\end{table*}'])
    (out/'R14_R15_R16_three_panel_table_V2.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main_v2():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',required=True)
    ap.add_argument('--e5',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--epsilon',type=float,default=.20)
    ap.add_argument('--code-dir',default='.')
    args=ap.parse_args()
    # Run existing R14/R15/R16 (legacy) pipeline unchanged for comparison.
    main()
    out=Path(args.out_dir)
    pool=load_csv(args.results,'route_candidates.csv')
    model=module(Path(args.code_dir)/'GATE_DT_R1_2_V2.py')
    raw,summ=guarded_ablation(pool,model,args.epsilon,out)
    combined_table(out)
    manifest=json.loads((out/'R14_R15_R16_manifest.json').read_text())
    manifest.update({'v2_guarded_ablation':'validated against CARA_Guarded source select() for every OD',
      'guarded_conditions':int(raw.ablation.nunique()),'guarded_pairs':int(raw[['city','od_id']].drop_duplicates().shape[0]),
      'r14_limitation':'synthetic E5 load feedback; not calibrated congestion assignment or grams CO2',
      'r15_limitation':'fixed candidate pool and epsilon=0.20; no parameter optimum is asserted',
      'r16_limitation':'guarded protection may limit route changes; results cannot be guaranteed superior'})
    (out/'R14_R15_R16_manifest_V2.json').write_text(json.dumps(manifest,indent=2))
    print('Validated CARA_Guarded against original selector for',manifest['guarded_pairs'],'city-OD cases.')
    print('Wrote R16_guarded_ablation_summary.csv and R14_R15_R16_three_panel_table_V2.tex')

if __name__=='__main__': main_v2()
