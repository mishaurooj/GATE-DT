#!/usr/bin/env python3
"""R1.3 independent historical-replay, simulated-feedback, and proxy-graph audit.
Uses original GATE_DT_R1_2_V2.py algorithms without altering R1.2 outputs.
Feedback is a scenario model, NOT measured real-world traffic dynamics.
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import rankdata

p=argparse.ArgumentParser()
p.add_argument('--data-dir',type=Path,required=True)
p.add_argument('--out-dir',type=Path,required=True)
p.add_argument('--code-dir',type=Path,default=Path(__file__).parent)
p.add_argument('--candidate-csv',type=Path,default=None)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--od-per-city',type=int,default=10)
p.add_argument('--replay-days',type=int,default=30)
p.add_argument('--feedback-steps',type=int,default=8)
p.add_argument('--demand',type=float,default=.02,help='Assigned traffic per route as fraction of mean nonzero cell load')
p.add_argument('--decay',type=float,default=.25,help='Feedback relaxation per iteration')
p.add_argument('--epsilon',type=float,default=.20)
a=p.parse_args()
if not (0<a.decay<=1 and a.demand>=0 and a.feedback_steps>=1 and a.replay_days>=2 and a.od_per_city>=1):p.error('Invalid parameters')
sys.path.insert(0,str(a.code_dir))
try:import GATE_DT_R1_2_V2 as core
except ImportError:
 import importlib.util
 matches=list(a.code_dir.glob('GATE_DT_R1_2_V2*.py'))
 if not matches:raise
 spec=importlib.util.spec_from_file_location('GATE_DT_R1_2_V2',matches[0]);core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
a.out_dir.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(a.seed)
replay=[];feedback=[];valid=[];network=[];odrows=[]
provided=pd.read_csv(a.candidate_csv) if a.candidate_csv else None
for city in core.CITIES:
 g,ts=core.load_city(a.data_dir,city)
 graph=core.build_graph(g)
 arr=g[['lat','lon']].to_numpy()
 comps=list(nx.connected_components(graph)); original_n=len(g)
 lengths=np.array([d['distance'] for _,_,d in graph.edges(data=True)])
 # Geometry validity is limited to proximity edges, not road geometry.
 network.append(dict(city=city,grid_nodes=original_n,retained_nodes=graph.number_of_nodes(),excluded_nodes=original_n-graph.number_of_nodes(),retained_fraction=graph.number_of_nodes()/original_n,edges=graph.number_of_edges(),connected_components_retained=len(comps),isolated_retained=sum(d==0 for _,d in graph.degree()),self_loops=nx.number_of_selfloops(graph),edge_length_min_km=float(lengths.min()),edge_length_mean_km=float(lengths.mean()),edge_length_max_km=float(lengths.max()),invalid_length_edges=int(np.sum((lengths<=0)|(lengths>1.75+1e-8))),road_geometry_verified=False))
 if provided is not None:
  for row in provided[provided.city==city].itertuples(index=False):
   path=[int(x) for x in str(row.path).split(';')]
   exists=all(x in graph for x in path)
   continuous=exists and all(graph.has_edge(x,y) for x,y in zip(path,path[1:]))
   distance=sum(graph[x][y]['distance'] for x,y in zip(path,path[1:])) if continuous else np.nan
   valid.append(dict(city=city,od_id=int(row.od_id),candidate=int(row.candidate),nodes_exist=bool(exists),continuous=bool(continuous),simple_path=len(set(path))==len(path),endpoint_match=path[0]==int(row.origin) and path[-1]==int(row.destination),recomputed_km=distance,reported_km=float(row.distance_km),distance_absolute_error_km=abs(distance-float(row.distance_km)) if continuous else np.nan))
 # Reuse original supervised predictions, with test dates only, in chronological order.
 _,preds,_=core.forecast_city(city,ts,a.seed)
 days=pd.DataFrame([x for x in preds if x['model']=='PredictionAgentBest']).sort_values('date').tail(a.replay_days)
 if len(days)<2:raise ValueError('Not enough test days')
 nodes=np.array(list(graph.nodes));ods=[]
 for _ in range(10000):
  o,d=map(int,rng.choice(nodes,2,replace=False));u,v=graph.nodes[o],graph.nodes[d]
  if core.haversine_km(u['lat'],u['lon'],v['lat'],v['lon'])<3 or (o,d) in ods:continue
  ods.append((o,d))
  if len(ods)>=a.od_per_city:break
 if len(ods)<a.od_per_city:raise ValueError('Insufficient OD pairs')
 base=np.array([graph.nodes[i]['traffic'] for i in range(len(g))],float)
 mean_daily=float(ts.traffic.dropna().mean())
 positive=base[base>0];mean_cell=float(positive.mean()) if len(positive) else 1.
 # Fixed candidate menus per OD, computed once using historical average. Each method uses same menu.
 menus=[]
 for oid,(o,d) in enumerate(ods):
  paths=core.route_candidates(graph,o,d,25,1.0)
  assert all(all(graph.has_edge(x,y) for x,y in zip(path,path[1:])) for path in paths)
  menus.append(paths);odrows.append(dict(city=city,od_id=oid,origin=o,destination=d,candidate_count=len(paths)))
 prev={('closed',i):None for i in range(len(ods))}
 for day in days.itertuples(index=False):
  rho=max(0,float(day.predicted)/mean_daily) if mean_daily>0 else 1.
  actual=max(0,float(day.actual)/mean_daily) if mean_daily>0 else 1.
  replay.append(dict(city=city,date=day.date,observed=float(day.actual),predicted=float(day.predicted),prediction_error=float(day.predicted-day.actual),abs_prediction_error=abs(float(day.predicted-day.actual)),forecast_multiplier=rho,observed_multiplier=actual,absolute_multiplier_error=abs(rho-actual)))
  # Paired static/open-loop vs simulated closed-loop scenarios, reset at each day.
  for mode in ('open','closed'):
   state=base*actual
   for step in range(a.feedback_steps):
    trank=rankdata(state,method='average')/len(state)
    hs=.7*trank+.2*g.rn.to_numpy()+.1*g.rp.to_numpy()
    top=(hs>=np.quantile(hs,.9)).astype(int)
    added=np.zeros(len(g));route_changes=0;total_cost=0;hotspot_cost=0
    for oid,paths in enumerate(menus):
     feat=core.route_features(graph,paths,traffic=state,rt=trank,hotspot=hs,pollutant=g.pollutant.to_numpy(),top10=top)
     scores,*_=core.component_scores(feat,a.epsilon,rho)
     choices,_=core.select(feat,scores,a.epsilon,only=[core.FULL])
     idx=choices[core.FULL][0];chosen=paths[idx]
     if mode=='closed' and prev.get((mode,oid)) is not None and prev[(mode,oid)]!=idx:route_changes+=1
     if mode=='closed':prev[(mode,oid)]=idx
     total_cost+=float(feat[idx,1]);hotspot_cost+=float(feat[idx,3])
     if mode=='closed':
      for node in chosen:added[node]+=a.demand*mean_cell/max(1,len(chosen))
    if mode=='closed':
     target=base*actual+added
     next_state=(1-a.decay)*state+a.decay*target
    else:next_state=state.copy()
    relative_change=float(np.linalg.norm(next_state-state)/max(np.linalg.norm(state),1e-12))
    feedback.append(dict(city=city,date=day.date,mode=mode,step=step+1,od_count=len(ods),traffic_sum=float(state.sum()),traffic_peak=float(state.max()),hotspot_top_decile_sum=float(state[hs>=np.quantile(hs,.9)].sum()),selected_route_traffic_cost=total_cost/len(ods),selected_route_hotspot_cost=hotspot_cost/len(ods),route_switches=route_changes,relative_state_change=relative_change,assigned_load=float(added.sum())))
    state=next_state
pd.DataFrame(replay).to_csv(a.out_dir/'E4_historical_replay.csv',index=False)
pd.DataFrame(feedback).to_csv(a.out_dir/'E5_feedback_iterations.csv',index=False)
pd.DataFrame(network).to_csv(a.out_dir/'road_network_summary.csv',index=False)
pd.DataFrame(valid).to_csv(a.out_dir/'candidate_route_validation.csv',index=False)
pd.DataFrame(odrows).to_csv(a.out_dir/'replay_od_pairs.csv',index=False)
r=pd.DataFrame(replay);f=pd.DataFrame(feedback);v=pd.DataFrame(valid)
summary=r.groupby('city').agg(replay_days=('date','size'),MAE=('abs_prediction_error','mean'),mean_absolute_multiplier_error=('absolute_multiplier_error','mean')).reset_index()
summary.to_csv(a.out_dir/'E4_replay_summary.csv',index=False)
fb=f.groupby(['city','mode','step'],as_index=False).agg(traffic_sum=('traffic_sum','mean'),traffic_peak=('traffic_peak','mean'),hotspot_top_decile_sum=('hotspot_top_decile_sum','mean'),selected_route_traffic_cost=('selected_route_traffic_cost','mean'),selected_route_hotspot_cost=('selected_route_hotspot_cost','mean'),route_switches=('route_switches','mean'),relative_state_change=('relative_state_change','mean'))
fb.to_csv(a.out_dir/'E5_feedback_summary.csv',index=False)
checks=dict(replay_chronological=bool(r.groupby('city').date.apply(lambda x:x.is_monotonic_increasing).all()),all_candidate_paths_continuous=bool(v.continuous.all()) if len(v) else None,all_candidate_endpoints_match=bool(v.endpoint_match.all()) if len(v) else None,all_candidate_distance_reproduced=bool((v.distance_absolute_error_km<1e-6).all()) if len(v) else None,road_geometry_verified=False)
manifest=dict(seed=a.seed,od_per_city=a.od_per_city,replay_days_requested=a.replay_days,feedback_steps=a.feedback_steps,demand=a.demand,decay=a.decay,epsilon=a.epsilon,model='synthetic load-allocation feedback; no observed link flows or calibrated dynamics',checks=checks)
(a.out_dir/'E4_E5_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Done:',a.out_dir,'\n',summary.to_string(index=False),'\nChecks:',checks)
