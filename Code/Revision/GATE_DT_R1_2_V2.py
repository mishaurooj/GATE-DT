#!/usr/bin/env python3
"""CARA R1.2 v2: safeguarded negotiation and matched, reproducible evaluation."""
from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

CITIES = ("Delhi", "Mumbai", "Chennai")
ALIAS = {"Delhi": "MC1", "Mumbai": "MC2", "Chennai": "MC3"}
AGENT_NAMES = ("prediction", "hotspot", "pollutant", "efficiency", "policy", "top10")
WEIGHTS = np.array([.25, .18, .07, .20, .25, .05], dtype=float)
assert abs(WEIGHTS.sum() - 1.0) < 1e-12
DISAGREEMENT_WEIGHT = .08
EPSILONS = (.05, .10, .15, .20)
PRIMARY_EPS = .20
BASELINES = ("ShortestPath", "CO2_Dijkstra", "AStar_EcoRouting", "Weighted_EcoRouting",
             "RandomForest_EcoRoute", "GradientBoost_EcoRoute", "XGBoost_EcoRoute",
             "StaticDT_Routing", "PredictiveDT_Routing", "SingleAgent_Routing",
             "MultiAgent_NoNegotiation")
ABLATIONS = ("CARA_Full", "CARA_without_PredictionAgent", "CARA_without_HotspotAgent",
             "CARA_without_PollutantAgent", "CARA_without_PolicyAgent",
             "CARA_without_NegotiationAgent", "SingleAgent_Routing", "StaticDT_Routing")
MODEL_NAMES = ("LinearRegression", "Ridge", "RandomForest", "GradientBoosting", "ExtraTrees")
LOG = logging.getLogger("GATE-DT")


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


def haversine_km(lat1, lon1, lat2, lon2):
    a = (np.sin(np.radians(lat2 - lat1) / 2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(np.radians(lon2 - lon1) / 2)**2)
    return 2 * 6371.0088 * np.arctan2(np.sqrt(np.clip(a, 0, 1)),
                                      np.sqrt(np.clip(1 - a, 0, 1)))


def percentile_rank(x):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return rankdata(x, method="average") / len(x)


def invmm(x):
    z = np.asarray(x, dtype=float)
    low, high = np.min(z), np.max(z)
    return np.ones_like(z) if high - low < 1.e-12 else 1 - (z-low)/(high-low)


def safe_pct(numer, denom):
    return float(numer / denom * 100) if abs(denom) > 1.e-12 else np.nan


def discover_city_files(directory, city):
    files = list(directory.glob("*.csv"))
    grids = [p for p in files if city.lower() in p.stem.lower() and "griddedmapdata" in p.stem.lower()]
    times = [p for p in files if city.lower() in p.stem.lower() and "timeseriesdataset" in p.stem.lower()]
    if len(grids) != 1 or len(times) != 1:
        raise FileNotFoundError(f"Expected one grid and one time-series CSV for {city} in {directory}; "
                                f"found {len(grids)} grid, {len(times)} time-series")
    return grids[0], times[0]


def load_city(directory, city):
    grid_file, ts_file = discover_city_files(directory, city)
    g, ts = pd.read_csv(grid_file), pd.read_csv(ts_file)
    required_grid = ["lat", "lon", "traffic", "traffic_NOx", "traffic_pm25"]
    required_ts = ["date", "traffic"]
    for label, frame, required in [("grid", g, required_grid), ("timeseries", ts, required_ts)]:
        missing = sorted(set(required) - set(frame.columns))
        if missing: raise ValueError(f"{city} {label} missing columns: {missing}")
    g = g.copy()
    g["lat"] = pd.to_numeric(g.lat, errors="coerce")
    g["lon"] = pd.to_numeric(g.lon, errors="coerce")
    g = g.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    for col in ["traffic", "traffic_NOx", "traffic_pm25", "total_co2", "total_pollutant"]:
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0).clip(lower=0) if col in g else 0.0
    g["rt"] = percentile_rank(g["traffic"])
    g["rn"] = percentile_rank(g["traffic_NOx"])
    g["rp"] = percentile_rank(g["traffic_pm25"])
    g["hotspot"] = .70*g.rt + .20*g.rn + .10*g.rp
    g["pollutant"] = .55*g.rn + .45*g.rp
    g["top10"] = (g.hotspot >= g.hotspot.quantile(.9)).astype(int)
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["traffic", "traffic_NOx", "traffic_pm25", "total", "power",
                "residential_scope1", "residential_scope2"]:
        ts[col] = pd.to_numeric(ts[col], errors="coerce") if col in ts else np.nan
    LOG.info("%s: %s grid cells; %s days with traffic", city, len(g), ts.traffic.notna().sum())
    return g, ts


def dataset_summary(city, g, ts, n_routes):
    vals = np.sort(g.traffic.to_numpy())[::-1]
    sum_t = vals.sum()
    present = ts.dropna(subset=["traffic"])
    return {"city": city, "paper_city": ALIAS[city], "grid_cells": len(g),
            "active_cells": int((g.traffic > 0).sum()), "traffic_sum": float(sum_t),
            "mean_daily_traffic_CO2_proxy": float(present.traffic.mean()),
            "top5_share_pct": safe_pct(vals[:max(1, int(.05*len(vals)))].sum(), sum_t),
            "top10_share_pct": safe_pct(vals[:max(1, int(.10*len(vals)))].sum(), sum_t),
            "observation_days": len(present), "OD_pairs_requested": n_routes,
            "start": str(present.date.min().date()), "end": str(present.date.max().date())}


def prediction_features(ts):
    df = ts.dropna(subset=["traffic"]).sort_values("date").copy()
    if len(df) < 70: raise ValueError("Need at least 70 valid daily traffic observations for forecasting")
    for col in ["traffic", "traffic_NOx", "traffic_pm25", "total", "power", "residential_scope1", "residential_scope2"]:
        df[col] = df[col].ffill().fillna(0.0)
    df["dayofweek"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    df["dayofyear"] = df.date.dt.dayofyear
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    df["dow_sin"] = np.sin(2*np.pi*df.dayofweek/7)
    df["dow_cos"] = np.cos(2*np.pi*df.dayofweek/7)
    df["doy_sin"] = np.sin(2*np.pi*df.dayofyear/366)
    df["doy_cos"] = np.cos(2*np.pi*df.dayofyear/366)
    for col in ["traffic", "traffic_NOx", "traffic_pm25", "total", "power"]:
        for lag in [1,2,3,7,14,21]: df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for w in [3,7,14,21]:
        df[f"traffic_roll{w}"] = df.traffic.shift(1).rolling(w).mean()
        df[f"traffic_std{w}"] = df.traffic.shift(1).rolling(w).std()
    df["traffic_trend7"] = df.traffic.shift(1) - df.traffic.shift(7)
    df["target_next_traffic"] = df.traffic.shift(-1)
    df = df.dropna(subset=["target_next_traffic", "traffic_lag1", "traffic_lag7", "traffic_roll7"])
    features = [c for c in df.select_dtypes(include="number").columns if c != "target_next_traffic"]
    df[features] = df[features].ffill().fillna(0.0)
    return df, features


def smape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = .5*(np.abs(a)+np.abs(b))
    return float(100*np.mean(np.divide(np.abs(a-b), den, out=np.zeros_like(a, dtype=float), where=den>1e-12)))


def pred_metrics(city, name, y, yp, split):
    return dict(city=city, paper_city=ALIAS[city], model=name, split=split,
                RMSE=float(np.sqrt(mean_squared_error(y,yp))),
                MAE=float(mean_absolute_error(y,yp)), sMAPE_pct=smape(y,yp),
                R2=float(r2_score(y,yp)))


def forecast_city(city, ts, seed):
    frame, features = prediction_features(ts)
    n = len(frame)
    a, b = int(.60*n), int(.80*n)
    train, val, test = frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]
    scaler = StandardScaler().fit(train[features].to_numpy())
    xtr, xv, xte = (scaler.transform(d[features].to_numpy()) for d in (train,val,test))
    ytr, yv, yte = (d.target_next_traffic.to_numpy() for d in (train,val,test))
    models = {"Ridge": Ridge(alpha=1.0), "LinearRegression": LinearRegression(),
              "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
              "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
              "GradientBoosting": GradientBoostingRegressor(random_state=seed)}
    metrics, predictions = [], []
    best_rmse, best_name, best_model = float("inf"), None, None
    base = {"Persistence": test.traffic.to_numpy(), "MovingAverage3": test.traffic_roll3.to_numpy(),
            "MovingAverage7": test.traffic_roll7.to_numpy(), "SeasonalNaive7": test.traffic_lag7.to_numpy(),
            "NaiveBlend": .5*test.traffic.to_numpy()+.5*test.traffic_roll7.to_numpy()}
    for name, yp in base.items():
        metrics.append(pred_metrics(city, name, yte, yp, "test"))
    for name, model in models.items():
        model.fit(xtr, ytr)
        val_rmse = np.sqrt(mean_squared_error(yv, model.predict(xv)))
        test_pred = model.predict(xte)
        metrics.append(pred_metrics(city, name, yte, test_pred, "test"))
        for date, actual, predicted in zip(test.date, yte, test_pred):
            predictions.append(dict(city=city, date=date.date().isoformat(), model=name,
                                    actual=float(actual), predicted=float(predicted)))
        if val_rmse < best_rmse:
            best_rmse, best_name, best_model = val_rmse, name, model
    for dt, yt, yp in zip(test.date, yte, best_model.predict(xte)):
        predictions.append(dict(city=city,date=dt.date().isoformat(),model="PredictionAgentBest",
                                actual=float(yt),predicted=float(yp)))
    return metrics, predictions, dict(city=city, selected_model=best_name,
                                      validation_RMSE=float(best_rmse), n_train=len(train),
                                      n_validation=len(val), n_test=len(test), n_features=len(features))


def build_graph(g, neighbor_k=9, max_link_km=1.75):
    arr = g[["lat","lon"]].to_numpy(dtype=float)
    graph = nx.Graph()
    for i in range(len(g)):
        r = g.iloc[i]
        graph.add_node(i, lat=float(r.lat), lon=float(r.lon), traffic=float(r.traffic),
                       rt=float(r.rt), rp=float(r.rp), rn=float(r.rn),
                       hotspot=float(r.hotspot), pollutant=float(r.pollutant), top10=int(r.top10))
    neighbors = NearestNeighbors(n_neighbors=min(len(g), neighbor_k)).fit(arr).kneighbors(arr, return_distance=False)
    for i, idxs in enumerate(neighbors):
        for j in idxs:
            if i == j: continue
            d = float(haversine_km(arr[i,0], arr[i,1], arr[j,0], arr[j,1]))
            if 0 < d <= max_link_km: graph.add_edge(i,int(j),distance=d)
    if not graph.number_of_edges(): raise ValueError("Spatial graph has no edges; check max_link_km")
    comp = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(comp).copy()
    LOG.info("Graph: %d nodes, %d edges", graph.number_of_nodes(),graph.number_of_edges())
    return graph


def weighted_edge(graph, mode, pred_mult):
    def weight(u,v,data):
        node = graph.nodes[v]
        d, rt, hs, rp, pol = data["distance"], node["rt"], node["hotspot"], node["rp"], node["pollutant"]
        if mode == "shortest": return d
        if mode == "co2": return .03*d + rt
        if mode == "hotspot": return .30*d + .70*hs
        if mode == "weighted": return .35*d + .45*rt + .15*hs + .05*rp # Eq. eco_edge
        if mode == "prediction": return .25*d + .75*pred_mult*rt
        if mode == "pollutant": return .35*d + .30*rt + .35*pol
        raise ValueError(mode)
    return weight


def route_candidates(graph, origin, destination, K, pred_mult):
    # A shared candidate set, with a strict total bound K (paper requirement).
    paths, seen = [], set()
    def add(path):
        if len(paths) < K and len(path) > 1 and tuple(path) not in seen:
            paths.append(tuple(path));seen.add(tuple(path))
    add(nx.shortest_path(graph,origin,destination,weight="distance"))
    for mode in ("co2","hotspot","weighted","prediction","pollutant"):
        if len(paths)>=K:break
        try: add(nx.shortest_path(graph,origin,destination,weight=weighted_edge(graph,mode,pred_mult)))
        except nx.NetworkXNoPath: pass
    if len(paths)<K:
        try:
            generator=nx.shortest_simple_paths(graph,origin,destination,weight="distance")
            for path in generator:
                add(path)
                if len(paths)>=K:break
        except nx.NetworkXNoPath: pass
    return paths


def diversified_candidates(graph,origin,destination,K,pred_mult):
    # Nested original candidate pool for K>25, plus deterministic objective-directed paths.
    base_k=min(25,max(2,K//2)) if K<=25 else 25
    paths=route_candidates(graph,origin,destination,base_k,pred_mult);seen=set(paths)
    def add(path):
        path=tuple(path)
        if len(paths)<K and len(path)>1 and path not in seen:paths.append(path);seen.add(path)
    scale=max(1e-12,float(np.mean([v['traffic'] for _,v in graph.nodes(data=True)])))
    # Fixed mixtures, not calibrated to the known OD results. Symmetric endpoint averages.
    mixtures=[(a,1-a,0,0) for a in (.10,.25,.40,.55,.70,.85)]
    mixtures += [(a,0,1-a,0) for a in (.20,.50,.80)] + [(a,0,0,1-a) for a in (.20,.50,.80)]
    mixtures += [(.25,.25,.25,.25),(.40,.40,.10,.10),(.40,.10,.25,.25),(.15,.55,.15,.15)]
    for wd,wc,wh,wp in mixtures:
        if len(paths)>=K:break
        def weight(u,v,edge):
            a,b=graph.nodes[u],graph.nodes[v]
            return wd*edge['distance']+wc*(a['traffic']+b['traffic'])/(2*scale)+wh*(a['hotspot']+b['hotspot'])/2+wp*(a['pollutant']+b['pollutant'])/2
        add(nx.shortest_path(graph,origin,destination,weight=weight))
    # Fill remaining slots with distance alternatives if objective mixtures duplicate paths.
    if len(paths)<K:
        for path in nx.shortest_simple_paths(graph,origin,destination,weight='distance'):
            add(path)
            if len(paths)>=K:break
    return paths


def route_features(graph, paths, traffic=None, rt=None, hotspot=None, pollutant=None, top10=None):
    # For stability tests, recalculate the scores from a perturbed traffic state.
    nodes=graph.nodes()
    if traffic is None:
        return np.array([[sum(graph[a][b]["distance"] for a,b in zip(p[:-1],p[1:])),
                          sum(nodes[i]["traffic"] for i in p), sum(nodes[i]["rt"] for i in p),
                          sum(nodes[i]["hotspot"] for i in p), sum(nodes[i]["pollutant"] for i in p),
                          sum(nodes[i]["top10"] for i in p)] for p in paths],dtype=float)
    # Re-index arrays by node ID; component was a subgraph of 0..N-1.
    return np.array([[sum(graph[a][b]["distance"] for a,b in zip(p[:-1],p[1:])),
                      sum(traffic[i] for i in p),sum(rt[i] for i in p),
                      sum(hotspot[i] for i in p),sum(pollutant[i] for i in p),
                      sum(top10[i] for i in p)] for p in paths],dtype=float)


def component_scores(feats, eps, pred_mult):
    # columns: distance, traffic-related CO2 proxy, traffic rank, hotspot, pollutant, top10 count.
    base = feats[np.argmin(feats[:,0])]
    delta = (feats[:,0]-(1+eps)*base[0])/base[0]*100
    violation = np.maximum(0,delta)
    policy = np.where(violation<=1.e-12,1.0,np.maximum(0,1.0-violation/20.0))
    scores = np.column_stack([invmm(pred_mult*feats[:,2]),invmm(feats[:,3]),invmm(feats[:,4]),
                              invmm(feats[:,0]),policy,invmm(feats[:,5])])
    disagreement = scores.std(axis=1,ddof=1)
    consensus = scores @ WEIGHTS
    cara_utility = consensus - DISAGREEMENT_WEIGHT*disagreement
    return scores,violation,consensus,disagreement,cara_utility


def baseline_choice(name, feats, scores):
    d,c,rt,hs,pol,hot = feats.T
    if name=="ShortestPath": return int(np.argmin(d))
    if name=="CO2_Dijkstra": return int(np.argmin(rt))
    if name=="AStar_EcoRouting": return int(np.argmin(.35*d+.65*rt))
    if name=="Weighted_EcoRouting": return int(np.argmin(.35*d+.45*rt+.15*hs+.05*pol))
    if name in ("RandomForest_EcoRoute","GradientBoost_EcoRoute","XGBoost_EcoRoute"):
        return int(np.argmin(.20*d+.80*rt))  # Original code: same routing selector for 3 labeled baselines
    if name=="StaticDT_Routing": return int(np.argmin(.50*d+.50*hs))
    if name=="PredictiveDT_Routing": return int(np.argmin(.30*d+.70*rt))
    if name=="SingleAgent_Routing": return int(np.argmin(.10*d+.55*rt+.25*hs+.10*pol))
    if name=="MultiAgent_NoNegotiation":return int(np.argmax(.25*scores[:,0]+.25*scores[:,1]+.20*scores[:,2]+.30*scores[:,3]))
    raise KeyError(name)


TAUS=(.10,.25,.40,.55,.70) # fixed before evaluation
TOL=1e-9
FULL='CARA_Guarded'
OLD='CARA_Negotiated'
STRICT='CARA_ParetoSafe'
DIRECT='Guarded_Direct'
STRICT_DIRECT='ParetoSafe_Direct'
MATCH='Legacy_Feasible'
METHODS=(FULL,OLD,STRICT,DIRECT,STRICT_DIRECT,'Guarded_OneRound','Guarded_FixedAspiration',MATCH,'Weighted_Feasible','Legacy_Soft','OneRound','FixedAspiration','NoPolicy','WithoutPrediction','WithoutHotspot','WithoutPollutant','WithoutEfficiency')
OTHER=('ShortestPath','CO2_Dijkstra','AStar_EcoRouting','Weighted_EcoRouting','Legacy_ML_Heuristic','StaticDT_Routing','PredictiveDT_Routing','SingleAgent_Routing','MultiAgent_NoNegotiation')
METRICS=('co2_reduction_pct','distance_penalty_pct','hotspot_reduction_pct','pollutant_reduction_pct','top10_reduction_pct','policy_violation_pp','policy_compliant_pct','operational_utility','score_agreement','weighted_regret','max_regret','initial_support_pct','decision_ms','total_ms','worst_agent_score','mean_agent_score','jain_satisfaction','regret_p90','fixed_support_025_pct','pareto_efficient_pct','utility_gap','carbon_gap_pct','guarded_pool_pct','strict_pool_pct','guarded_improvement_available_pct','strict_improvement_available_pct')

def eligible(f,eps):
    return (f[:,0]-f[:,0].min())/f[:,0].min()*100 <= eps*100+TOL

@dataclass
class PreferenceAgent:
    name: str
    candidate_ids: np.ndarray
    scores: np.ndarray

    def preferences(self):
        return dict(favorite=int(self.candidate_ids[self.scores.argmax()]),
                    candidate_ids=self.candidate_ids.tolist(),scores=self.scores.tolist())

    def respond(self,candidate,threshold):
        local_regret=self.scores.max()-self.scores
        i=int(np.flatnonzero(self.candidate_ids==candidate)[0])
        acceptable=self.candidate_ids[local_regret<=threshold+TOL]
        return dict(candidate=int(candidate),regret=float(local_regret[i]),
                    favorite=int(self.candidate_ids[self.scores.argmax()]),
                    acceptable_ids=acceptable.tolist(),accept=bool(candidate in acceptable))


def negotiate(scores,mask,taus=TAUS,active=None,trace=False):
    active=np.arange(6) if active is None else np.asarray(active)
    ids=np.flatnonzero(mask); s=scores[np.ix_(ids,active)]
    w=WEIGHTS[active]/WEIGHTS[active].sum(); regret=s.max(axis=0)-s
    u=s@w-.08*s.std(axis=1,ddof=1)
    agents=[AGENT_NAMES[a] for a in active]; participants=[PreferenceAgent(a,ids,s[:,j]) for j,a in enumerate(agents)]
    def best(key):return min(range(len(ids)),key=key)
    proposal=best(lambda i:(-u[i],ids[i])); initial=int(ids[proposal]); seen=set()
    events=[]; rounds=[]; count=0; agreed=False
    def send(t,src,dst,kind,**payload):
        nonlocal count
        count+=1
        if trace:events.append(dict(round=t,sender=src,recipient=dst,kind=kind,**payload))
    for agent in participants:send(0,agent.name,'mediator','PREFERENCES',**agent.preferences())
    if 'policy' in agents:send(0,'policy','mediator','FEASIBILITY',eligible=ids.tolist())
    for t,tau in enumerate(taus,1):
        seen.add(proposal); replies=[]
        for agent in participants:
            send(t,'mediator',agent.name,'PROPOSE',candidate=int(ids[proposal]),threshold=tau)
            reply=agent.respond(int(ids[proposal]),tau);replies.append(reply)
            send(t,agent.name,'mediator','ACCEPT' if reply['accept'] else 'REJECT_COUNTERPROPOSE',**reply)
        votes=np.array([r['accept'] for r in replies],dtype=bool)
        accepts=np.column_stack([np.isin(ids,r['acceptable_ids']) for r in replies])
        rounds.append(dict(round=t,threshold=tau,candidate=int(ids[proposal]),accepting=int(votes.sum()),active_agents=len(active),support_pct=100*votes.mean(),weighted_support_pct=100*(votes@w),max_regret=float(regret[proposal].max()),weighted_regret=float(regret[proposal]@w),agent_votes=json.dumps(dict(zip(agents,map(bool,votes))))))
        if votes.all():agreed=True;break
        if t<len(taus):
            # Counterproposal selection responds to the returned acceptable sets.
            support=accepts@w
            proposal=best(lambda i:(-round(float(support[i]),12),i in seen,float(regret[i].max()),float(regret[i]@w),-u[i],ids[i]))
    if not agreed:proposal=best(lambda i:(float(regret[i].max()),float(regret[i]@w),-u[i],ids[i]))
    selected=int(ids[proposal]); tau=taus[len(rounds)-1]
    for a in agents:send(len(rounds),'mediator',a,'COMMIT' if agreed else 'FALLBACK',candidate=selected)
    info=dict(initial_candidate=initial,selected_candidate=selected,rounds=len(rounds),agreed=int(agreed),fallback=int(not agreed),changed=int(initial!=selected),final_threshold=tau,initial_support_pct=rounds[0]['support_pct'],final_support_pct=100*(regret[proposal]<=tau+TOL).mean(),fixed_initial_support_pct=100*(regret[proposal]<=taus[0]+TOL).mean(),logical_messages=count,proposal_changes=sum(a['candidate']!=b['candidate'] for a,b in zip(rounds,rounds[1:])))
    return selected,info,rounds,events

def utility_vector(f):
    d,c,_,h,_,_=f.T; i=int(d.argmin())
    # Normalized route cost from the static grid; no target-day observation enters selection.
    cr=100*(c[i]-c)/c[i] if c[i]>0 else np.zeros(len(f))
    hr=100*(h[i]-h)/h[i] if h[i]>0 else np.zeros(len(f))
    return cr-200*(d-d[i])/d[i]+.5*hr

def safeguards(f,s,eps):
    mask=eligible(f,eps); ids=np.flatnonzero(mask)
    u=s@WEIGHTS-.08*s.std(axis=1,ddof=1)
    anchor=int(ids[np.argmax(u[ids])]); r=s[mask].max(axis=0)-s
    worst=r.max(axis=1); wr=r@WEIGHTS; agr=1-s.std(axis=1,ddof=1); util=utility_vector(f)
    # Predefined protected outcomes: carbon proxy, operational utility, maximum regret.
    guard=mask&(f[:,1]<=f[anchor,1]+TOL)&(util>=util[anchor]-TOL)&(worst<=worst[anchor]+TOL)
    costs=f[:,[0,1,3,4,5]]
    strict=guard&np.all(costs<=costs[anchor]+TOL,axis=1)&(wr<=wr[anchor]+TOL)&(agr>=agr[anchor]-TOL)
    assert guard[anchor] and strict[anchor]
    return mask,anchor,guard,strict,r,util

def guarded_protocol(f,s,eps,menu,anchor,r,util,trace=False,taus=TAUS):
    # Regret references remain the entire policy-feasible set, not the restricted menu.
    ids=np.flatnonzero(menu); w=WEIGHTS; worst=r.max(axis=1); wr=r@w
    def best(candidates):
        return min(map(int,candidates),key=lambda i:(worst[i],-util[i],wr[i],i))
    proposal=anchor; initial=anchor; events=[]; rounds=[]; count=0; agreed=False; seen=set()
    def send(t,src,dst,kind,**payload):
        nonlocal count
        count+=1
        if trace:events.append(dict(round=t,sender=src,recipient=dst,kind=kind,**payload))
    for j,a in enumerate(AGENT_NAMES):
        send(0,a,'mediator','PREFERENCES',candidate_ids=ids.tolist(),scores=s[ids,j].tolist(),reference_score=float(s[anchor,j]+r[anchor,j]))
    send(0,'policy','mediator','FEASIBILITY_AND_GUARD',eligible=ids.tolist(),anchor=anchor)
    for t,tau in enumerate(taus,1):
        seen.add(proposal); accepts=[]; votes=[]
        for j,a in enumerate(AGENT_NAMES):
            send(t,'mediator',a,'PROPOSE',candidate=proposal,threshold=tau)
            accept_ids=ids[r[ids,j]<=tau+TOL]; vote=bool(r[proposal,j]<=tau+TOL)
            accepts.append(accept_ids);votes.append(vote)
            send(t,a,'mediator','ACCEPT' if vote else 'REJECT_COUNTERPROPOSE',candidate=proposal,regret=float(r[proposal,j]),acceptable_ids=accept_ids.tolist(),accept=vote)
        votes=np.array(votes,dtype=bool); support=np.column_stack([np.isin(ids,x) for x in accepts])@w
        rounds.append(dict(round=t,threshold=tau,candidate=proposal,accepting=int(votes.sum()),active_agents=6,support_pct=100*votes.mean(),weighted_support_pct=100*(votes@w),max_regret=float(worst[proposal]),weighted_regret=float(wr[proposal]),agent_votes=json.dumps(dict(zip(AGENT_NAMES,map(bool,votes))))))
        if votes.all():agreed=True;break
        if t<len(taus):
            # Reply-derived support is primary; minimax regret and utility resolve ties.
            proposal=min(map(int,ids),key=lambda i:(-round(float(support[np.searchsorted(ids,i)]),12),worst[i],-util[i],wr[i],i))
    if not agreed:proposal=best(ids)
    tau=taus[len(rounds)-1]
    for a in AGENT_NAMES:send(len(rounds),'mediator',a,'COMMIT' if agreed else 'FALLBACK',candidate=proposal)
    info=dict(initial_candidate=initial,selected_candidate=proposal,rounds=len(rounds),agreed=int(agreed),fallback=int(not agreed),changed=int(initial!=proposal),final_threshold=tau,initial_support_pct=rounds[0]['support_pct'],final_support_pct=100*(r[proposal]<=tau+TOL).mean(),fixed_initial_support_pct=100*(r[proposal]<=taus[0]+TOL).mean(),logical_messages=count,proposal_changes=sum(a['candidate']!=b['candidate'] for a,b in zip(rounds,rounds[1:])),menu_size=len(ids),reference_candidate=anchor)
    return proposal,info,rounds,events

def select(f,s,eps,trace=False,only=None):
    prep_start=time.perf_counter();mask,anchor,guard,strict,r,util=safeguards(f,s,eps);prep_ms=1000*(time.perf_counter()-prep_start);ids=np.flatnonzero(mask);out={};protocols={}
    def simple(name,fn):
        if only is not None and name not in only:return
        t=time.perf_counter();idx=int(fn());out[name]=(idx,1000*(time.perf_counter()-t))
    def proto(name,m=mask,taus=TAUS,active=None):
        if only is not None and name not in only:return
        t=time.perf_counter();idx,info,rr,e=negotiate(s,m,taus,active);elapsed=1000*(time.perf_counter()-t)
        if trace and name==OLD:
            idx2,info,rr,e=negotiate(s,m,taus,active,True);assert idx==idx2
        out[name]=(idx,elapsed);protocols[name]=(info,rr,e)
    def guarded(name,menu,taus=TAUS):
        if only is not None and name not in only:return
        t=time.perf_counter();idx,info,rr,e=guarded_protocol(f,s,eps,menu,anchor,r,util,False,taus);elapsed=1000*(time.perf_counter()-t)
        if trace:
            idx2,info,rr,e=guarded_protocol(f,s,eps,menu,anchor,r,util,True,taus);assert idx==idx2
        out[name]=(idx,elapsed+prep_ms);protocols[name]=(info,rr,e)
    guarded(FULL,guard);guarded(STRICT,strict)
    guarded('Guarded_OneRound',guard,TAUS[:1]);guarded('Guarded_FixedAspiration',guard,(TAUS[0],)*len(TAUS))
    for name,menu in ((DIRECT,guard),(STRICT_DIRECT,strict)):
        simple(name,lambda m=menu:min(map(int,np.flatnonzero(m)),key=lambda i:(r[i].max(),-util[i],r[i]@WEIGHTS,i)))
        if name in out:out[name]=(out[name][0],out[name][1]+prep_ms)
    proto(OLD)
    simple(MATCH,lambda:ids[np.argmax((s@WEIGHTS-.08*s.std(axis=1,ddof=1))[ids])])
    simple('Weighted_Feasible',lambda:ids[np.argmax((s@WEIGHTS)[ids])])
    simple('Legacy_Soft',lambda:np.argmax(s@WEIGHTS-.08*s.std(axis=1,ddof=1)))
    proto('OneRound',taus=TAUS[:1]);proto('FixedAspiration',taus=(TAUS[0],)*len(TAUS));proto('NoPolicy',m=np.ones(len(f),bool),active=[0,1,2,3,5])
    for name,removed in [('WithoutPrediction',[0]),('WithoutHotspot',[1,5]),('WithoutPollutant',[2]),('WithoutEfficiency',[3])]:proto(name,active=[j for j in range(6) if j not in removed])
    for name in OTHER:simple(name,lambda n=name:baseline_choice('RandomForest_EcoRoute' if n=='Legacy_ML_Heuristic' else n,f,s))
    return out,protocols

def metrics(f,s,eps,idx):
    d,c,_,h,p,b=f[idx];d0,c0,_,h0,p0,b0=f[f[:,0].argmin()]
    red=lambda x,y:safe_pct(x-y,x)
    cr,dp,hr=red(c0,c),safe_pct(d-d0,d0),red(h0,h);v=max(0,dp-100*eps)
    mask,anchor,guard,strict,reg,util=safeguards(f,s,eps);r=np.maximum(0,reg[idx]); z=s[idx];costs=f[:,[0,1,3,4,5]]
    dominates=np.all(costs[mask]<=costs[idx]+TOL,axis=1)&np.any(costs[mask]<costs[idx]-TOL,axis=1)
    return dict(distance_km=d,co2_proxy=c,hotspot_cost=h,pollutant_cost=p,top10_cells=b,co2_reduction_pct=cr,distance_penalty_pct=dp,hotspot_reduction_pct=hr,pollutant_reduction_pct=red(p0,p),top10_reduction_pct=red(b0,b),policy_violation_pp=v,policy_compliant_pct=100*int(v<=TOL),operational_utility=cr-2*dp+.5*hr-8*v,score_agreement=1-z.std(ddof=1),weighted_regret=float(r@WEIGHTS),max_regret=float(r.max()),initial_support_pct=100*(r<=TAUS[0]+TOL).mean(),worst_agent_score=float(z.min()),mean_agent_score=float(z.mean()),jain_satisfaction=float(z.sum()**2/(len(z)*np.sum(z*z))) if np.any(z) else np.nan,regret_p90=float(np.quantile(r,.9)),fixed_support_025_pct=100*(r<=.25+TOL).mean(),pareto_efficient_pct=100*int(mask[idx] and not dominates.any()),utility_gap=float(util[mask].max()-util[idx]) if mask[idx] else np.nan,carbon_gap_pct=safe_pct(c-f[mask,1].min(),c0) if mask[idx] else np.nan,guarded_pool_pct=100*guard.sum()/mask.sum(),strict_pool_pct=100*strict.sum()/mask.sum(),guarded_improvement_available_pct=100*int(np.any(guard&(np.arange(len(f))!=anchor)&((util>util[anchor]+TOL)|(f[:,1]<f[anchor,1]-TOL)|(reg.max(axis=1)<reg[anchor].max()-TOL)))),strict_improvement_available_pct=100*int(np.any(strict&np.any(costs<costs[anchor]-TOL,axis=1))))

def stability(g,graph,paths,eps,rho,actual,choices,rng,n):
    if not n:return []
    names=(FULL,OLD,STRICT,DIRECT,STRICT_DIRECT,MATCH,'Weighted_Feasible','Legacy_Soft','OneRound','FixedAspiration');seq={m:[choices[m][0]] for m in names}
    edges=[{tuple(sorted((a,b))) for a,b in zip(p,p[1:])} for p in paths]
    traffic=g.traffic.to_numpy();rn=g.rn.to_numpy();rp=g.rp.to_numpy();outcomes={m:[] for m in names}
    for _ in range(n):
        tr=np.maximum(0,traffic*rng.normal(1,.05,len(g)));rt=percentile_rank(tr);hs=.7*rt+.2*rn+.1*rp;pol=.55*rn+.45*rp;top=(hs>=np.quantile(hs,.9)).astype(int)
        f=route_features(graph,paths,tr,rt,hs,pol,top);s,*_=component_scores(f,eps,rho);ch,_=select(f,s,eps,only=names)
        for m in names:
            idx=ch[m][0];seq[m].append(idx);outcomes[m].append(metrics(f,s,eps,idx))
    rows=[]
    for m,a in seq.items():
        df=pd.DataFrame(outcomes[m]);u=df.operational_utility.dropna();c=df.co2_reduction_pct.dropna()
        rows.append(dict(method=m,perturbations=n,retention_pct=100*np.mean(np.array(a[:-1])==a[1:]),switching_pct=100*np.mean(np.array(a[:-1])!=a[1:]),anchor_retention_pct=100*np.mean(np.array(a[1:])==a[0]),edge_overlap_pct=100*np.mean([len(edges[a[0]]&edges[i])/len(edges[a[0]]|edges[i]) for i in a[1:]]),shock_mean_utility=u.mean(),shock_p10_utility=u.quantile(.1),shock_mean_carbon_reduction=c.mean(),shock_p10_carbon_reduction=c.quantile(.1),shock_max_regret=df.max_regret.mean(),shock_compliance_pct=df.policy_compliant_pct.mean()))
    return rows

PRIMARY_METRICS=('co2_reduction_pct','operational_utility','max_regret')
LOWER={'distance_penalty_pct','policy_violation_pp','weighted_regret','max_regret','regret_p90','decision_ms','total_ms','utility_gap','carbon_gap_pct','switching_pct','shock_max_regret','fallback','rounds','logical_messages'}
ROUTE_TESTS=tuple(m for m in METRICS if not m.endswith('pool_pct') and not m.endswith('available_pct'))
STABILITY_TESTS=('retention_pct','switching_pct','anchor_retention_pct','edge_overlap_pct','shock_mean_utility','shock_p10_utility','shock_mean_carbon_reduction','shock_p10_carbon_reduction','shock_max_regret','shock_compliance_pct')
PROTOCOL_TESTS=('agreed','fallback','rounds','fixed_initial_support_pct','logical_messages')

def paired(raw,seed,B,stab=None,neg=None):
    rng=np.random.default_rng(seed+929);rows=[]
    def compare(frame,metrics_set,comparisons,kind):
        for city,df in frame[np.isclose(frame.epsilon,PRIMARY_EPS)].groupby('city'):
            for metric in metrics_set:
                p=df.pivot(index='od_id',columns='method',values=metric)
                for proposed,comp in comparisons:
                    if proposed not in p or comp not in p:continue
                    z=(p[proposed]-p[comp]).dropna().to_numpy();n=len(z)
                    if not n:continue
                    obs=z.mean();means=[];exceed=0
                    for i in range(0,B,500):
                        k=min(500,B-i);means.extend(z[rng.integers(0,n,(k,n))].mean(axis=1));exceed+=np.count_nonzero(abs((rng.choice([-1,1],(k,n))*z).mean(axis=1))>=abs(obs)-1e-12)
                    lo,hi=np.quantile(means,[.025,.975]);direction=-1 if metric in LOWER else 1;oriented=z*direction
                    family='primary_9' if kind=='routing' and proposed==FULL and comp==MATCH and metric in PRIMARY_METRICS else 'secondary_'+kind
                    rows.append(dict(city=city,proposed=proposed,comparator=comp,metric=metric,kind=kind,family=family,direction='lower' if direction<0 else 'higher',n=n,delta=obs,ci_low=lo,ci_high=hi,p=(exceed+1)/(B+1),draws=B,win_pct=100*np.mean(oriented>1e-8),tie_pct=100*np.mean(abs(oriented)<=1e-8),loss_pct=100*np.mean(oriented<-1e-8)))
    compare(raw,ROUTE_TESTS,[(FULL,m) for m in (MATCH,OLD,DIRECT,'Weighted_Feasible','Guarded_OneRound','Guarded_FixedAspiration')]+[(STRICT,MATCH),(STRICT,STRICT_DIRECT)],'routing')
    if stab is not None and len(stab):compare(stab,STABILITY_TESTS,[(FULL,MATCH),(FULL,OLD),(FULL,DIRECT),(STRICT,MATCH)],'stability')
    if neg is not None and len(neg):compare(neg,PROTOCOL_TESTS,[(FULL,OLD),(FULL,'Guarded_OneRound'),(FULL,'Guarded_FixedAspiration')],'protocol')
    r=pd.DataFrame(rows);r['p_holm']=1.
    for _,sub in r.groupby('family'):
        order=sub.sort_values('p').index;vals=r.loc[order,'p'].to_numpy();r.loc[order,'p_holm']=np.minimum(1,np.maximum.accumulate(vals*(len(order)-np.arange(len(order)))))
    oriented=r.delta*np.where(r.direction=='lower',-1,1)
    r['outcome']=np.where(abs(oriented)<=1e-8,'tie',np.where(oriented>0,'improved','worse'))
    r['significant_holm']=r.p_holm<.05
    return r

def joint_outcomes(raw):
    # Strictly separate wins from ties on the same OD sample; undefined percentages stay excluded.
    metrics_set=('co2_reduction_pct','distance_penalty_pct','hotspot_reduction_pct','pollutant_reduction_pct','top10_reduction_pct','operational_utility','policy_compliant_pct','weighted_regret','max_regret','score_agreement')
    records=[]
    for (city,eps),df in raw.groupby(['city','epsilon']):
        for method in (FULL,OLD,STRICT,DIRECT,STRICT_DIRECT):
            a=df[df.method==method].set_index('od_id');b=df[df.method==MATCH].set_index('od_id');z=a[list(metrics_set)]-b[list(metrics_set)]; z=z.dropna();n=len(z)
            if not n:continue
            for m in metrics_set:
                if m in LOWER:z[m]*=-1
            win=z>1e-8;loss=z<-1e-8
            records.append(dict(city=city,epsilon=eps,method=method,comparator=MATCH,n_complete_OD=n,all_noninferior_pct=100*(~loss.any(axis=1)).mean(),pareto_win_pct=100*((~loss.any(axis=1))&win.any(axis=1)).mean(),all_tie_pct=100*((~loss.any(axis=1))&(~win.any(axis=1))).mean(),tradeoff_pct=100*(loss.any(axis=1)&win.any(axis=1)).mean(),all_nonbetter_with_loss_pct=100*(loss.any(axis=1)&(~win.any(axis=1))).mean(),route_changed_pct=100*(a.selected_candidate!=b.selected_candidate).mean()))
    return pd.DataFrame(records)

def write_tex(out,summary,protocol,stats,stab,joint):
    def fmt(v):return '--' if pd.isna(v) else f'{v:.3f}'
    def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%')
    def table(caption,label,heads,rows):
        return '\n'.join([r'\begin{table*}[!t]',r'\centering',r'\begingroup\color{blue}\scriptsize',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{ll'+'r'*(len(heads)-2)+'}',r'\hline',' & '.join(heads)+r' \\',r'\hline',*[' & '.join(map(str,row))+r' \\' for row in rows],r'\hline',r'\end{tabular}}',r'\endgroup',r'\end{table*}',''])
    chunks=['% Generated from actual experiments. Requires xcolor and graphicx.']
    main=summary[np.isclose(summary.epsilon,PRIMARY_EPS)&summary.method.isin([FULL,OLD,STRICT,MATCH,DIRECT,STRICT_DIRECT])]
    routing_cols=['co2_reduction_pct','distance_penalty_pct','hotspot_reduction_pct','pollutant_reduction_pct','top10_reduction_pct','policy_violation_pp','policy_compliant_pct','operational_utility']
    agreement_cols=['score_agreement','weighted_regret','max_regret','initial_support_pct','decision_ms','total_ms','worst_agent_score','jain_satisfaction','regret_p90','fixed_support_025_pct','pareto_efficient_pct','utility_gap']
    for city,df in main.groupby('city'):
        for suffix,cols,heads in [('routes',routing_cols,[r'CO$_2$ (\%)',r'Detour (\%)',r'Hotspot (\%)',r'Pollutant (\%)',r'Top10 (\%)','Violation (pp)',r'Feasible (\%)','Utility']),('agreement',agreement_cols,['Score agreement','Weighted regret','Max regret',r'Support (\%)','Decision (ms)','Total (ms)','Worst score','Jain index','P90 regret',r'Support at 0.25 (\%)',r'Pareto efficient (\%)','Utility gap'])]:
            chunks.append(table('Matched outcomes, '+ALIAS[city]+r' ($\epsilon=0.20$).','tab:r12_'+suffix+'_'+city.lower(),['City','Method',*heads],[[ALIAS[city],esc(r.method),*[fmt(r[c]) for c in cols]] for _,r in df.iterrows()]))
    chunks.append(table('Full CARA negotiation behavior across detour thresholds.','tab:r12_protocol',['City',r'$\epsilon$',r'Agreement (\%)',r'Fallback (\%)','Rounds',r'Changed (\%)',r'Initial support (\%)',r'Final support (\%)','Messages'],[[ALIAS[r.city],fmt(r.epsilon),fmt(100*r.agreed),fmt(100*r.fallback),fmt(r.rounds),fmt(100*r.changed),fmt(r.initial_support_pct),fmt(r.final_support_pct),fmt(r.logical_messages)] for _,r in protocol[protocol.method==FULL].iterrows()]))
    chunks.append(table('Paired changes: negotiated minus matched feasible legacy CARA.','tab:r12_paired',['City','Metric','N','Difference',r'95\% low',r'95\% high','$p$',r'$p_{Holm}$'],[[ALIAS[r.city],esc(r.metric),r.n,fmt(r.delta),fmt(r.ci_low),fmt(r.ci_high),f'{r.p:.5f}',f'{r.p_holm:.5f}'] for _,r in stats[(stats.comparator==MATCH)&(stats.proposed==FULL)&(stats.family=='primary_9')].iterrows()]))
    if len(stab):chunks.append(table('Route retention under independent grid traffic perturbations.','tab:r12_retention',['City','Method',r'Retention (\%)',r'Switching (\%)',r'Anchor (\%)'],[[ALIAS[r.city],esc(r.method),fmt(r.retention_pct),fmt(r.switching_pct),fmt(r.anchor_retention_pct)] for _,r in stab.iterrows()]))
    chunks.append(table('Joint metric comparison against legacy feasible CARA (complete cases).','tab:r12_v2_joint',['City','Method','N',r'Noninferior (\%)',r'Pareto win (\%)',r'All tie (\%)',r'Tradeoff (\%)',r'Route changed (\%)'],[[ALIAS[r.city],esc(r.method),r.n_complete_OD,fmt(r.all_noninferior_pct),fmt(r.pareto_win_pct),fmt(r.all_tie_pct),fmt(r.tradeoff_pct),fmt(r.route_changed_pct)] for _,r in joint[np.isclose(joint.epsilon,PRIMARY_EPS)].iterrows()]))
    (out/'R1_2_V2_paper_tables_generated.tex').write_text('\n'.join(chunks),encoding='utf-8')

def main():
    import hashlib,platform
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path(r'E:\other\GATE-DT-main\Dataset'));ap.add_argument('--out-dir',type=Path,default=Path(r'E:\other\GATE-DT-main\Results_R1_2_V2'))
    ap.add_argument('--candidate-mode',choices=['original','diversified'],default='original');ap.add_argument('--exclude-seed',type=int,default=None,help='Exclude prior OD pairs sampled with this seed');ap.add_argument('--exclude-count',type=int,default=100);ap.add_argument('--n-routes',type=int,default=100);ap.add_argument('--k-paths',type=int,default=25);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--draws',type=int,default=10000);ap.add_argument('--perturbations',type=int,default=20);ap.add_argument('--quick-check',action='store_true');ap.add_argument('--cities',nargs='+',choices=CITIES,default=list(CITIES));ap.add_argument('--skip-figures',action='store_true');a=ap.parse_args();out=a.out_dir
    if a.quick_check:a.n_routes=min(3,a.n_routes);a.k_paths=min(8,a.k_paths);a.draws=min(100,a.draws);a.perturbations=min(2,a.perturbations);out=out.with_name(out.name+'_SMOKE')
    if a.n_routes<1 or a.k_paths<2 or a.draws<1 or a.perturbations<0:ap.error('Invalid counts')
    out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    raw=[];neg=[];rr=[];events=[];cand=[];agent_outcomes=[];st=[];preds=[];pm=[];sels=[];ds=[];hashes={}
    for city in a.cities:
        g,ts=load_city(a.data_dir,city);ds.append(dataset_summary(city,g,ts,a.n_routes));m,p,sel=forecast_city(city,ts,a.seed);preds+=p;pm+=m;sels.append(sel)
        for file in discover_city_files(a.data_dir,city):hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        graph=build_graph(g);nodes=np.array(list(graph.nodes));rng=np.random.default_rng(a.seed+17*CITIES.index(city));shock=np.random.default_rng(a.seed+700+CITIES.index(city));dates=pd.DataFrame(p);dates=dates[dates.model=='PredictionAgentBest'][['date','actual','predicted']].to_records(index=False);mean=float(ts.traffic.dropna().mean());od=[];used=set()
        def sample_od(generator,count,excluded=()):
            out=[];used=set(excluded)
            for _ in range(max(2000,200*count)):
                o,d=map(int,generator.choice(nodes,2,replace=False));x,y=graph.nodes[o],graph.nodes[d]
                if (o,d) in used or haversine_km(x['lat'],x['lon'],y['lat'],y['lon'])<3:continue
                used.add((o,d));out.append((o,d))
                if len(out)==count:break
            if len(out)!=count:raise ValueError('Not enough eligible OD pairs')
            return out
        excluded=set()
        if a.exclude_seed is not None:
            previous=sample_od(np.random.default_rng(a.exclude_seed+17*CITIES.index(city)),a.exclude_count)
            excluded=set(previous)|{(d,o) for o,d in previous}
        od=sample_od(rng,a.n_routes,excluded)
        assert not set(od)&excluded
        for oid,(o,d) in enumerate(od):
            day=dates[rng.integers(len(dates))];actual=float(day.actual)/mean if mean>0 else 1.;rho=max(0,float(day.predicted)/mean) if mean>0 else 1.
            if not np.isfinite(actual) or actual<=0:raise ValueError("Positive actual daily multiplier required for paired proxy ratios")
            t=time.perf_counter();paths=(route_candidates if a.candidate_mode=='original' else diversified_candidates)(graph,o,d,a.k_paths,rho);gen=1000*(time.perf_counter()-t);f=route_features(graph,paths);evaluation_f=f.copy();evaluation_f[:,1]*=actual
            assert 1<=len(paths)<=a.k_paths and len(set(paths))==len(paths)
            for i,path in enumerate(paths):cand.append(dict(city=city,od_id=oid,origin=o,destination=d,date=day.date,candidate=i,path=';'.join(map(str,path)),distance_km=f[i,0],co2_proxy=evaluation_f[i,1],static_co2_proxy=f[i,1],traffic_rank=f[i,2],hotspot=f[i,3],pollutant=f[i,4],top10=f[i,5],rho=rho))
            for eps in EPSILONS:
                t=time.perf_counter();s,*_=component_scores(f,eps,rho);sc=1000*(time.perf_counter()-t);choices,protos=select(f,s,eps,True);key=dict(city=city,epsilon=eps,od_id=oid)
                assert set(choices)==set(METHODS+OTHER)
                _,anchor,guard,strict,_,_=safeguards(f,s,eps)
                assert guard[choices[FULL][0]] and strict[choices[STRICT][0]]
                ref=metrics(evaluation_f,s,eps,anchor)
                for variant,protected in ((FULL,('co2_reduction_pct','operational_utility','max_regret')),(STRICT,('co2_reduction_pct','operational_utility','max_regret','weighted_regret','distance_penalty_pct','hotspot_reduction_pct','pollutant_reduction_pct','top10_reduction_pct','score_agreement'))):
                    vv=metrics(evaluation_f,s,eps,choices[variant][0])
                    for metric in protected:
                        if np.isfinite(vv[metric]) and np.isfinite(ref[metric]):assert (vv[metric]-ref[metric])*(-1 if metric in LOWER else 1)>=-1e-7,(city,oid,eps,variant,metric)
                for name,(idx,ms) in choices.items():
                    row=dict(**key,method=name,candidate_count=len(paths),feasible_count=int(eligible(f,eps).sum()),selected_candidate=idx,generation_ms=gen,scoring_ms=sc,decision_ms=ms,total_ms=gen+sc+ms,**metrics(evaluation_f,s,eps,idx));raw.append(row)
                    r=s[eligible(f,eps)].max(axis=0)-s[idx]
                    for j,agent in enumerate(AGENT_NAMES):agent_outcomes.append(dict(**key,method=name,agent=agent,score=float(s[idx,j]),regret=float(max(0,r[j])),accept_at_010=int(r[j]<=.10+TOL),accept_at_025=int(r[j]<=.25+TOL)))
                    if name in set(METHODS)-{'NoPolicy','Legacy_Soft'}:assert row['policy_compliant_pct']==100
                for name,(info,r,e) in protos.items():
                    neg.append(dict(**key,method=name,**info));rr.extend(dict(**key,method=name,**x) for x in r);events.extend(dict(**key,method=name,**x) for x in e)
                    assert info['agreed']+info['fallback']==1 and info['rounds']<=len(TAUS)
                    if info['agreed']:assert r[-1]['accepting']==r[-1]['active_agents'] and info['selected_candidate']==r[-1]['candidate']
                    if name in (FULL,OLD,STRICT,'Guarded_OneRound','Guarded_FixedAspiration'):assert info['logical_messages']==len(e)
                if eps==PRIMARY_EPS:st.extend(dict(**key,**x) for x in stability(g,graph,paths,eps,rho,actual,choices,shock,a.perturbations))
            if (oid+1)%10==0 or oid+1==len(od):LOG.info('%s %d/%d OD pairs',city,oid+1,len(od))
    agents=pd.DataFrame(agent_outcomes);save_csv(agents,out/'agent_outcomes_raw.csv');save_csv(agents.groupby(['city','epsilon','method','agent'],as_index=False)[['score','regret','accept_at_010','accept_at_025']].mean(),out/'agent_outcomes_summary.csv');raw=pd.DataFrame(raw);neg=pd.DataFrame(neg);rr=pd.DataFrame(rr);st=pd.DataFrame(st);group=['city','epsilon','method']
    summary=raw.groupby(group,as_index=False)[list(METRICS)+['generation_ms','scoring_ms']].mean().merge(raw.groupby(group).size().rename('n_OD').reset_index(),on=group)
    metric_counts=raw.groupby(group)[list(METRICS)].count().reset_index();save_csv(metric_counts,out/'metric_valid_counts.csv')
    prot=neg.groupby(group,as_index=False)[['rounds','agreed','fallback','changed','initial_support_pct','final_support_pct','fixed_initial_support_pct','logical_messages','proposal_changes','final_threshold']].mean();stats=paired(raw,a.seed,a.draws,st,neg);joint=joint_outcomes(raw);save_csv(joint,out/'joint_outcomes.csv');save_csv(stats[(stats.proposed==FULL)&(stats.comparator==MATCH)],out/'metric_improvement_audit.csv')
    assert len(raw)==len(a.cities)*a.n_routes*len(EPSILONS)*len(METHODS+OTHER) and not raw.duplicated(group+['od_id']).any();assert raw.groupby(['city','epsilon','od_id']).candidate_count.nunique().eq(1).all()
    ss=pd.DataFrame()
    if len(st):assert np.allclose(st.retention_pct+st.switching_pct,100);ss=st.groupby(group,as_index=False)[list(STABILITY_TESTS)].mean()
    for name,df in [('routing_raw_results',raw),('routing_results',summary),('agent_negotiation_results',neg),('negotiation_summary',prot),('negotiation_rounds',rr),('route_candidates',pd.DataFrame(cand)),('route_stability_raw',st),('route_stability_summary',ss),('paired_statistical_tests',stats),('daily_predictions',pd.DataFrame(preds)),('prediction_results',pd.DataFrame(pm)),('prediction_model_selection',pd.DataFrame(sels)),('dataset_summary',pd.DataFrame(ds)),('agent_ablation_results',summary[summary.method.isin(METHODS)]),('agent_ablation_raw_results',raw[raw.method.isin(METHODS)]),('timing_metrics',raw[group+['od_id','generation_ms','scoring_ms','decision_ms','total_ms']])]:save_csv(df,out/(name+'.csv'))
    save_csv(raw[raw.method==FULL][group+['od_id','selected_candidate']],out/'selected_routes.csv')
    with (out/'negotiation_messages.jsonl').open('w',encoding='utf-8') as h:
        for x in events:h.write(json.dumps(x)+'\n')
    write_tex(out,summary,prot,stats,ss,joint)
    if not a.skip_figures:
        fig=out/'figures';fig.mkdir(exist_ok=True)
        for col in ('agreed','rounds','changed','logical_messages'):
            ax=prot[prot.method==FULL].pivot(index='epsilon',columns='city',values=col).plot(marker='o');ax.set_ylabel(col);ax.figure.tight_layout();ax.figure.savefig(fig/(col+'.png'),dpi=220);plt.close(ax.figure)
        for col in ('operational_utility','max_regret','co2_reduction_pct','policy_compliant_pct','weighted_regret','pareto_efficient_pct','utility_gap','jain_satisfaction'):
            ax=summary[np.isclose(summary.epsilon,PRIMARY_EPS)&summary.method.isin((FULL,OLD,STRICT,MATCH,DIRECT,STRICT_DIRECT,'Guarded_OneRound','Guarded_FixedAspiration'))].pivot(index='method',columns='city',values=col).plot.bar(rot=35);ax.set_ylabel(col);ax.figure.tight_layout();ax.figure.savefig(fig/(col+'.png'),dpi=220);plt.close(ax.figure)
    manifest=dict(quick_check=a.quick_check,seed=a.seed,exclude_seed=a.exclude_seed,exclude_count=a.exclude_count,OD_per_city=a.n_routes,epsilons=EPSILONS,thresholds=TAUS,cities=a.cities,k=a.k_paths,candidate_mode=a.candidate_mode,draws=a.draws,perturbations=a.perturbations,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,networkx=nx.__version__,data_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),logical_messages_not_network_traffic=True,primary_comparator=MATCH,primary_metrics=PRIMARY_METRICS,primary_epsilon=PRIMARY_EPS,checks_passed=True,rows=len(raw),messages=len(events));(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');LOG.info('Complete: %s',out)

if __name__=='__main__':main()
