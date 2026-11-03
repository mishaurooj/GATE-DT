#!/usr/bin/env python3
"""GATE-DT++ / CARA reproducible, unified experiments.

Matches the formulas and experimental protocol in V2-revision-blue(5).tex.
All reported values are calculated from inputs. NO manuscript numbers are hardcoded as results.
The paper contains mutually inconsistent numerical claims; see audit outputs.

Windows (from E:\\other\\GATE-DT-main):
    python Code\\gate_dt_paper_unified_final.py
    python Code\\gate_dt_paper_unified_final.py --quick-check

Dependencies: numpy pandas scipy scikit-learn networkx matplotlib.
"""
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
        df[col] = df[col].ffill().bfill().fillna(0.0)
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


def ablation_choices(scores, cara_scores, disagreement, feats):
    full=scores @ WEIGHTS - DISAGREEMENT_WEIGHT*disagreement
    ablated={"CARA_Full":int(full.argmax())}
    for key,index in (("PredictionAgent",0),("HotspotAgent",1),("PollutantAgent",2)):
        w=WEIGHTS.copy();w[index]=0
        if index==1:w[5]=0
        # Recompute disagreement on the *remaining* scores, rather than retaining removed agent.
        cols=np.flatnonzero(w>0)
        arr=scores[:,cols]
        utility = (scores @ w) - DISAGREEMENT_WEIGHT*arr.std(axis=1,ddof=1)
        ablated[f"CARA_without_{key}"]=int(utility.argmax())
    w=WEIGHTS.copy();w[4]=0
    arr=scores[:,[0,1,2,3,5]]
    ablated["CARA_without_PolicyAgent"]=int((scores@w - DISAGREEMENT_WEIGHT*arr.std(axis=1,ddof=1)).argmax())
    # Without negotiation, use a fixed weighted average, without disagreement penalty or policy voting.
    static=.25*scores[:,0]+.25*scores[:,1]+.20*scores[:,2]+.30*scores[:,3]
    ablated["CARA_without_NegotiationAgent"]=int(static.argmax())
    ablated["SingleAgent_Routing"]=baseline_choice("SingleAgent_Routing",feats,scores)
    ablated["StaticDT_Routing"]=baseline_choice("StaticDT_Routing",feats,scores)
    return ablated


def path_metrics(city, eps, oid, method, idx, feats, cons, disagree, cara_util, runtime, category, npaths):
    base=feats[np.argmin(feats[:,0])]
    chosen=feats[idx]
    d,c,rt,hs,pol,top=chosen
    d0,c0,_,h0,_,t0=base
    red=safe_pct(c0-c,c0)
    dp=safe_pct(d-d0,d0)
    hr=safe_pct(h0-hs,h0)
    tr=safe_pct(t0-top,t0)
    viol=max(0.0,dp - eps*100) if np.isfinite(dp) else np.nan
    # Eq. report_utility; paper does not clip distance penalty (all routes have d>=d0).
    utility=red-2*dp+.5*hr-8*viol
    return dict(city=city,paper_city=ALIAS[city],epsilon=eps,od_id=oid,
                method=method,group=category,candidate_routes=npaths,chosen_candidate=idx,
                distance_km=d, CO2_proxy_cost=c, hotspot_cost=hs,pollutant_cost=pol,top10_hotspot_cells=top,
                CO2_reduction_pct_vs_shortest=red,distance_penalty_pct_vs_shortest=dp,
                hotspot_reduction_pct_vs_shortest=hr,top10_reduction_pct_vs_shortest=tr,
                policy_violation_pct=viol,policy_compliant=int(viol<=1.e-9),
                operational_utility_score=utility,consensus_utility=float(cons[idx]),
                agent_disagreement=float(disagree[idx]),consensus_stability=float(1-np.clip(disagree[idx],0,1)),
                cara_route_utility=float(cara_util[idx]),runtime_seconds=runtime)


def summarize_methods(raw):
    groups=[]
    for (city,eps,method,group),frame in raw.groupby(["city","epsilon","method","group"],sort=False):
        r={"city":city,"paper_city":ALIAS[city],"epsilon":eps,"method":method,"group":group,"n_OD":len(frame)}
        for name in ("CO2_reduction_pct_vs_shortest","distance_penalty_pct_vs_shortest","hotspot_reduction_pct_vs_shortest",
                     "top10_reduction_pct_vs_shortest","policy_violation_pct","operational_utility_score",
                     "consensus_stability","consensus_utility","cara_route_utility","runtime_seconds"):
            col=frame[name].dropna()
            r[name+"_mean"]=float(col.mean()) if len(col) else np.nan
            r[name+"_median"]=float(col.median()) if len(col) else np.nan
            r[name+"_std"]=float(col.std(ddof=1)) if len(col)>1 else np.nan
        r["policy_feasible_route_pct"]=100*frame.policy_compliant.mean()
        groups.append(r)
    return pd.DataFrame(groups)


def paired_statistics(raw, eps, seed, bootstrap=10000, permutations=10000):
    # Primary test: CARA minus ShortestPath in OD-wise percentage-point CO2 reductions.
    rng=np.random.default_rng(seed+1701)
    rows=[]
    part=raw[(np.isclose(raw.epsilon,eps))&(raw.group=="routing")]
    for city in CITIES:
        citydf=part[part.city==city]
        p=citydf.pivot_table(index="od_id",columns="method",values="CO2_reduction_pct_vs_shortest",aggfunc="first")
        if not {"CARA_Full","ShortestPath"}.issubset(p.columns):continue
        differences=(p["CARA_Full"]-p["ShortestPath"]).dropna().to_numpy()
        n=len(differences)
        if not n:continue
        # Use batches to bound RAM independently of the number of requested permutations.
        sample_means=[]
        for b in range(0,bootstrap,1000):
            indices=rng.integers(0,n,(min(1000,bootstrap-b),n))
            sample_means.extend(differences[indices].mean(axis=1))
        lo,hi=np.quantile(sample_means,[.025,.975])
        obs=abs(differences.mean());ge=0
        for b in range(0,permutations,1000):
            flips=rng.choice(np.array([-1,1],dtype=np.int8),(min(1000,permutations-b),n))
            ge+=int(np.count_nonzero(abs((flips*differences).mean(axis=1)) >= obs - 1e-12))
        pval=(ge+1)/(permutations+1)
        rows.append(dict(city=city,paper_city=ALIAS[city],epsilon=eps,n_matched_OD=n,
                         baseline="ShortestPath",statistic="paired_CO2_reduction_pp",
                         paired_mean_pp=float(differences.mean()),bootstrap_resamples=bootstrap,
                         CI_95_lower_pp=float(lo),CI_95_upper_pp=float(hi),
                         permutation_draws=permutations,permutation_two_sided_p=float(pval),
                         number_nonzero_differences=int((abs(differences)>1e-12).sum())))
    return pd.DataFrame(rows)


def route_stability(graph, od_records, city, eps, M, perturb_sigma, seed):
    # Controlled, independent node-level shocks. Re-evaluate fixed candidate pool per OD.
    # consecutive retention matches Table metric_definitions; anchor_retention matches preceding text.
    rng=np.random.default_rng(seed+700+CITIES.index(city))
    nodes=graph.nodes
    n=1+max(nodes)
    grid_traffic=np.zeros(n);nox=np.zeros(n);pm=np.zeros(n)
    live=np.array(list(nodes),dtype=int)
    grid_traffic[live]=np.array([nodes[i]["traffic"] for i in live]);
    nox[live]=np.array([nodes[i]["rn"] for i in live]);pm[live]=np.array([nodes[i]["rp"] for i in live])
    rows=[]
    # Hold spatial network, OD, and candidate set fixed to isolate score sensitivity.
    for record in od_records:
        paths,feats0=record["paths"],record["feats"]
        base_idx=int(np.argmin(feats0[:,0]))
        scores,viol,cons,disag,util=component_scores(feats0,eps,record["rho"])
        selected=[int(np.argmax(util))]
        for m in range(M):
            pert=np.maximum(0,grid_traffic[live]*rng.normal(1,perturb_sigma,len(live)))
            traffic=grid_traffic.copy();traffic[live]=pert
            rt=np.zeros(n);rt[live]=percentile_rank(traffic[live])
            hot=np.zeros(n);hot[live]=.70*rt[live]+.20*nox[live]+.10*pm[live]
            pol=np.zeros(n);pol[live]=.55*nox[live]+.45*pm[live]
            top=np.zeros(n);top[live]=(hot[live]>=np.quantile(hot[live],.90)).astype(int)
            feats=route_features(graph,paths,traffic,rt,hot,pol,top)
            scores,viol,cons,disag,util=component_scores(feats,eps,record["rho"])
            selected.append(int(np.argmax(util)))
        prev=np.array(selected[:-1]);nxt=np.array(selected[1:])
        rows.append(dict(city=city,paper_city=ALIAS[city],epsilon=eps,od_id=record["od_id"],
                         baseline_selected_candidate=selected[0],n_perturbations=M,
                         route_retention_pct=float(100*(prev==nxt).mean()),
                         route_switching_pct=float(100*(prev!=nxt).mean()),
                         anchor_retention_pct=float(100*(np.array(selected[1:])==selected[0]).mean())))
    return pd.DataFrame(rows)


def run_routing(city,g,ts,pred,epsilons,n_routes,K,seed,stability_n,perturb_sigma,neighbor_k,max_link):
    rng=np.random.default_rng(seed+17*CITIES.index(city))
    graph=build_graph(g,neighbor_k,max_link)
    node_list=np.array(list(graph.nodes))
    if len(node_list)<20: raise RuntimeError(f"{city}: largest connected component too small: {len(node_list)}")
    mean_t=float(ts.traffic.dropna().mean())
    forecast=pd.DataFrame(pred)
    selected=forecast[(forecast.city==city)&(forecast.model=="PredictionAgentBest")]
    if selected.empty:raise ValueError(f"{city}: no predictions")
    dates=selected[["actual","predicted"]].to_numpy()
    # Select 100 unique, connected OD pairs with geographic separation.
    OD=[];used=set();attempts=0
    while len(OD)<n_routes and attempts < max(2000,200*n_routes):
        attempts+=1
        o,d=map(int,rng.choice(node_list,size=2,replace=False))
        if (o,d) in used:continue
        a,b=graph.nodes[o],graph.nodes[d]
        if haversine_km(a["lat"],a["lon"],b["lat"],b["lon"])<3.:continue
        used.add((o,d));OD.append((o,d))
    if len(OD)!=n_routes:raise RuntimeError(f"{city}: only {len(OD)} eligible OD pairs found, requested {n_routes}")
    raw_rows=[];timing=[];negotiation=[];route_logs=[];od_records=[]
    for oid,(origin,dest) in enumerate(OD):
        draw=dates[rng.integers(0,len(dates))]
        actual_mult=float(draw[0])/mean_t if mean_t>0 else 1
        rho=max(0,float(draw[1])/mean_t) if mean_t>0 else 1
        t0=time.perf_counter()
        paths=route_candidates(graph,origin,dest,K,rho)
        generation=time.perf_counter()-t0
        if not paths:raise RuntimeError(f"{city}: zero routes generated for OD {oid}")
        feats=route_features(graph,paths)
        # Re-scale *all* route traffic cost by the same actual-day multiplier.
        feats[:,1] *= actual_mult
        od_records.append(dict(od_id=oid,origin=origin,dest=dest,paths=paths,feats=feats,rho=rho))
        for eps in epsilons:
            t1=time.perf_counter()
            scores,viol,cons,disag,util=component_scores(feats,eps,rho)
            idx=int(np.argmax(util))  # Exact paper Eq. cara_route_selection: NO silent shortest-path override.
            negotiation_seconds=time.perf_counter()-t1
            cara_runtime=generation+negotiation_seconds
            record=path_metrics(city,eps,oid,"CARA_Full",idx,feats,cons,disag,util,cara_runtime,"routing",len(paths))
            raw_rows.append(record)
            route_logs.append(dict(city=city,paper_city=ALIAS[city],epsilon=eps,od_id=oid,
                                   origin=origin,destination=dest,chosen_route=";".join(map(str,paths[idx])),
                                   candidate_count=len(paths)))
            negotiation.append(dict(city=city,paper_city=ALIAS[city],epsilon=eps,od_id=oid,
                                    candidate_routes=len(paths),chosen_candidate=idx,
                                    consensus_utility=cons[idx],agent_disagreement=disag[idx],
                                    consensus_stability=1-np.clip(disag[idx],0,1),cara_route_utility=util[idx],
                                    policy_violation_pct=viol[idx],generation_time_seconds=generation,
                                    negotiation_time_seconds=negotiation_seconds,total_decision_time_seconds=cara_runtime))
            timing.append(dict(city=city,paper_city=ALIAS[city],epsilon=eps,od_id=oid,
                               candidate_routes=len(paths),generation_time_seconds=generation,
                               negotiation_time_seconds=negotiation_seconds,total_CARA_time_seconds=cara_runtime))
            for method in BASELINES:
                t2=time.perf_counter();bidx=baseline_choice(method,feats,scores)
                raw_rows.append(path_metrics(city,eps,oid,method,bidx,feats,cons,disag,util,
                                             time.perf_counter()-t2,"routing",len(paths)))
            ablated=ablation_choices(scores,util,disag,feats)
            for method,bidx in ablated.items():
                raw_rows.append(path_metrics(city,eps,oid,method,bidx,feats,cons,disag,util,
                                             0.0,"ablation",len(paths)))
        if (oid+1)%25==0:LOG.info("%s OD %d/%d complete",city,oid+1,n_routes)
    if stability_n:
        stability=route_stability(graph,od_records,city,PRIMARY_EPS,stability_n,perturb_sigma,seed)
    else:stability=pd.DataFrame()
    return (pd.DataFrame(raw_rows),pd.DataFrame(timing),pd.DataFrame(negotiation),
            pd.DataFrame(route_logs),stability)


def city_hotspot_stats(city,g):
    total=g.traffic.sum();rows=[]
    for frac in [.05,.10,.20]:
        k=max(1,int(len(g)*frac))
        rows.append(dict(city=city,paper_city=ALIAS[city],fraction=frac,cells=k,
                         traffic_share_by_traffic_rank_pct=safe_pct(g.nlargest(k,"traffic").traffic.sum(),total),
                         traffic_share_by_hotspot_rank_pct=safe_pct(g.nlargest(k,"hotspot").traffic.sum(),total),
                         mean_top_hotspot_score=float(g.nlargest(k,"hotspot").hotspot.mean())))
    return rows


def cross_city_statistics(summary,eps):
    cara=summary[(summary.method=="CARA_Full")&(summary.group=="routing")&(np.isclose(summary.epsilon,eps))]
    output=[]
    for metric in ["CO2_reduction_pct_vs_shortest_mean","distance_penalty_pct_vs_shortest_mean",
                   "consensus_stability_mean","operational_utility_score_mean"]:
        arr=cara[metric].dropna().to_numpy()
        if len(arr)==0:continue
        mu=arr.mean();sd=arr.std(ddof=1) if len(arr)>1 else np.nan
        output.append(dict(metric=metric,n_cities=len(arr),mean=mu,sample_SD=sd,
                           CV_pct=safe_pct(sd,abs(mu))))
    return pd.DataFrame(output)


def network_feedback(summary,dataset_summary_df,stats,stability,cross,eps):
    c=summary[(summary.method=="CARA_Full")&(summary.group=="routing")&(np.isclose(summary.epsilon,eps))]
    result=[]
    for city in CITIES:
        rows=c[c.city==city]
        ds=dataset_summary_df[dataset_summary_df.city==city]
        ss=stats[stats.city==city]
        stab=stability[stability.city==city] if not stability.empty else pd.DataFrame()
        if rows.empty:continue
        a,b=rows.iloc[0],ds.iloc[0]
        cr=a.CO2_reduction_pct_vs_shortest_mean
        dp=a.distance_penalty_pct_vs_shortest_mean
        r=dict(city=city,paper_city=ALIAS[city],epsilon=eps,
               evaluated_OD_pairs=int(a.n_OD),observation_days=int(b.observation_days),
               CO2_cost_reduction_pct=cr,distance_penalty_pct=dp,
               operational_utility=a.operational_utility_score_mean,
               consensus_score=a.consensus_stability_mean,
               policy_violation_pct=a.policy_violation_pct_mean,
               policy_compliance_pct=a.policy_feasible_route_pct,
               compliant_OD_selections=round(a.policy_feasible_route_pct/100*a.n_OD),
               CO2_div_distance=(cr/dp if dp else np.nan),
               CO2_minus_distance_pp=cr-dp)
        if not ss.empty:r.update(ss.iloc[0].drop(["city","paper_city","epsilon"]).to_dict())
        if not stab.empty:
            r["route_retention_pct"]=stab.route_retention_pct.mean()
            r["route_switching_pct"]=stab.route_switching_pct.mean()
            r["anchor_retention_pct"]=stab.anchor_retention_pct.mean()
        result.append(r)
    return pd.DataFrame(result)


def manuscript_audit(feedback):
    # Reference values copied solely for comparison; they are NEVER used to produce results.
    paper={"Delhi": dict(CO2_cost_reduction_pct=30.712,distance_penalty_pct=3.934,
                         operational_utility=31.609,consensus_score=.785,policy_violation_pct=0,
                         paired_mean_pp=30.7,CI_95_lower_pp=27.1,CI_95_upper_pp=34.3,
                         permutation_two_sided_p=.001,route_retention_pct=91,route_switching_pct=9),
           "Mumbai":dict(CO2_cost_reduction_pct=23.156,distance_penalty_pct=3.445,
                         operational_utility=23.634,consensus_score=.772,policy_violation_pct=0,
                         paired_mean_pp=23.2,CI_95_lower_pp=19.8,CI_95_upper_pp=26.6,
                         permutation_two_sided_p=.002,route_retention_pct=89,route_switching_pct=11),
           "Chennai":dict(CO2_cost_reduction_pct=21.640,distance_penalty_pct=3.171,
                         operational_utility=24.498,consensus_score=.743,policy_violation_pct=0,
                         paired_mean_pp=21.6,CI_95_lower_pp=18.1,CI_95_upper_pp=25.1,
                         permutation_two_sided_p=.003,route_retention_pct=87,route_switching_pct=13)}
    records=[]
    for _,row in feedback.iterrows():
        for key,claimed in paper[row.city].items():
            val=row.get(key,np.nan)
            tolerance=.0005 if key in ("consensus_score", "permutation_two_sided_p") else .005
            records.append(dict(city=row.city,paper_city=ALIAS[row.city],metric=key,
                                manuscript_claim=claimed,observed=val,
                                absolute_difference=abs(val-claimed) if np.isfinite(val) else np.nan,
                                agrees_within_rounding=(bool(abs(val-claimed)<=tolerance) if np.isfinite(val) else False)))
    return pd.DataFrame(records)


def make_figures(fig_dir, grids, pred_metrics_df, summary, stability, timing, eps):
    fig_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"figure.dpi":120,"savefig.dpi":220,"font.size":9})
    main=summary[(summary.group=="routing")&(np.isclose(summary.epsilon,eps))]
    abl=summary[(summary.group=="ablation")&(np.isclose(summary.epsilon,eps))]
    # Paper six-panel overview. Data are loaded from run tables, never hardcoded.
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    ax=axes[0,0]
    pivot=pred_metrics_df[pred_metrics_df.model.isin(MODEL_NAMES)].pivot(index="model",columns="city",values="sMAPE_pct").reindex(MODEL_NAMES)
    pivot.plot.bar(ax=ax,rot=45);ax.set_title("(a) Prediction sMAPE");ax.set_ylabel("sMAPE (%)");ax.legend(fontsize=7)
    ax=axes[0,1]
    p=main.pivot(index="method",columns="city",values="operational_utility_score_mean")
    p.plot.bar(ax=ax,rot=85,legend=False);ax.set_title("(b) Operational utility")
    ax=axes[0,2]
    main.pivot(index="method",columns="city",values="policy_violation_pct_mean").plot.bar(ax=ax,rot=85)
    ax.set_title("(c) Detour-policy violation");ax.set_ylabel("Percentage points");ax.legend(fontsize=7)
    ax=axes[1,0]
    abl.pivot(index="method",columns="city",values="operational_utility_score_mean").plot.bar(ax=ax,rot=85)
    ax.set_title("(d) Agent-ablation utility");ax.legend(fontsize=7)
    ax=axes[1,1]
    for city in CITIES:
        d=main[main.city==city]
        ax.scatter(d.distance_penalty_pct_vs_shortest_mean,d.CO2_reduction_pct_vs_shortest_mean,label=city,s=30)
    ax.set_title("(e) Carbon-cost/distance trade-off");ax.set_xlabel("Distance penalty (%)");ax.set_ylabel("Carbon-cost reduction (%)");ax.legend(fontsize=7)
    ax=axes[1,2]
    allcara=summary[(summary.method=="CARA_Full")&(summary.group=="routing")]
    for city in CITIES:
        d=allcara[allcara.city==city].sort_values("epsilon")
        ax.plot(d.epsilon,d.consensus_stability_mean,marker="o",label=city)
    ax.set_title("(f) Consensus stability vs epsilon");ax.set_xlabel("Epsilon");ax.set_ylabel("1 - score disagreement");ax.legend(fontsize=7)
    fig.savefig(fig_dir/"gate_dt_cara.png",bbox_inches="tight");plt.close(fig)
    # Static model architecture, matching the existing manuscript's image filename.
    architecture, axarch = plt.subplots(figsize=(14, 5))
    axarch.set_xlim(0, 12);axarch.set_ylim(0, 5);axarch.axis("off")
    architecture_nodes=[(1,3.6,"Spatial grids\nTraffic / NOx / PM2.5"),
                        (4,3.6,"Digital-twin state\nRanks + hotspot map"),
                        (7,3.6,"Forecast + graph\nCandidate routes (K <= 25)"),
                        (10,3.6,"CARA scoring\nSix components"),
                        (10,1.1,"Output route\nUtility + compliance"),
                        (6,1.1,"Statistical tests\nBootstrap + stability")]
    for x,y,title in architecture_nodes:
        axarch.text(x,y,title,ha="center",va="center",fontsize=10,
                    bbox=dict(boxstyle="round,pad=.6",fc="#e9f1fb",ec="#285580",lw=1.5))
    for a,b in [((2,3.6),(3,3.6)),((5.3,3.6),(5.8,3.6)),((8.25,3.6),(8.65,3.6)),
                ((10,3.08),(10,1.65)),((9,1.1),(7.6,1.1))]:
        axarch.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.5))
    axarch.set_title("GATE-DT++ digital twin and CARA decision pipeline",fontsize=14)
    architecture.savefig(fig_dir/"GATE-DT++.jpg",dpi=200,bbox_inches="tight")
    plt.close(architecture)
    # City hotspot distributions, used for figure fig:city_hotspots.
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    for ax,city in zip(axes,CITIES):
        g=grids[city];sc=ax.scatter(g.lon,g.lat,c=g.hotspot,cmap="viridis",s=3,rasterized=True)
        ax.set_title(f"{ALIAS[city]} ({city})");ax.set_xlabel("Longitude");ax.set_ylabel("Latitude")
        fig.colorbar(sc,ax=ax,label="Hotspot score")
        single,ax1=plt.subplots(figsize=(6,5));p=ax1.scatter(g.lon,g.lat,c=g.hotspot,cmap="viridis",s=3,rasterized=True)
        ax1.set(title=f"{city} hotspot score",xlabel="Longitude",ylabel="Latitude");single.colorbar(p,ax=ax1)
        single.savefig(fig_dir/f"{city.lower()}_hotspot_map.png",bbox_inches="tight");plt.close(single)
    fig.savefig(fig_dir/"city_hotspots.png",bbox_inches="tight");plt.close(fig)
    # Agent ablation utility differences relative to full CARA.
    pa=abl.pivot(index="method",columns="city",values="operational_utility_score_mean")
    if "CARA_Full" in pa.index:
        change=pa.subtract(pa.loc["CARA_Full"],axis="columns");change=change.drop(index="CARA_Full")
        ax=change.plot.bar(figsize=(11,5));ax.axhline(0,color="black",lw=.6)
        ax.set(title="Ablation operational utility minus full CARA",ylabel="Utility difference")
        ax.figure.tight_layout();ax.figure.savefig(fig_dir/"ablation_utility_publication.png",bbox_inches="tight");plt.close(ax.figure)
    # Decision-time across epsilon (all route means).
    ptime=timing.pivot_table(index="epsilon",columns="city",values="total_CARA_time_seconds",aggfunc="mean")
    ax=ptime.plot.bar(figsize=(9,4.8));ax.set(xlabel="Epsilon",ylabel="Seconds per route",title="CARA decision time (generation + scoring)")
    ax.figure.tight_layout();ax.figure.savefig(fig_dir/"cara_decision_time.png",bbox_inches="tight");plt.close(ax.figure)
    # Extra companion figures.
    for column,filename,label in [("CO2_reduction_pct_vs_shortest_mean","CO2_reduction_by_method.png","CO2-cost reduction (%)"),
                                   ("policy_violation_pct_mean","policy_violation_by_method.png","Detour violation (%)"),
                                   ("operational_utility_score_mean","operational_utility_by_method.png","Operational utility")]:
        ax=main.pivot(index="method",columns="city",values=column).plot.bar(figsize=(12,5))
        ax.set_ylabel(label);ax.figure.tight_layout();ax.figure.savefig(fig_dir/filename,bbox_inches="tight");plt.close(ax.figure)


def fmt(v,digits=3):
    return "--" if not np.isfinite(v) else f"{float(v):.{digits}f}"


def latex_tables(dataset,summary,pred,feedback,stats,stability,cross,eps,output):
    # Standalone LaTeX fragments generated from experiments, to replace paper's hard-coded values.
    c=summary[(summary.group=="routing")&(np.isclose(summary.epsilon,eps))]
    a=summary[(summary.group=="ablation")&(np.isclose(summary.epsilon,eps))]
    tex=[]
    def begin(title, columns, head):
        tex.extend([f"% {title}",r"\begin{tabular}{"+columns+"}",r"\hline",head+r" \\",r"\hline"])
    def end():tex.extend([r"\hline",r"\end{tabular}",""])
    begin("tab:dataset_summary_final", "lrrrrrrrr",r"City & Grid & Active & Traffic & Mean daily traffic proxy & Top5 (\%) & Top10 (\%) & Days & OD")
    for city in CITIES:
        r=dataset[dataset.city==city].iloc[0]
        tex.append(f"{ALIAS[city]} & {int(r.grid_cells)} & {int(r.active_cells)} & {fmt(r.traffic_sum,0)} & {fmt(r.mean_daily_traffic_CO2_proxy,2)} & {fmt(r.top5_share_pct,2)} & {fmt(r.top10_share_pct,2)} & {int(r.observation_days)} & {int(r.OD_pairs_requested)}"+r" \\")
    end()
    begin("tab:cross_city_results (primary epsilon)","llrrrrr",r"Method & City & CO$_2$ proxy red. (\%) & Policy violation (\%) & Distance penalty (\%) & Consensus & Utility")
    for method in ["ShortestPath","Weighted_EcoRouting","PredictiveDT_Routing","SingleAgent_Routing","CO2_Dijkstra","CARA_Full"]:
        for city in CITIES:
            rr=c[(c.city==city)&(c.method==method)]
            if rr.empty:continue
            r=rr.iloc[0]
            tex.append(f"{method.replace('_',' ')} & {ALIAS[city]} & {fmt(r.CO2_reduction_pct_vs_shortest_mean)} & {fmt(r.policy_violation_pct_mean)} & {fmt(r.distance_penalty_pct_vs_shortest_mean)} & {fmt(r.consensus_stability_mean)} & {fmt(r.operational_utility_score_mean)}"+r" \\")
    end()
    begin("tab:prediction_models_final","l"+"rrr"*3,r"Model & MC1 RMSE & MC1 sMAPE & MC1 R$^2$ & MC2 RMSE & MC2 sMAPE & MC2 R$^2$ & MC3 RMSE & MC3 sMAPE & MC3 R$^2$")
    for model in MODEL_NAMES:
        v=[]
        for city in CITIES:
            r=pred[(pred.city==city)&(pred.model==model)]
            v.extend([fmt(float(r.RMSE.iloc[0])) if len(r) else '--',fmt(float(r.sMAPE_pct.iloc[0]),2) if len(r) else '--',fmt(float(r.R2.iloc[0])) if len(r) else '--'])
        tex.append(model+" & "+" & ".join(v)+r" \\")
    end()
    begin("tab:unified_ablation","l"+"rrrr"*3,r"Configuration & MC1 CO$_2$ & MC1 Policy & MC1 Distance & MC1 Utility & MC2 CO$_2$ & MC2 Policy & MC2 Distance & MC2 Utility & MC3 CO$_2$ & MC3 Policy & MC3 Distance & MC3 Utility")
    for method in ABLATIONS:
        v=[]
        for city in CITIES:
            r=a[(a.city==city)&(a.method==method)]
            if r.empty:v.extend(["--"]*4);continue
            z=r.iloc[0]
            v.extend([fmt(z.CO2_reduction_pct_vs_shortest_mean),fmt(z.policy_violation_pct_mean),fmt(z.distance_penalty_pct_vs_shortest_mean),fmt(z.operational_utility_score_mean)])
        tex.append(method.replace("_"," ")+" & "+" & ".join(v)+r" \\")
    end()
    begin("tab:network_feedback parts b,c,d,f","lccc",r"Metric & MC1 & MC2 & MC3")
    all_metrics=[("CO2-cost reduction (%)","CO2_cost_reduction_pct"),("Distance penalty (%)","distance_penalty_pct"),
                 ("Operational utility","operational_utility"),("Consensus stability","consensus_score"),
                 ("Policy violation (%)","policy_violation_pct"),("Policy compliance (%)","policy_compliance_pct"),
                 ("CO2 reduction / distance","CO2_div_distance"),("CO2 minus distance (pp)","CO2_minus_distance_pp"),
                 ("Paired CO2 difference (pp)","paired_mean_pp"),("Bootstrap 95% lower","CI_95_lower_pp"),
                 ("Bootstrap 95% upper","CI_95_upper_pp"),("Permutation p-value","permutation_two_sided_p"),
                 ("Route retention (%)","route_retention_pct"),("Route switching (%)","route_switching_pct")]
    for label,key in all_metrics:
        values=[]
        for city in CITIES:
            r=feedback[feedback.city==city].iloc[0]
            values.append(fmt(r.get(key,np.nan),4 if "p-value" in label else 3))
        tex.append(label+" & "+" & ".join(values)+r" \\")
    end()
    begin("tab:network_feedback part e, sample SD","lrrr",r"Metric & Mean & Sample SD & CV (\%)")
    for _,r in cross.iterrows():tex.append(r.metric+f" & {fmt(r['mean'])} & {fmt(r.sample_SD)} & {fmt(r.CV_pct,2)}"+r" \\")
    end()
    output.write_text("\n".join(tex),encoding="utf-8")


def verify_results(raw, summary, timing, stability, stats, route_log, args, fig_dir):
    """Ten automated invariants. Passing is code QA, not a publication validity score."""
    outcomes={}
    def check(name, condition, detail):
        outcomes[name]=dict(passed=bool(condition), detail=detail)
    c=raw[raw.group=="routing"]
    selected=c[c.method=="CARA_Full"]
    check("all_cities_and_eps",set(args.cities)==set(selected.city.unique()) and
          set(np.round(args.epsilons,9))==set(np.round(selected.epsilon.unique(),9)),
          "Every requested city and epsilon has CARA records")
    expected_od=args.n_routes*len(args.cities)*len(args.epsilons)
    check("matched_od_count",len(selected)==expected_od and len(route_log)==expected_od,
          f"Expected {expected_od} CARA observations, found {len(selected)}")
    check("common_candidate_bound",c.candidate_routes.between(1,args.k_paths).all(),
          f"All candidate counts must be between 1 and K={args.k_paths}")
    check("common_candidate_pool",c.groupby(["city","epsilon","od_id"]).candidate_routes.nunique().max()==1,
          "All routing methods use an identical pool per OD/epsilon")
    check("epsilon_same_OD_set",selected.groupby(["city","epsilon"]).od_id.nunique().eq(args.n_routes).all(),
          "The same OD IDs are present at every epsilon")
    cr=selected.CO2_reduction_pct_vs_shortest
    d=selected.distance_penalty_pct_vs_shortest
    h=selected.hotspot_reduction_pct_vs_shortest
    p=selected.policy_violation_pct
    expected_utility=cr-2*d+.5*h-8*p
    check("utility_formula",np.allclose(selected.operational_utility_score,expected_utility,equal_nan=True),
          "Evaluation utility matches Eq. report_utility")
    check("detour_formula",np.allclose(p,np.maximum(0,d-100*selected.epsilon),equal_nan=True,atol=1e-8),
          "Policy violation matches Eq. violation")
    check("statistics_replicates",len(stats)==len(args.cities) and
          (stats.bootstrap_resamples==args.bootstrap).all() and
          (stats.permutation_draws==args.permutations).all() and
          stats.permutation_two_sided_p.between(0,1).all(),
          "All requested paired tests and draws completed")
    check("stability_complement",(not args.perturbations) or
          (len(stability)==args.n_routes*len(args.cities) and
           np.allclose(stability.route_retention_pct+stability.route_switching_pct,100)),
          "Route switching and retention sum to 100 percent")
    expected_images=["gate_dt_cara.png","ablation_utility_publication.png","cara_decision_time.png",
                     "delhi_hotspot_map.png","mumbai_hotspot_map.png","chennai_hotspot_map.png","GATE-DT++.jpg"]
    check("all_paper_figures",args.skip_figures or all((fig_dir/p).exists() for p in expected_images),
          "All referenced experimental figures and architecture diagram are present")
    score=10*sum(v["passed"] for v in outcomes.values())
    return dict(score_out_of_100=score, scope="automated code/data consistency checks only",
                publication_reproducibility_not_certified=True,tests=outcomes)


def parse_args(argv=None):
    root=Path(__file__).resolve().parent.parent
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root",type=Path,default=root,help="GATE-DT-main root; auto-detected when script is inside Code")
    ap.add_argument("--data-dir",type=Path,default=None)
    ap.add_argument("--out-dir",type=Path,default=None)
    ap.add_argument("--fig-dir",type=Path,default=None)
    ap.add_argument("--cities",nargs="+",choices=CITIES,default=list(CITIES))
    ap.add_argument("--n-routes",type=int,default=100)
    ap.add_argument("--k-paths",type=int,default=25)
    ap.add_argument("--epsilons",nargs="+",type=float,default=list(EPSILONS))
    ap.add_argument("--primary-epsilon",type=float,default=PRIMARY_EPS)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--bootstrap",type=int,default=10000)
    ap.add_argument("--permutations",type=int,default=10000)
    ap.add_argument("--perturbations",type=int,default=20)
    ap.add_argument("--perturb-sigma",type=float,default=.05)
    ap.add_argument("--neighbor-k",type=int,default=9)
    ap.add_argument("--max-link-km",type=float,default=1.75)
    ap.add_argument("--skip-figures",action="store_true")
    ap.add_argument("--quick-check",action="store_true",help="Small smoke run (NOT publishable results)")
    ap.add_argument("--verbose",action="store_true")
    args=ap.parse_args(argv)
    args.root=args.root.resolve()
    args.data_dir=(args.data_dir or args.root/"Dataset").resolve()
    args.out_dir=(args.out_dir or args.root/"Results_PaperExact").resolve()
    args.fig_dir=(args.fig_dir or args.root/"Figs").resolve()
    if args.quick_check:
        args.n_routes=min(args.n_routes,3);args.k_paths=min(args.k_paths,8)
        if args.out_dir == (args.root/"Results_PaperExact").resolve():
            args.out_dir=(args.root/"Results_PaperExact_SMOKE").resolve()
        if args.fig_dir == (args.root/"Figs").resolve():
            args.fig_dir=(args.root/"Figs_SMOKE").resolve()
        args.bootstrap=min(args.bootstrap,100);args.permutations=min(args.permutations,100)
        args.perturbations=min(args.perturbations,3)
    if args.primary_epsilon not in args.epsilons:raise ValueError("primary-epsilon must be among --epsilons")
    if args.n_routes<1 or args.k_paths<2 or args.perturbations<0 or args.bootstrap<1 or args.permutations<1:
        raise ValueError("Invalid simulation or statistical trial counts")
    return args


def main(argv=None):
    args=parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,format="%(asctime)s %(levelname)s: %(message)s",datefmt="%H:%M:%S")
    t0=time.perf_counter()
    LOG.info("Input: %s",args.data_dir)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    grids, summaries, pred_metrics_all, pred_all, selections, hotspot_all=[],[],[],[],[],[]
    raw_all,timing_all,neg_all,route_all,stab_all=[],[],[],[],[]
    data_grids={}
    for city in args.cities:
        LOG.info("Running %s",city)
        g,ts=load_city(args.data_dir,city)
        data_grids[city]=g
        summaries.append(dataset_summary(city,g,ts,args.n_routes))
        hotspot_all+=city_hotspot_stats(city,g)
        metrics,preds,select=forecast_city(city,ts,args.seed)
        pred_metrics_all+=metrics;pred_all+=preds;selections.append(select)
        raw,timing,neg,route,stab=run_routing(city,g,ts,preds,args.epsilons,args.n_routes,
            args.k_paths,args.seed,args.perturbations,args.perturb_sigma,args.neighbor_k,args.max_link_km)
        raw_all.append(raw);timing_all.append(timing);neg_all.append(neg);route_all.append(route)
        if not stab.empty:stab_all.append(stab)
    dataset=pd.DataFrame(summaries);pred=pd.DataFrame(pred_metrics_all);pred_daily=pd.DataFrame(pred_all)
    raw=pd.concat(raw_all,ignore_index=True);timing=pd.concat(timing_all,ignore_index=True)
    negotiation=pd.concat(neg_all,ignore_index=True);route_log=pd.concat(route_all,ignore_index=True)
    stability=pd.concat(stab_all,ignore_index=True) if stab_all else pd.DataFrame()
    summary=summarize_methods(raw)
    stats=paired_statistics(raw,args.primary_epsilon,args.seed,args.bootstrap,args.permutations)
    cross=cross_city_statistics(summary,args.primary_epsilon)
    feedback=network_feedback(summary,dataset,stats,stability,cross,args.primary_epsilon)
    audit=manuscript_audit(feedback)
    for name,df in [("dataset_summary.csv",dataset),("hotspot_results.csv",pd.DataFrame(hotspot_all)),
                    ("prediction_results.csv",pred),("daily_predictions.csv",pred_daily),
                    ("prediction_model_selection.csv",pd.DataFrame(selections)),
                    ("routing_raw_results.csv",raw[raw.group=="routing"]),
                    ("agent_ablation_raw_results.csv",raw[raw.group=="ablation"]),
                    ("routing_results.csv",summary[summary.group=="routing"]),
                    ("agent_ablation_results.csv",summary[summary.group=="ablation"]),
                    ("agent_negotiation_results.csv",negotiation),("timing_metrics.csv",timing),
                    ("selected_routes.csv",route_log),("route_stability_raw.csv",stability),
                    ("paired_statistical_tests.csv",stats),("cross_city_statistics.csv",cross),
                    ("network_feedback_report.csv",feedback),
                    ("manuscript_discrepancy_audit.csv",audit)]:
        save_csv(df,args.out_dir/name)
    if not args.skip_figures:
        make_figures(args.fig_dir,data_grids,pred,summary,stability,timing,args.primary_epsilon)
    latex_tables(dataset,summary,pred,feedback,stats,stability,cross,args.primary_epsilon,
                 args.out_dir/"paper_tables_generated.tex")
    validation=verify_results(raw,summary,timing,stability,stats,route_log,args,args.fig_dir)
    (args.out_dir/"validation_report.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    if validation["score_out_of_100"]<100:
        LOG.warning("Validation check score: %s/100. Open validation_report.json",validation["score_out_of_100"])
    settings={"cities":args.cities,"epsilons":args.epsilons,"primary_epsilon":args.primary_epsilon,
              "n_routes":args.n_routes,"k_paths":args.k_paths,"seed":args.seed,"bootstrap":args.bootstrap,
              "permutations":args.permutations,"perturbations":args.perturbations,
              "perturb_sigma":args.perturb_sigma,"neighbor_k":args.neighbor_k,
              "max_link_km":args.max_link_km,"runtime_seconds":time.perf_counter()-t0,
              "soft_policy_only":True,"reselection_shortest_if_violated":False,
              "route_cost_is_carbon_proxy_not_calibrated_CO2":True,
              "figure_directory":str(args.fig_dir),"experiment_directory":str(args.out_dir),
              "paper_report_values_are_not_assumed":True,"quick_check":args.quick_check}
    (args.out_dir/"run_manifest.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    (args.out_dir/"PAPER_RECONCILIATION_README.txt").write_text(
        "Reproducibility note: These are freshly computed metrics, NOT copied numbers.\n"
        "Compare manuscript_discrepancy_audit.csv with the manuscript. If values differ, "
        "replace manuscript results with paper_tables_generated.tex outputs or revise methods transparently.\n"
        "The manuscript's soft-policy argmax cannot guarantee zero violations.\n"
        "A route's carbon-cost proxy sums grid traffic; it is NOT a calibrated kg-CO2 estimate.\n"
        "Confidence intervals are paired percentile bootstrap and p-values are two-sided paired sign-flip tests.\n"
        "Route retention compares consecutive independent spatial perturbations of a fixed candidate set.\n"
        "This script does NOT implement multi-round decentralized negotiation, traffic-flow assignment, "
        "real road-network routing, genuine XGBoost routing, or independent cross-city transfer learning.\n"
        "The reported manuscript results may be impossible to exactly recover without original run logs.\n",encoding="utf-8")
    LOG.info("FINISHED in %.1f sec, tables in %s, figures in %s",settings["runtime_seconds"],args.out_dir,args.fig_dir)
    LOG.info("Manuscript audit agreement: %s/%s metrics within declared rounding",int(audit.agrees_within_rounding.sum()),len(audit))
    LOG.info("Structural consistency checks: %d/100",validation["score_out_of_100"])
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        LOG.exception("Experiment failed: %s",e)
        sys.exit(1)
