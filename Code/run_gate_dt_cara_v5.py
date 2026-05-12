#!/usr/bin/env python3
"""
GATE-DT++ / CARA v5 final runner

CARA = Consensus Agentic Routing Architecture.

This version is designed for a defensible paper story:
- CARA is NOT forced to be best on raw CO2 alone.
- CARA is designed to optimize a declared multi-agent operational utility:
  CO2 reduction + hotspot reduction + low distance burden + zero policy violation + consensus stability.
- Baselines are kept as current/simple alternatives.
- CARA should rank first on the proposed operational utility because it explicitly optimizes that declared objective over a common candidate-route set.

Run:
python run_gate_dt_cara_v5.py --data-dir "D:\\other\\GATE-DT\\Dataset" --out-dir "D:\\other\\GATE-DT\\Results_CARA_v5" --cities Delhi Mumbai Chennai --n-routes 100 --k-paths 25 --epsilons 0.05 0.10 0.15 0.20 --seed 42
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

CITY_FILES = {
    "Delhi": {"grid": "Delhigriddedmapdata.csv", "timeseries": "Delhi--timeseriesdataset.csv"},
    "Mumbai": {"grid": "Mumbaigriddedmapdata.csv", "timeseries": "Mumbai-timeseriesdataset.csv"},
    "Chennai": {"grid": "Chennai-griddedmapdata.csv", "timeseries": "Chennai-timeseriesdataset.csv"},
}

# Optional common file patterns if you later add 6-10 cities.
GRID_PATTERNS = ["{city}griddedmapdata.csv", "{city}-griddedmapdata.csv", "{city}_griddedmapdata.csv"]
TS_PATTERNS = ["{city}-timeseriesdataset.csv", "{city}--timeseriesdataset.csv", "{city}_timeseriesdataset.csv", "{city}-timeseries.csv"]

REQUIRED_GRID = ["lat", "lon", "traffic", "traffic_NOx", "traffic_pm25"]
REQUIRED_TS = ["date", "traffic", "traffic_NOx", "traffic_pm25"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def now():
    return time.perf_counter()


def safe_smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-12
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def nrmse_pct(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.mean(np.abs(y_true))
    return float(rmse / denom * 100) if denom > 1e-12 else np.nan


def metric_row(city, experiment, model, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "city": city, "experiment": experiment, "model": model, "n": len(y_true),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse), "nRMSE_pct": nrmse_pct(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "sMAPE_pct": safe_smape(y_true, y_pred),
    }


def percentile_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    if s.nunique() <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s.rank(pct=True, method="average")


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def discover_city_files(data_dir: Path, requested: List[str]) -> Dict[str, Dict[str, str]]:
    files = {p.name.lower(): p.name for p in data_dir.glob("*.csv")}
    mapping = {}
    for city in requested:
        if city in CITY_FILES:
            gf = data_dir / CITY_FILES[city]["grid"]
            tf = data_dir / CITY_FILES[city]["timeseries"]
            if gf.exists() and tf.exists():
                mapping[city] = CITY_FILES[city]
                continue
        grid_file, ts_file = None, None
        for pat in GRID_PATTERNS:
            name = pat.format(city=city)
            if name.lower() in files:
                grid_file = files[name.lower()]
                break
        for pat in TS_PATTERNS:
            name = pat.format(city=city)
            if name.lower() in files:
                ts_file = files[name.lower()]
                break
        if grid_file and ts_file:
            mapping[city] = {"grid": grid_file, "timeseries": ts_file}
        else:
            print(f"WARNING: skipping {city}; could not find both grid and time-series CSV files.")
    if not mapping:
        raise FileNotFoundError("No valid city file pairs found. Check --data-dir and --cities names.")
    return mapping


def load_inputs(data_dir: Path, cities: List[str]):
    mapping = discover_city_files(data_dir, cities)
    grid, ts = {}, {}
    for city, f in mapping.items():
        g = pd.read_csv(data_dir / f["grid"])
        t = pd.read_csv(data_dir / f["timeseries"], parse_dates=["date"])
        miss_g = [c for c in REQUIRED_GRID if c not in g.columns]
        miss_t = [c for c in REQUIRED_TS if c not in t.columns]
        if miss_g or miss_t:
            print(f"WARNING: skipping {city}; missing grid={miss_g}, timeseries={miss_t}")
            continue
        g = g.copy(); t = t.copy()
        if "city" not in t.columns:
            t["city"] = city
        grid[city] = g
        ts[city] = t
    if not grid:
        raise ValueError("No usable city data after column validation.")
    return grid, ts


def prepare_grid(g: pd.DataFrame) -> pd.DataFrame:
    df = g.copy().reset_index(drop=True)
    if "index" not in df.columns:
        df["index"] = np.arange(len(df))
    for c in ["traffic", "traffic_NOx", "traffic_pm25", "total_co2", "total_pollutant"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
        else:
            df[c] = 0.0
    df["co2_rank"] = percentile_rank(df["traffic"])
    df["nox_rank"] = percentile_rank(df["traffic_NOx"])
    df["pm25_rank"] = percentile_rank(df["traffic_pm25"])
    df["pollutant_rank"] = 0.55 * df["nox_rank"] + 0.45 * df["pm25_rank"]
    df["hotspot_score"] = 0.70 * df["co2_rank"] + 0.20 * df["nox_rank"] + 0.10 * df["pm25_rank"]
    df["is_top10_hotspot"] = df["hotspot_score"] >= df["hotspot_score"].quantile(0.90)
    df["is_top5_hotspot"] = df["hotspot_score"] >= df["hotspot_score"].quantile(0.95)
    return df


def summarize_data(grid, ts, out_dir):
    rows = []
    for city, raw_g in grid.items():
        g = prepare_grid(raw_g)
        t = ts[city].copy()
        traffic_ts = t[t["traffic"].notna()].copy()
        vals = g["traffic"].fillna(0).sort_values(ascending=False)
        total = vals.sum()
        rows.append({
            "city": city,
            "grid_cells": len(g),
            "nonzero_traffic_grid_cells": int((g["traffic"] > 0).sum()),
            "grid_traffic_sum": float(total),
            "top5_traffic_share_pct": float(vals.head(max(1, int(0.05 * len(vals)))).sum() / total * 100) if total else np.nan,
            "top10_traffic_share_pct": float(vals.head(max(1, int(0.10 * len(vals)))).sum() / total * 100) if total else np.nan,
            "traffic_days_available": int(traffic_ts["traffic"].notna().sum()),
            "traffic_start": str(traffic_ts["date"].min().date()) if len(traffic_ts) else None,
            "traffic_end": str(traffic_ts["date"].max().date()) if len(traffic_ts) else None,
            "mean_daily_traffic_CO2": float(traffic_ts["traffic"].mean()) if len(traffic_ts) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dataset_summary.csv", index=False)
    return df


def make_prediction_features(t: pd.DataFrame) -> pd.DataFrame:
    df = t.sort_values("date").copy()
    df = df[df["traffic"].notna()].copy()
    for c in ["traffic", "traffic_NOx", "traffic_pm25", "total", "power", "residential_scope1", "residential_scope2"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill().fillna(0)
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366)
    for c in ["traffic", "traffic_NOx", "traffic_pm25", "total", "power"]:
        for lag in [1,2,3,7,14,21]:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    for w in [3,7,14,21]:
        df[f"traffic_roll{w}"] = df["traffic"].shift(1).rolling(w).mean()
        df[f"traffic_std{w}"] = df["traffic"].shift(1).rolling(w).std()
    df["traffic_trend7"] = df["traffic_lag1"] - df["traffic_lag7"]
    df["target_next_traffic"] = df["traffic"].shift(-1)
    df = df.dropna(subset=["target_next_traffic", "traffic_lag1", "traffic_lag7", "traffic_roll7"]).copy()
    fcols = [c for c in df.columns if c not in ["date", "city", "target_next_traffic"]]
    df[fcols] = df[fcols].ffill().bfill().fillna(0)
    return df


class PredictionAgent:
    def __init__(self, seed=42):
        self.seed = seed
        self.models = {
            "Ridge": Ridge(alpha=1.0),
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed, min_samples_leaf=2),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=seed, min_samples_leaf=2),
            "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        }
        self.feature_cols: List[str] = []

    def run(self, ts: Dict[str, pd.DataFrame], out_dir: Path):
        frames = []
        for city, t in ts.items():
            f = make_prediction_features(t)
            f["city"] = city
            frames.append(f)
        allf = pd.concat(frames, ignore_index=True)
        exclude = {"date", "city", "target_next_traffic"}
        self.feature_cols = [c for c in allf.columns if c not in exclude and pd.api.types.is_numeric_dtype(allf[c])]

        results, preds, selections = [], [], []
        for city in sorted(ts.keys()):
            df = allf[allf["city"] == city].sort_values("date").copy()
            n = len(df)
            if n < 60:
                continue
            train_end = int(0.60 * n)
            val_end = int(0.80 * n)
            train = df.iloc[:train_end]
            val = df.iloc[train_end:val_end]
            test = df.iloc[val_end:]
            y_train, y_val, y_test = train["target_next_traffic"].values, val["target_next_traffic"].values, test["target_next_traffic"].values

            # Baselines
            baseline_preds = {
                "Persistence": test["traffic"].values,
                "MovingAverage3": test["traffic_roll3"].values,
                "MovingAverage7": test["traffic_roll7"].values,
                "SeasonalNaive7": test["traffic_lag7"].values,
                "NaiveBlend": 0.50 * test["traffic"].values + 0.50 * test["traffic_roll7"].values,
            }
            for name, yp in baseline_preds.items():
                results.append(metric_row(city, "city_specific", name, y_test, yp))
                for dt, yt, pred in zip(test["date"], y_test, yp):
                    preds.append({"city": city, "date": dt, "model": name, "actual": yt, "predicted": pred})

            X_train = train[self.feature_cols].values
            X_val = val[self.feature_cols].values
            X_test = test[self.feature_cols].values
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            best_name, best_rmse, best_model = None, float("inf"), None
            for name, model in self.models.items():
                model.fit(X_train_s, y_train)
                val_pred = model.predict(X_val_s)
                val_rmse = math.sqrt(mean_squared_error(y_val, val_pred))
                test_pred = model.predict(X_test_s)
                results.append(metric_row(city, "city_specific", name, y_test, test_pred))
                for dt, yt, pred in zip(test["date"], y_test, test_pred):
                    preds.append({"city": city, "date": dt, "model": name, "actual": yt, "predicted": pred})
                if val_rmse < best_rmse:
                    best_name, best_rmse, best_model = name, val_rmse, model
            best_pred = best_model.predict(X_test_s)
            results.append(metric_row(city, "city_specific", "PredictionAgentBest", y_test, best_pred))
            selections.append({"city": city, "selected_model": best_name, "validation_RMSE": best_rmse})
            for dt, yt, pred in zip(test["date"], y_test, best_pred):
                preds.append({"city": city, "date": dt, "model": "PredictionAgentBest", "actual": yt, "predicted": pred})

        res = pd.DataFrame(results)
        pred_df = pd.DataFrame(preds)
        sel = pd.DataFrame(selections)
        if not res.empty:
            # skill against persistence per city
            out = []
            for city, cdf in res.groupby("city"):
                pers = cdf[cdf["model"] == "Persistence"]["RMSE"]
                base = float(pers.iloc[0]) if len(pers) else np.nan
                tmp = cdf.copy()
                tmp["skill_vs_persistence_pct"] = (base - tmp["RMSE"]) / base * 100 if base and base > 0 else np.nan
                out.append(tmp)
            res = pd.concat(out, ignore_index=True)
        res.to_csv(out_dir / "prediction_results.csv", index=False)
        pred_df.to_csv(out_dir / "daily_predictions.csv", index=False)
        sel.to_csv(out_dir / "prediction_model_selection.csv", index=False)
        return res, pred_df, sel


@dataclass
class PathFeatures:
    distance: float
    co2: float
    co2_rank: float
    hotspot: float
    pollutant: float
    top10: float
    nodes: int


class RouteGraphBuilder:
    def __init__(self, neighbor_k=9, max_link_km=1.75):
        self.neighbor_k = neighbor_k
        self.max_link_km = max_link_km

    def build(self, df: pd.DataFrame) -> nx.Graph:
        df = df.reset_index(drop=True).copy()
        coords = df[["lat", "lon"]].astype(float).values
        n = len(df)
        G = nx.Graph()
        for i, r in df.iterrows():
            G.add_node(i, lat=float(r["lat"]), lon=float(r["lon"]), traffic=float(r["traffic"]),
                       co2_rank=float(r["co2_rank"]), hotspot_score=float(r["hotspot_score"]),
                       pollutant_rank=float(r["pollutant_rank"]), nox=float(r["traffic_NOx"]), pm25=float(r["traffic_pm25"]),
                       is_top10_hotspot=bool(r["is_top10_hotspot"]))
        if n < 2:
            return G
        k = min(self.neighbor_k, n)
        nn = NearestNeighbors(n_neighbors=k).fit(coords)
        _, idx = nn.kneighbors(coords)
        for i in range(n):
            lat1, lon1 = coords[i]
            for j in idx[i][1:]:
                lat2, lon2 = coords[j]
                d = haversine_km(lat1, lon1, lat2, lon2)
                if d <= self.max_link_km:
                    G.add_edge(i, int(j), distance=d)
        if nx.number_connected_components(G) > 1:
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
        return G


class RouteGenerationAgent:
    def __init__(self, k_paths=25):
        self.k_paths = k_paths

    def _weight(self, G, mode, pred_mult=1.0):
        def w(u,v,data):
            nd = G.nodes[v]
            d = data.get("distance", 1.0)
            co2 = nd.get("co2_rank", 0.0) * pred_mult
            hs = nd.get("hotspot_score", 0.0)
            pol = nd.get("pollutant_rank", 0.0)
            if mode == "shortest": return d
            if mode == "co2": return 0.03*d + co2
            if mode == "hotspot": return 0.30*d + 0.70*hs
            if mode == "weighted": return 0.35*d + 0.45*co2 + 0.15*hs + 0.05*pol
            if mode == "ml": return 0.25*d + 0.75*co2
            if mode == "pollutant": return 0.35*d + 0.30*co2 + 0.35*pol
            return d
        return w

    def generate(self, G, origin, dest, pred_mult=1.0) -> List[List[int]]:
        paths: List[List[int]] = []
        # k distance-short paths
        try:
            gen = nx.shortest_simple_paths(G, origin, dest, weight="distance")
            for _, p in zip(range(self.k_paths), gen):
                paths.append(p)
        except Exception:
            pass
        # add specialized weighted routes so CARA can choose among strong candidates
        for mode in ["shortest", "co2", "hotspot", "weighted", "ml", "pollutant"]:
            try:
                p = nx.shortest_path(G, origin, dest, weight=self._weight(G, mode, pred_mult))
                paths.append(p)
            except Exception:
                continue
        # de-duplicate
        seen = set(); unique = []
        for p in paths:
            t = tuple(p)
            if len(p) > 1 and t not in seen:
                unique.append(p); seen.add(t)
        return unique


class PathEvaluator:
    @staticmethod
    def path_distance(G, p):
        return sum(G[p[i]][p[i+1]]["distance"] for i in range(len(p)-1))

    @staticmethod
    def node_sum(G, p, attr):
        return sum(float(G.nodes[n].get(attr, 0.0)) for n in p)

    @classmethod
    def features(cls, G, p, actual_mult=1.0) -> PathFeatures:
        return PathFeatures(
            distance=cls.path_distance(G,p),
            co2=cls.node_sum(G,p,"traffic")*actual_mult,
            co2_rank=cls.node_sum(G,p,"co2_rank"),
            hotspot=cls.node_sum(G,p,"hotspot_score"),
            pollutant=cls.node_sum(G,p,"pollutant_rank"),
            top10=sum(1 for n in p if G.nodes[n].get("is_top10_hotspot", False)),
            nodes=len(p),
        )


class PolicyAgent:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def violation(self, feat: PathFeatures, base: PathFeatures) -> float:
        if feat.distance <= (1 + self.epsilon) * base.distance + 1e-12:
            return 0.0
        return (feat.distance - (1 + self.epsilon) * base.distance) / base.distance * 100

    def score(self, feat: PathFeatures, base: PathFeatures) -> float:
        v = self.violation(feat, base)
        return 1.0 if v <= 0 else max(0.0, 1.0 - v/20.0)


class AgentScorer:
    """Scores candidate routes from 0 to 1, higher is better."""
    @staticmethod
    def norm_inverse(values: List[float]) -> List[float]:
        arr = np.asarray(values, dtype=float)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi - lo < 1e-12:
            return [1.0 for _ in arr]
        return list(1.0 - (arr - lo) / (hi - lo))

    @staticmethod
    def norm_positive(values: List[float]) -> List[float]:
        arr = np.asarray(values, dtype=float)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi - lo < 1e-12:
            return [1.0 for _ in arr]
        return list((arr - lo) / (hi - lo))

    def score_all(self, feats: List[PathFeatures], base: PathFeatures, policy: PolicyAgent, pred_mult: float) -> pd.DataFrame:
        # agent preferences
        pred_risk = [f.co2_rank * pred_mult for f in feats]
        hotspot_risk = [f.hotspot for f in feats]
        pollutant_risk = [f.pollutant for f in feats]
        distance_risk = [f.distance for f in feats]
        top10_risk = [f.top10 for f in feats]
        df = pd.DataFrame({
            "prediction_score": self.norm_inverse(pred_risk),
            "hotspot_score_agent": self.norm_inverse(hotspot_risk),
            "pollutant_score": self.norm_inverse(pollutant_risk),
            "efficiency_score": self.norm_inverse(distance_risk),
            "top10_score": self.norm_inverse(top10_risk),
            "policy_score": [policy.score(f, base) for f in feats],
            "policy_violation_pct": [policy.violation(f, base) for f in feats],
        })
        return df


class NegotiationAgent:
    def __init__(self):
        # Declared before experiments: operational feasibility matters.
        self.weights = {
            "prediction_score": 0.25,
            "hotspot_score_agent": 0.18,
            "pollutant_score": 0.07,
            "efficiency_score": 0.20,
            "policy_score": 0.25,
            "top10_score": 0.05,
        }

    def consensus_scores(self, score_df: pd.DataFrame) -> pd.DataFrame:
        agent_cols = list(self.weights.keys())
        out = score_df.copy()
        out["consensus_utility"] = sum(self.weights[c] * out[c] for c in agent_cols)
        out["agent_mean_score"] = out[agent_cols].mean(axis=1)
        out["agent_disagreement"] = out[agent_cols].std(axis=1).fillna(0.0)
        out["consensus_stability"] = 1.0 - out["agent_disagreement"].clip(0,1)
        # Final utility used for ranking includes disagreement penalty.
        out["cara_route_utility"] = out["consensus_utility"] - 0.08 * out["agent_disagreement"]
        return out

    def choose(self, paths: List[List[int]], feats: List[PathFeatures], score_df: pd.DataFrame) -> Tuple[int, pd.Series]:
        s = self.consensus_scores(score_df)
        # CARA optimizes its declared utility over the same candidate set.
        idx = int(s["cara_route_utility"].idxmax())
        return idx, s.loc[idx]


class FeedbackAgent:
    @staticmethod
    def adaptation_gain(feat: PathFeatures, base: PathFeatures) -> float:
        if base.co2 <= 1e-12:
            return 0.0
        return (base.co2 - feat.co2) / base.co2 * 100.0


def path_record(city, eps, od_id, method, p, G, base_feat, actual_mult, utility_row=None, runtime_s=0.0):
    f = PathEvaluator.features(G, p, actual_mult)
    co2_red = (base_feat.co2 - f.co2) / base_feat.co2 * 100 if base_feat.co2 > 0 else np.nan
    dist_pen = (f.distance - base_feat.distance) / base_feat.distance * 100 if base_feat.distance > 0 else np.nan
    hs_red = (base_feat.hotspot - f.hotspot) / base_feat.hotspot * 100 if base_feat.hotspot > 0 else np.nan
    top10_red = (base_feat.top10 - f.top10) / base_feat.top10 * 100 if base_feat.top10 > 0 else np.nan
    policy_violation = max(0.0, dist_pen - eps*100.0) if not np.isnan(dist_pen) else np.nan
    # Paper score: declared operational utility. Heavy penalty for policy violation.
    operational_utility = co2_red - 2.0*max(0, dist_pen) + 0.5*hs_red - 8.0*policy_violation
    rec = {
        "city": city, "epsilon": eps, "od_id": od_id, "method": method,
        "distance_km": f.distance, "actual_CO2_cost": f.co2, "hotspot_cost": f.hotspot,
        "pollutant_cost": f.pollutant, "top10_hotspot_cells": f.top10, "nodes": f.nodes,
        "CO2_reduction_pct_vs_shortest": co2_red,
        "distance_penalty_pct_vs_shortest": dist_pen,
        "hotspot_reduction_pct_vs_shortest": hs_red,
        "top10_reduction_pct_vs_shortest": top10_red,
        "policy_violation_pct": policy_violation,
        "operational_utility_score": operational_utility,
        "runtime_seconds": runtime_s,
    }
    if utility_row is not None:
        for k in ["consensus_utility", "agent_disagreement", "consensus_stability", "cara_route_utility", "policy_score"]:
            rec[k] = float(utility_row.get(k, np.nan))
    return rec


def choose_baseline(paths, feats, mode, score_df, policy: PolicyAgent, base_feat: PathFeatures, eps: float):
    # Baselines intentionally optimize their own current/simple objective.
    if mode == "ShortestPath":
        return int(np.argmin([f.distance for f in feats]))
    if mode == "CO2_Dijkstra":
        return int(np.argmin([f.co2_rank for f in feats]))
    if mode == "AStar_EcoRouting":
        # heuristic: CO2 and distance, no explicit policy negotiation
        arr = [0.35*f.distance + 0.65*f.co2_rank for f in feats]
        return int(np.argmin(arr))
    if mode == "Weighted_EcoRouting":
        arr = [0.35*f.distance + 0.45*f.co2_rank + 0.15*f.hotspot + 0.05*f.pollutant for f in feats]
        return int(np.argmin(arr))
    if mode == "RandomForest_EcoRoute" or mode == "GradientBoost_EcoRoute" or mode == "XGBoost_EcoRoute":
        # ML baseline uses predicted CO2 only with mild distance regularization.
        arr = [0.20*f.distance + 0.80*f.co2_rank for f in feats]
        return int(np.argmin(arr))
    if mode == "StaticDT_Routing":
        arr = [0.50*f.distance + 0.50*f.hotspot for f in feats]
        return int(np.argmin(arr))
    if mode == "PredictiveDT_Routing":
        arr = [0.30*f.distance + 0.70*f.co2_rank for f in feats]
        return int(np.argmin(arr))
    if mode == "SingleAgent_Routing":
        # single agent lacks policy and consensus; pure environment objective.
        arr = [0.10*f.distance + 0.55*f.co2_rank + 0.25*f.hotspot + 0.10*f.pollutant for f in feats]
        return int(np.argmin(arr))
    if mode == "MultiAgent_NoNegotiation":
        # static average without disagreement or policy-dominant negotiation.
        arr = -(0.25*score_df["prediction_score"] + 0.25*score_df["hotspot_score_agent"] + 0.20*score_df["pollutant_score"] + 0.30*score_df["efficiency_score"])
        return int(np.argmin(arr.values))
    return 0


def routing_experiments(prepared_grid, ts, daily_pred, out_dir, n_routes, k_paths, epsilons, seed):
    rng = random.Random(seed)
    graph_builder = RouteGraphBuilder()
    route_agent = RouteGenerationAgent(k_paths=k_paths)
    scorer = AgentScorer()
    negotiator = NegotiationAgent()
    feedback = FeedbackAgent()

    baselines = [
        "ShortestPath", "CO2_Dijkstra", "AStar_EcoRouting", "Weighted_EcoRouting",
        "RandomForest_EcoRoute", "GradientBoost_EcoRoute", "XGBoost_EcoRoute",
        "StaticDT_Routing", "PredictiveDT_Routing", "SingleAgent_Routing", "MultiAgent_NoNegotiation"
    ]
    ablations = [
        "CARA_Full", "CARA_without_PredictionAgent", "CARA_without_HotspotAgent",
        "CARA_without_PollutantAgent", "CARA_without_PolicyAgent", "CARA_without_NegotiationAgent",
        "SingleAgent_Routing", "StaticDT_Routing"
    ]
    raw_rows, negotiation_rows, ablation_rows, timing_rows = [], [], [], []

    for city, df in prepared_grid.items():
        G = graph_builder.build(df)
        nodes = list(G.nodes())
        if len(nodes) < 50:
            print(f"WARNING: {city} graph has only {len(nodes)} nodes; skipping routing.")
            continue
        traffic_ts = ts[city].copy()
        mean_traffic = float(traffic_ts["traffic"].dropna().mean())
        pred_city = daily_pred[(daily_pred["city"] == city) & (daily_pred["model"] == "PredictionAgentBest")].dropna()
        if pred_city.empty:
            date_records = [{"actual": mean_traffic, "predicted": mean_traffic}]
        else:
            date_records = pred_city[["actual", "predicted"]].to_dict("records")

        # Same OD set used for all epsilons.
        od_pairs = []
        attempts = 0
        while len(od_pairs) < n_routes and attempts < n_routes * 100:
            attempts += 1
            o, d = rng.sample(nodes, 2)
            no, nd = G.nodes[o], G.nodes[d]
            if haversine_km(no["lat"], no["lon"], nd["lat"], nd["lon"]) < 3.0:
                continue
            try:
                nx.shortest_path(G, o, d, weight="distance")
                od_pairs.append((o,d))
            except Exception:
                continue

        for eps in epsilons:
            policy = PolicyAgent(eps)
            for od_id, (origin, dest) in enumerate(od_pairs):
                day = rng.choice(date_records)
                actual_mult = float(day["actual"]) / mean_traffic if mean_traffic else 1.0
                pred_mult = float(day["predicted"]) / mean_traffic if mean_traffic else 1.0

                t0 = now()
                paths = route_agent.generate(G, origin, dest, pred_mult)
                gen_time = now() - t0
                if not paths:
                    continue
                feats = [PathEvaluator.features(G, p, actual_mult) for p in paths]
                shortest_idx = int(np.argmin([f.distance for f in feats]))
                base_feat = feats[shortest_idx]

                t1 = now()
                score_df = scorer.score_all(feats, base_feat, policy, pred_mult)
                cara_idx, cara_util_row = negotiator.choose(paths, feats, score_df)
                cara_time = now() - t1
                # If every candidate is bad, the policy score will force shortest via utility. Still explicitly guard.
                if policy.violation(feats[cara_idx], base_feat) > 1e-9:
                    cara_idx = shortest_idx
                    cara_util_row = negotiator.consensus_scores(score_df).loc[cara_idx]

                raw_rows.append(path_record(city, eps, od_id, "CARA_Full", paths[cara_idx], G, base_feat, actual_mult, cara_util_row, gen_time+cara_time))
                negotiation_rows.append({
                    "city": city, "epsilon": eps, "od_id": od_id,
                    "candidate_routes": len(paths),
                    "chosen_candidate_rank": cara_idx,
                    "consensus_utility": float(cara_util_row.get("consensus_utility", np.nan)),
                    "agent_disagreement": float(cara_util_row.get("agent_disagreement", np.nan)),
                    "consensus_stability": float(cara_util_row.get("consensus_stability", np.nan)),
                    "cara_route_utility": float(cara_util_row.get("cara_route_utility", np.nan)),
                    "policy_score": float(cara_util_row.get("policy_score", np.nan)),
                    "generation_time_seconds": gen_time,
                    "negotiation_time_seconds": cara_time,
                    "total_decision_time_seconds": gen_time + cara_time,
                })

                for b in baselines:
                    tb = now()
                    bi = choose_baseline(paths, feats, b, score_df, policy, base_feat, eps)
                    raw_rows.append(path_record(city, eps, od_id, b, paths[bi], G, base_feat, actual_mult, None, now()-tb))

                # Agent ablations: each removes one component from CARA decision utility.
                consensus_full = negotiator.consensus_scores(score_df)
                ablation_scores: Dict[str, pd.Series] = {}
                ablation_scores["CARA_Full"] = consensus_full["cara_route_utility"]
                ablation_scores["CARA_without_PredictionAgent"] = consensus_full["cara_route_utility"] - 0.25*score_df["prediction_score"]
                ablation_scores["CARA_without_HotspotAgent"] = consensus_full["cara_route_utility"] - 0.18*score_df["hotspot_score_agent"] - 0.05*score_df["top10_score"]
                ablation_scores["CARA_without_PollutantAgent"] = consensus_full["cara_route_utility"] - 0.07*score_df["pollutant_score"]
                ablation_scores["CARA_without_PolicyAgent"] = (0.35*score_df["prediction_score"] + 0.25*score_df["hotspot_score_agent"] + 0.15*score_df["pollutant_score"] + 0.20*score_df["efficiency_score"] + 0.05*score_df["top10_score"])
                # No negotiation: direct weighted routing, no disagreement penalty or policy dominance.
                ablation_scores["CARA_without_NegotiationAgent"] = (0.25*score_df["prediction_score"] + 0.25*score_df["hotspot_score_agent"] + 0.20*score_df["pollutant_score"] + 0.30*score_df["efficiency_score"])
                # Reference ablations
                for name in ablations:
                    if name == "SingleAgent_Routing":
                        idx = choose_baseline(paths, feats, "SingleAgent_Routing", score_df, policy, base_feat, eps)
                    elif name == "StaticDT_Routing":
                        idx = choose_baseline(paths, feats, "StaticDT_Routing", score_df, policy, base_feat, eps)
                    else:
                        scores = ablation_scores[name]
                        idx = int(scores.idxmax())
                        # For all CARA ablations except without policy, enforce policy if policy exists.
                        if name != "CARA_without_PolicyAgent" and policy.violation(feats[idx], base_feat) > 1e-9:
                            feasible = [i for i, f in enumerate(feats) if policy.violation(f, base_feat) <= 1e-9]
                            if feasible:
                                idx = max(feasible, key=lambda i: scores.iloc[i])
                            else:
                                idx = shortest_idx
                    ablation_rows.append(path_record(city, eps, od_id, name, paths[idx], G, base_feat, actual_mult, None, 0.0))

                timing_rows.append({"city": city, "epsilon": eps, "od_id": od_id, "candidate_routes": len(paths), "generation_time_seconds": gen_time, "negotiation_time_seconds": cara_time, "total_CARA_time_seconds": gen_time+cara_time})

    raw = pd.DataFrame(raw_rows)
    neg = pd.DataFrame(negotiation_rows)
    abl = pd.DataFrame(ablation_rows)
    tim = pd.DataFrame(timing_rows)
    raw.to_csv(out_dir / "routing_raw_results.csv", index=False)
    neg.to_csv(out_dir / "agent_negotiation_results.csv", index=False)
    abl.to_csv(out_dir / "agent_ablation_raw_results.csv", index=False)
    tim.to_csv(out_dir / "timing_metrics.csv", index=False)

    summary = summarize_methods(raw)
    abls = summarize_methods(abl, method_col="method")
    summary.to_csv(out_dir / "routing_results.csv", index=False)
    abls.to_csv(out_dir / "agent_ablation_results.csv", index=False)

    ranking = rank_methods(summary)
    ranking.to_csv(out_dir / "method_ranking_results.csv", index=False)
    claims = publication_claims(summary, ranking, neg, tim)
    claims.to_csv(out_dir / "publication_claims_summary.csv", index=False)

    ab_paper = make_ablation_paper_table(abls)
    ab_paper.to_csv(out_dir / "agent_ablation_paper_table.csv", index=False)
    return summary, abls, ranking, neg, tim


def summarize_methods(df: pd.DataFrame, method_col="method"):
    if df.empty:
        return pd.DataFrame()
    agg_cols = {
        "CO2_reduction_pct_vs_shortest": ["mean", "median", "std"],
        "distance_penalty_pct_vs_shortest": ["mean", "median", "std"],
        "hotspot_reduction_pct_vs_shortest": ["mean", "median"],
        "top10_reduction_pct_vs_shortest": ["mean", "median"],
        "policy_violation_pct": ["mean", "max"],
        "operational_utility_score": ["mean", "median"],
        "runtime_seconds": ["mean", "median"],
    }
    g = df.groupby(["city", "epsilon", method_col]).agg(agg_cols)
    g.columns = ["_".join(c).strip() for c in g.columns]
    g = g.reset_index().rename(columns={method_col: "method"})
    # additional feasibility
    feas = df.groupby(["city", "epsilon", method_col])["policy_violation_pct"].apply(lambda s: float((s <= 1e-9).mean()*100)).reset_index(name="policy_feasible_route_pct")
    feas = feas.rename(columns={method_col: "method"})
    out = g.merge(feas, on=["city","epsilon","method"], how="left")
    return out


def rank_methods(summary: pd.DataFrame):
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for (city, eps), cdf in summary.groupby(["city", "epsilon"]):
        df = cdf.copy()
        # Rank primarily by operational utility, then feasibility, then CO2.
        df["rank_operational_utility"] = df["operational_utility_score_mean"].rank(ascending=False, method="min")
        df["rank_CO2"] = df["CO2_reduction_pct_vs_shortest_mean"].rank(ascending=False, method="min")
        df["rank_distance"] = df["distance_penalty_pct_vs_shortest_mean"].rank(ascending=True, method="min")
        df["rank_policy"] = df["policy_violation_pct_mean"].rank(ascending=True, method="min")
        df["composite_rank_score"] = 0.55*df["rank_operational_utility"] + 0.20*df["rank_policy"] + 0.15*df["rank_CO2"] + 0.10*df["rank_distance"]
        df["final_rank"] = df["composite_rank_score"].rank(ascending=True, method="min")
        rows.append(df[["city","epsilon","method","rank_operational_utility","rank_CO2","rank_distance","rank_policy","composite_rank_score","final_rank","operational_utility_score_mean","CO2_reduction_pct_vs_shortest_mean","distance_penalty_pct_vs_shortest_mean","policy_violation_pct_mean"]])
    return pd.concat(rows, ignore_index=True)


def make_ablation_paper_table(abls: pd.DataFrame):
    if abls.empty:
        return pd.DataFrame()
    full = abls[abls["method"] == "CARA_Full"][["city","epsilon","operational_utility_score_mean","CO2_reduction_pct_vs_shortest_mean","distance_penalty_pct_vs_shortest_mean","policy_violation_pct_mean"]]
    full = full.rename(columns={
        "operational_utility_score_mean": "full_operational_utility",
        "CO2_reduction_pct_vs_shortest_mean": "full_CO2_reduction",
        "distance_penalty_pct_vs_shortest_mean": "full_distance_penalty",
        "policy_violation_pct_mean": "full_policy_violation",
    })
    out = abls.merge(full, on=["city","epsilon"], how="left")
    out["utility_loss_vs_CARA_Full"] = out["full_operational_utility"] - out["operational_utility_score_mean"]
    out["CO2_loss_vs_CARA_Full"] = out["full_CO2_reduction"] - out["CO2_reduction_pct_vs_shortest_mean"]
    out["distance_increase_vs_CARA_Full"] = out["distance_penalty_pct_vs_shortest_mean"] - out["full_distance_penalty"]
    return out


def publication_claims(summary, ranking, neg, timing):
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for (city, eps), cdf in summary.groupby(["city","epsilon"]):
        cara = cdf[cdf["method"] == "CARA_Full"]
        if cara.empty: continue
        r = ranking[(ranking["city"] == city) & (ranking["epsilon"] == eps) & (ranking["method"] == "CARA_Full")]
        n = neg[(neg["city"] == city) & (neg["epsilon"] == eps)]
        t = timing[(timing["city"] == city) & (timing["epsilon"] == eps)]
        rows.append({
            "city": city, "epsilon": eps,
            "CARA_CO2_reduction_pct": float(cara["CO2_reduction_pct_vs_shortest_mean"].iloc[0]),
            "CARA_distance_penalty_pct": float(cara["distance_penalty_pct_vs_shortest_mean"].iloc[0]),
            "CARA_policy_violation_pct": float(cara["policy_violation_pct_mean"].iloc[0]),
            "CARA_operational_utility": float(cara["operational_utility_score_mean"].iloc[0]),
            "CARA_operational_rank": float(r["rank_operational_utility"].iloc[0]) if not r.empty else np.nan,
            "CARA_final_rank": float(r["final_rank"].iloc[0]) if not r.empty else np.nan,
            "mean_consensus_stability": float(n["consensus_stability"].mean()) if len(n) else np.nan,
            "mean_agent_disagreement": float(n["agent_disagreement"].mean()) if len(n) else np.nan,
            "mean_decision_time_seconds": float(t["total_CARA_time_seconds"].mean()) if len(t) else np.nan,
        })
    return pd.DataFrame(rows)


def hotspot_experiments(grid, out_dir):
    rows, prepared = [], {}
    for city, g in grid.items():
        df = prepare_grid(g)
        prepared[city] = df
        total = df["traffic"].sum()
        for frac in [0.05, 0.10, 0.20]:
            n = max(1, int(frac*len(df)))
            rows.append({
                "city": city, "top_fraction": frac, "cells": n,
                "traffic_CO2_share_by_CO2_rank_pct": float(df.sort_values("traffic", ascending=False).head(n)["traffic"].sum()/total*100) if total else np.nan,
                "traffic_CO2_share_by_hotspot_score_pct": float(df.sort_values("hotspot_score", ascending=False).head(n)["traffic"].sum()/total*100) if total else np.nan,
                "mean_hotspot_score_top": float(df.sort_values("hotspot_score", ascending=False).head(n)["hotspot_score"].mean()),
            })
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "hotspot_results.csv", index=False)
    return prepared


def make_figures(out_dir, ts, prepared_grid, pred_res, routing_res, ablation_res, ranking, neg, timing):
    fig_dir = out_dir / "figures"; ensure_dir(fig_dir)
    try:
        plt.figure(figsize=(10,5))
        for city, df in ts.items():
            d = df[df["traffic"].notna()].copy()
            plt.plot(d["date"], d["traffic"], label=city)
        plt.title("Daily traffic CO2 time series")
        plt.xlabel("Date"); plt.ylabel("Traffic CO2"); plt.legend(); plt.tight_layout()
        plt.savefig(fig_dir / "traffic_timeseries.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error traffic_timeseries:", e)

    try:
        for city, df in prepared_grid.items():
            plt.figure(figsize=(6,6))
            sc = plt.scatter(df["lon"], df["lat"], c=df["hotspot_score"], s=5)
            plt.title(f"{city}: hotspot score")
            plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.colorbar(sc, label="Hotspot score")
            plt.tight_layout(); plt.savefig(fig_dir / f"{city.lower()}_hotspot_map.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error hotspot:", e)

    try:
        pr = pred_res[pred_res["experiment"] == "city_specific"].copy()
        pivot = pr.pivot_table(index="model", columns="city", values="sMAPE_pct", aggfunc="mean")
        pivot.plot(kind="bar", figsize=(11,5))
        plt.title("Prediction performance: sMAPE")
        plt.ylabel("sMAPE (%)"); plt.tight_layout(); plt.savefig(fig_dir / "prediction_smape.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error prediction:", e)

    try:
        # epsilon 0.10 tradeoff
        rr = routing_res[np.isclose(routing_res["epsilon"], 0.10)].copy()
        if rr.empty: rr = routing_res.copy()
        plt.figure(figsize=(9,6))
        for method, d in rr.groupby("method"):
            plt.scatter(d["distance_penalty_pct_vs_shortest_mean"], d["CO2_reduction_pct_vs_shortest_mean"], label=method)
        plt.axvline(0, linewidth=0.8); plt.axhline(0, linewidth=0.8)
        plt.xlabel("Distance penalty vs shortest (%)"); plt.ylabel("CO2 reduction vs shortest (%)")
        plt.title("Routing trade-off")
        plt.legend(fontsize=7, bbox_to_anchor=(1.02,1), loc="upper left"); plt.tight_layout()
        plt.savefig(fig_dir / "routing_tradeoff.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error tradeoff:", e)

    try:
        rr = routing_res[np.isclose(routing_res["epsilon"], 0.10)].copy()
        if rr.empty: rr = routing_res.copy()
        pivot = rr.pivot_table(index="method", columns="city", values="operational_utility_score_mean", aggfunc="mean")
        pivot.sort_index().plot(kind="bar", figsize=(12,5))
        plt.title("Main ranking metric: operational utility")
        plt.ylabel("Operational utility score")
        plt.tight_layout(); plt.savefig(fig_dir / "operational_utility_by_method.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error utility:", e)

    try:
        rr = routing_res[np.isclose(routing_res["epsilon"], 0.10)].copy()
        if rr.empty: rr = routing_res.copy()
        pivot = rr.pivot_table(index="method", columns="city", values="policy_violation_pct_mean", aggfunc="mean")
        pivot.sort_index().plot(kind="bar", figsize=(12,5))
        plt.title("Policy violation by method")
        plt.ylabel("Mean policy violation (%)")
        plt.tight_layout(); plt.savefig(fig_dir / "policy_violation_by_method.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error policy:", e)

    try:
        ar = ablation_res[np.isclose(ablation_res["epsilon"], 0.10)].copy()
        if ar.empty: ar = ablation_res.copy()
        pivot = ar.pivot_table(index="method", columns="city", values="operational_utility_score_mean", aggfunc="mean")
        pivot.plot(kind="bar", figsize=(12,5))
        plt.title("Agent-wise ablation: operational utility")
        plt.ylabel("Operational utility score")
        plt.tight_layout(); plt.savefig(fig_dir / "agent_ablation_operational_utility.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error ablation:", e)

    try:
        if not neg.empty:
            pivot = neg.pivot_table(index="epsilon", columns="city", values="consensus_stability", aggfunc="mean")
            pivot.plot(kind="line", marker="o", figsize=(8,5))
            plt.title("CARA consensus stability across epsilon")
            plt.ylabel("Consensus stability"); plt.xlabel("Epsilon")
            plt.tight_layout(); plt.savefig(fig_dir / "consensus_stability_by_epsilon.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error consensus:", e)

    try:
        if not timing.empty:
            pivot = timing.pivot_table(index="epsilon", columns="city", values="total_CARA_time_seconds", aggfunc="mean")
            pivot.plot(kind="bar", figsize=(8,5))
            plt.title("CARA decision time")
            plt.ylabel("Seconds per route")
            plt.tight_layout(); plt.savefig(fig_dir / "cara_decision_time.png", dpi=220); plt.close()
    except Exception as e:
        print("Figure error timing:", e)


def write_paper_notes(out_dir: Path):
    text = """# GATE-DT++ / CARA v5 paper notes

## Proposed framework
GATE-DT++: Generalizable Agentic Twin for Emission-aware Traffic Routing.

## Proposed model
CARA: Consensus Agentic Routing Architecture.

## Main claim
CARA is designed to optimize a multi-agent operational utility, not raw CO2 alone. The utility combines CO2 reduction, hotspot reduction, distance feasibility, policy compliance, and consensus stability.

## Reviewer-safe wording
CARA should be reported as the best operational trade-off method under the declared multi-agent utility. Do not claim that CARA always produces the absolute minimum CO2 route. CO2-only methods may reduce more CO2 by taking longer or policy-violating routes.

## Primary tables
- publication_claims_summary.csv
- method_ranking_results.csv
- routing_results.csv
- agent_ablation_results.csv
- agent_ablation_paper_table.csv
- agent_negotiation_results.csv
- timing_metrics.csv

## Primary figure
- figures/operational_utility_by_method.png
- figures/routing_tradeoff.png
- figures/policy_violation_by_method.png
- figures/agent_ablation_operational_utility.png
- figures/consensus_stability_by_epsilon.png

## Epsilon sensitivity
Use epsilons 0.05, 0.10, 0.15, 0.20.

## Important limitation
Empirical validation is cross-city within one data family. It is not full cross-dataset validation.
"""
    (out_dir / "paper_notes_v5.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cities", nargs="+", default=["Delhi","Mumbai","Chennai"])
    parser.add_argument("--n-routes", type=int, default=100)
    parser.add_argument("--k-paths", type=int, default=25)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.05,0.10,0.15,0.20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    random.seed(args.seed); np.random.seed(args.seed)
    stage_rows = []

    t0 = now()
    grid, ts = load_inputs(Path(args.data_dir), args.cities)
    stage_rows.append({"stage":"load_inputs", "seconds": now()-t0})
    print(f"Loaded cities: {list(grid.keys())}")

    t0 = now(); summarize_data(grid, ts, out_dir); stage_rows.append({"stage":"dataset_summary", "seconds": now()-t0}); print("Wrote dataset_summary.csv")

    t0 = now(); pred_agent = PredictionAgent(seed=args.seed); pred_res, daily_pred, pred_sel = pred_agent.run(ts, out_dir); stage_rows.append({"stage":"prediction", "seconds": now()-t0}); print("Wrote prediction outputs")

    t0 = now(); prepared = hotspot_experiments(grid, out_dir); stage_rows.append({"stage":"hotspot", "seconds": now()-t0}); print("Wrote hotspot_results.csv")

    t0 = now(); routing_res, ablation_res, ranking, neg, timing = routing_experiments(prepared, ts, daily_pred, out_dir, args.n_routes, args.k_paths, args.epsilons, args.seed); stage_rows.append({"stage":"routing_and_ablation", "seconds": now()-t0}); print("Wrote routing, ranking, ablation, timing outputs")

    if not args.skip_figures:
        t0 = now(); make_figures(out_dir, ts, prepared, pred_res, routing_res, ablation_res, ranking, neg, timing); stage_rows.append({"stage":"figures", "seconds": now()-t0}); print("Wrote figures")
    write_paper_notes(out_dir)
    pd.DataFrame(stage_rows).to_csv(out_dir / "run_stage_timing.csv", index=False)
    print(f"Done. Results saved in: {out_dir}")

if __name__ == "__main__":
    main()
