# Figures

---

# Framework Architecture

<p align="center">
  <img src="gate_dt_cara_architecture.png" width="95%">
</p>

### Description

The figure presents the complete GATE-DT++ and CARA routing architecture.

The transportation digital twin synchronizes:

- Traffic states
- Pollution measurements
- Hotspot distributions
- Graph routing information

CARA then performs:

- Candidate-route evaluation
- Multi-agent negotiation
- Policy-feasibility analysis
- Consensus-aware route selection

---

# CARA Routing Workflow

<p align="center">
  <img src="gate_dt_cara.png" width="90%">
</p>

### Description

The workflow illustrates:

1. Candidate-route generation  
2. Traffic prediction  
3. Hotspot scoring  
4. Pollutant evaluation  
5. Policy-feasibility checking  
6. Consensus negotiation  
7. Final route selection  

The framework balances emissions, congestion exposure, feasibility, and routing stability.

---

# Delhi Hotspot Distribution

<p align="center">
  <img src="delhi_hotspot_map.png" width="75%">
</p>

### Description

Delhi exhibits:

- Large-scale congestion regions
- Distributed hotspot concentration
- Dense arterial-road traffic
- Strong pollutant-density variation

The hotspot distribution shows persistent urban congestion clusters.

---

# Mumbai Hotspot Distribution

<p align="center">
  <img src="mumbai_hotspot_map.png" width="75%">
</p>

### Description

Mumbai exhibits:

- Compact corridor-dominant traffic flow
- Localized congestion bottlenecks
- High hotspot concentration
- Dense coastal routing structure

The traffic flow remains concentrated within narrow urban corridors.

---

# Chennai Hotspot Distribution

<p align="center">
  <img src="chennai_hotspot_map.png" width="75%">
</p>

### Description

Chennai exhibits:

- Smoother congestion transitions
- Lower hotspot fragmentation
- Medium-density urban traffic
- More spatially distributed flow

The routing environment remains comparatively stable.

---

# Traffic Forecasting Analysis

<p align="center">
  <img src="traffic_timeseries.png" width="90%">
</p>

### Description

The forecasting module estimates short-term traffic behavior before route negotiation.

The prediction pipeline uses:

- Temporal traffic statistics
- Lag-based features
- Ensemble-tree regression
- Dynamic traffic synchronization

The forecasting stage improves adaptive routing under changing traffic conditions.

---

# Routing Decision-Time Analysis

<p align="center">
  <img src="cara_decision_time.png" width="75%">
</p>

### Description

The routing-time evaluation measures the computational overhead introduced by consensus negotiation.

The results show:

- Operationally feasible runtime
- Moderate negotiation overhead
- Stable metropolitan-scale routing performance
- Efficient candidate-route evaluation

Consensus negotiation introduces limited additional computation relative to graph-search operations.
