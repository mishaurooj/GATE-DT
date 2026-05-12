# GATE-DT Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-Traffic%20%26%20Air%20Quality-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Cities-Delhi%20%7C%20Mumbai%20%7C%20Chennai-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Format-CSV-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Routes-100%20OD%20Pairs-purple?style=for-the-badge">
</p>

---

# Dataset Overview

This directory contains the processed datasets used for the GATE-DT eco-routing and congestion-aware transportation framework. The datasets include:

* City-scale traffic observations
* Road-network grid statistics
* Time-series mobility data
* Air-quality indicators
* NOx and PM$_{2.5}$ aligned measurements
* Multi-city experimental routing records

---

# Dataset Structure

| Dataset File                    | Description                                      | City    | Type         |
| ------------------------------- | ------------------------------------------------ | ------- | ------------ |
| `Delhi-timeseriesdataset.csv`   | Daily traffic and environmental time-series data | Delhi   | Time-Series  |
| `Delhigriddedmapdata.csv`       | Grid-based spatial traffic representation        | Delhi   | Spatial Grid |
| `Mumbai-timeseriesdataset.csv`  | Daily traffic and environmental time-series data | Mumbai  | Time-Series  |
| `Mumbaigriddedmapdata.csv`      | Grid-based spatial traffic representation        | Mumbai  | Spatial Grid |
| `Chennai-timeseriesdataset.csv` | Daily traffic and environmental time-series data | Chennai | Time-Series  |
| `Chennai-griddedmapdata.csv`    | Grid-based spatial traffic representation        | Chennai | Spatial Grid |

---

# Quick Dataset Preview

## Delhi Time-Series Dataset

<table>
<tr>
<th style="background-color:#0D47A1;color:white;">Date</th>
<th style="background-color:#0D47A1;color:white;">Traffic Density</th>
<th style="background-color:#0D47A1;color:white;">Average Speed</th>
<th style="background-color:#0D47A1;color:white;">NOx</th>
<th style="background-color:#0D47A1;color:white;">PM2.5</th>
</tr>
<tr>
<td>2021-01-01</td>
<td>0.82</td>
<td>34.1</td>
<td>58.3</td>
<td>112.7</td>
</tr>
<tr>
<td>2021-01-02</td>
<td>0.79</td>
<td>35.6</td>
<td>55.8</td>
<td>108.2</td>
</tr>
<tr>
<td>2021-01-03</td>
<td>0.88</td>
<td>31.5</td>
<td>61.4</td>
<td>119.9</td>
</tr>
</table>

---

## Mumbai Time-Series Dataset

<table>
<tr>
<th style="background-color:#1B5E20;color:white;">Date</th>
<th style="background-color:#1B5E20;color:white;">Traffic Density</th>
<th style="background-color:#1B5E20;color:white;">Average Speed</th>
<th style="background-color:#1B5E20;color:white;">NOx</th>
<th style="background-color:#1B5E20;color:white;">PM2.5</th>
</tr>
<tr>
<td>2021-01-01</td>
<td>0.91</td>
<td>28.4</td>
<td>66.1</td>
<td>131.4</td>
</tr>
<tr>
<td>2021-01-02</td>
<td>0.87</td>
<td>30.2</td>
<td>63.7</td>
<td>126.8</td>
</tr>
<tr>
<td>2021-01-03</td>
<td>0.93</td>
<td>27.5</td>
<td>68.5</td>
<td>135.9</td>
</tr>
</table>

---

## Chennai Time-Series Dataset

<table>
<tr>
<th style="background-color:#6A1B9A;color:white;">Date</th>
<th style="background-color:#6A1B9A;color:white;">Traffic Density</th>
<th style="background-color:#6A1B9A;color:white;">Average Speed</th>
<th style="background-color:#6A1B9A;color:white;">NOx</th>
<th style="background-color:#6A1B9A;color:white;">PM2.5</th>
</tr>
<tr>
<td>2021-01-01</td>
<td>0.71</td>
<td>41.8</td>
<td>44.3</td>
<td>82.7</td>
</tr>
<tr>
<td>2021-01-02</td>
<td>0.69</td>
<td>42.9</td>
<td>42.6</td>
<td>79.5</td>
</tr>
<tr>
<td>2021-01-03</td>
<td>0.75</td>
<td>39.7</td>
<td>46.2</td>
<td>85.8</td>
</tr>
</table>

---

# Spatial Grid Dataset Preview

## Delhi Grid Dataset

<table>
<tr>
<th style="background-color:#37474F;color:white;">Grid ID</th>
<th style="background-color:#37474F;color:white;">Latitude</th>
<th style="background-color:#37474F;color:white;">Longitude</th>
<th style="background-color:#37474F;color:white;">Congestion Score</th>
<th style="background-color:#37474F;color:white;">Emission Level</th>
</tr>
<tr>
<td>DL_001</td>
<td>28.6139</td>
<td>77.2090</td>
<td>0.87</td>
<td>High</td>
</tr>
<tr>
<td>DL_002</td>
<td>28.6450</td>
<td>77.1870</td>
<td>0.73</td>
<td>Medium</td>
</tr>
</table>

---

# Experimental Configuration

| Parameter              | Value                   |
| ---------------------- | ----------------------- |
| Number of Cities       | 3                       |
| Daily Samples per City | 365                     |
| OD Pairs per City      | 100                     |
| Candidate Routes       | 25                      |
| Random Seed            | 42                      |
| Traffic Features       | Congestion, Speed, Flow |
| Environmental Features | NOx, PM$_{2.5}$         |

---

# Loading the Dataset

## Python Example

```python
import pandas as pd

# Load Delhi dataset
file_path = "Dataset/Delhi-timeseriesdataset.csv"

df = pd.read_csv(file_path)

print(df.head())
```

---

# Dataset Visualization

<p align="center">
  <img src="../Figures/cara_decision_time.png" width="850">
</p>

---

# Citation

```bibtex
@article{gate_dt_2026,
  title={Carbon-Aware Agentic Digital Twin Framework for Eco-Routing},
  author={Suleman et al.},
  journal={Under Review},
  year={2026}
}
```

---

# License

This dataset is provided for academic and research purposes only.

