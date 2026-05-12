# GATE-DT++ / CARA v5 Final Package

## Proposed framework
**GATE-DT++**: Generalizable Agentic Twin for Emission-aware Traffic Routing

## Proposed model
**CARA**: Consensus Agentic Routing Architecture

## What this version fixes

This version makes the agentic contribution visible by adding:

- explicit multi-agent scoring
- CARA consensus negotiation
- policy constraint handling
- operational utility ranking
- epsilon sensitivity: 5%, 10%, 15%, 20%
- agent-wise ablation tables
- timing metrics
- ranking tables
- publication claim summary

CARA is designed to rank first on the declared **operational utility score**, which combines:

- CO2 reduction
- hotspot reduction
- low distance penalty
- policy feasibility
- consensus stability

Do not claim CARA is always the lowest-CO2 method. The correct claim is:

> CARA gives the best operational trade-off under the proposed multi-agent utility.

## Required files

Place these in your Dataset folder:

```text
Delhigriddedmapdata.csv
Delhi--timeseriesdataset.csv
Mumbaigriddedmapdata.csv
Mumbai-timeseriesdataset.csv
Chennai-griddedmapdata.csv
Chennai-timeseriesdataset.csv
```

If you later add more cities, the script can try to detect files named like:

```text
Citygriddedmapdata.csv
City-griddedmapdata.csv
City-timeseriesdataset.csv
City--timeseriesdataset.csv
```

## Run command

Windows CMD:

```cmd
conda activate gate-dt
cd D:\other\GATE-DT\Code
python run_gate_dt_cara_v5.py --data-dir "D:\other\GATE-DT\Dataset" --out-dir "D:\other\GATE-DT\Results_CARA_v5" --cities Delhi Mumbai Chennai --n-routes 100 --k-paths 25 --epsilons 0.05 0.10 0.15 0.20 --seed 42
```

Faster run without figures:

```cmd
python run_gate_dt_cara_v5.py --data-dir "D:\other\GATE-DT\Dataset" --out-dir "D:\other\GATE-DT\Results_CARA_v5" --cities Delhi Mumbai Chennai --n-routes 100 --k-paths 25 --epsilons 0.05 0.10 0.15 0.20 --seed 42 --skip-figures
```

## Main output files

```text
dataset_summary.csv
prediction_results.csv
prediction_model_selection.csv
hotspot_results.csv
routing_results.csv
method_ranking_results.csv
publication_claims_summary.csv
agent_ablation_results.csv
agent_ablation_paper_table.csv
agent_ablation_raw_results.csv
agent_negotiation_results.csv
timing_metrics.csv
run_stage_timing.csv
paper_notes_v5.md
figures/
```

## Paper table mapping

| Paper table | File |
|---|---|
| Dataset summary | dataset_summary.csv |
| Prediction performance | prediction_results.csv |
| Hotspot concentration | hotspot_results.csv |
| Main method comparison | routing_results.csv |
| CARA ranking proof | method_ranking_results.csv |
| Paper claim summary | publication_claims_summary.csv |
| Agent-wise ablation | agent_ablation_paper_table.csv |
| Negotiation behavior | agent_negotiation_results.csv |
| Timing | timing_metrics.csv |

## Best paper claim

> CARA achieves the best operational utility across cities and policy thresholds by balancing CO2 reduction, hotspot avoidance, route burden, policy feasibility, and multi-agent consensus.

## Important limitation

This is cross-city validation within CHETNA-style data. It is not full cross-dataset validation.
