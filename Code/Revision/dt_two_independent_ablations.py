#!/usr/bin/env python3
"""Standalone reproducible SIMULATED benchmark. No GATE-DT files required.
Two separate ablations: (1) synchronization on/off; (2) continuous updates on/off.
Numbers are synthetic performance measurements, NOT physical sensor or live-twin validation.
"""
import argparse, csv, math, random, statistics, time
from pathlib import Path

CITIES = {'MC1 (Delhi)': (8072, 1.0), 'MC2 (Mumbai)': (2260, 1.2), 'MC3 (Chennai)': (1997, 0.8)}
FIELDS = ['City','Configuration','Events','Latency mean (ms)','Latency p95 (ms)','Throughput (events/s)','Staleness mean (ms)','State MAE','Update success (%)','Route changes (%)']

def percentile(a, q):
    a=sorted(a); p=(len(a)-1)*q; lo=int(p); hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(p-lo)

def simulate(city, n_nodes, scale, ablation, enabled, events, seed):
    rng=random.Random(seed)
    # Virtual event-time simulation, with deterministic synthetic traffic observations.
    # Both arms see identical underlying observations for paired comparison.
    truth=[50+10*math.sin(i*.055)+scale*4*math.sin(i*.011)+rng.gauss(0,2) for i in range(events)]
    # Changes are applied to a synthetic state; route selection is a toy threshold controller.
    state=truth[0]; latencies=[]; stale=[]; errors=[]; routes=[]; successes=0
    # Virtual event time interval: 100 ms. Processing delay is synthetic and city-size dependent.
    for i, obs in enumerate(truth):
        event_t=i*100.0
        delay=max(.1, rng.gauss(7+0.001*n_nodes, 2.0))
        if ablation=='sync':
            # ON: synthetic incoming observation delivered every event.
            # OFF: periodic batch ingestion once every 10 events.
            apply=(i%1==0) if enabled else (i%10==0)
        else:
            # ON: state refreshed on every event; OFF: frozen after initial snapshot.
            apply=True if enabled else (i==0)
        if apply:
            state=obs
            successes+=1
            last_update_t=event_t+delay
            latencies.append(delay)
        else:
            latencies.append(0.0) # no update occurred: excluded from delivered-update latency below
        age=max(0.0,event_t-(last_update_t if 'last_update_t' in locals() else 0.0))
        stale.append(age)
        errors.append(abs(state-obs))
        routes.append(int(state>=50))
    delivered=[v for v in latencies if v>0]
    # Throughput is successfully applied updates per virtual second (100ms cadence).
    duration_s=events*.1
    return [city, ('ON' if enabled else 'OFF'), events,
            statistics.mean(delivered) if delivered else 0,
            percentile(delivered,.95) if delivered else 0,
            successes/duration_s,statistics.mean(stale),statistics.mean(errors),
            100*successes/events,100*sum(routes[i]!=routes[i-1] for i in range(1,events))/(events-1)]

def fmt(row):
    return [row[0],row[1],str(row[2])]+[f'{x:.3f}' for x in row[3:]]

def latex_table(rows, title, label):
    lines=[r'\begin{table*}[!t]',r'\centering',r'\begingroup\color{blue}',r'\scriptsize',rf'\caption{{\textcolor{{blue}}{{{title}}}}}',rf'\label{{{label}}}',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llrrrrrrrr}',r'\hline',r'City & Mode & Events & Lat. mean (ms) & Lat. p95 (ms) & Throughput (ev/s) & Staleness (ms) & State MAE & Success (\%) & Route changes (\%) \\',r'\hline']
    for row in rows:
        v=fmt(row);lines.append(' & '.join(v)+r' \\')
    lines += [r'\hline',r'\end{tabular}}',r'\endgroup',r'\end{table*}']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--out',default='dt_ablation_outputs')
    a=p.parse_args()
    if a.events<20: p.error('--events must be >= 20')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for ablation, name, caption in [('sync','synchronization_ablation','(d) Simulated synchronization ablation (synthetic event stream)'),('update','continuous_update_ablation','(e) Simulated continuous-state-update ablation (synthetic event stream)')]:
        rows=[]
        for j,(city,(nodes,scale)) in enumerate(CITIES.items()):
            for enabled in (False,True):
                rows.append(simulate(city,nodes,scale,ablation,enabled,a.events,a.seed+j*101))
        with (out/f'{name}.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(FIELDS);w.writerows(fmt(r) for r in rows)
        (out/f'{name}.tex').write_text(latex_table(rows,caption,'tab:dt_'+('sync' if ablation=='sync' else 'continuous')),encoding='utf8')
        print(name)
        for row in rows: print(' | '.join(fmt(row)))
    (out/'README.txt').write_text('SIMULATION ONLY. Synthetic traffic observations, virtual timestamps and simulated processing delays. NOT evidence of live sensor synchronization or real-time physical twin performance.\nLatency is modeled per delivered update; OFF may have fewer delivered updates. Throughput is delivered updates per virtual second. Staleness is event-time age of latest applied state. State MAE is synthetic traffic intensity units. Success is fraction of events with applied update; Route changes is percentage of successive route decisions differing.\n',encoding='utf8')
    print('Files saved to',out)

if __name__=='__main__': main()
