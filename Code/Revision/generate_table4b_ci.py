#!/usr/bin/env python3
"""Standalone Table 4(b) generator. Python standard library only.
Wilson 95% CIs for binary negotiation success, 100 OD pairs/city.
Aggregate retention/support are deliberately excluded from binomial CIs.
"""
from math import sqrt
from pathlib import Path
import csv

N = 100
Z = 1.959963984540054
DATA = [
    ('MC1 (Delhi)', 32, 98),
    ('MC2 (Mumbai)', 32, 88),
    ('MC3 (Chennai)', 43, 91),
]

def wilson(k, n=N, z=Z):
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return 100*(center-half), 100*(center+half)

rows = []
for city, one, cara in DATA:
    l1, u1 = wilson(one)
    l2, u2 = wilson(cara)
    rows.append((city, one, l1, u1, cara, l2, u2, cara-one))

out = Path(__file__).resolve().parent
with (out/'table4b_negotiation_ci.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['city','one_round_success_pct','one_round_ci_low_pct','one_round_ci_high_pct','cara_success_pct','cara_ci_low_pct','cara_ci_high_pct','observed_difference_pp'])
    for row in rows:
        writer.writerow([row[0],row[1],f'{row[2]:.2f}',f'{row[3]:.2f}',row[4],f'{row[5]:.2f}',f'{row[6]:.2f}',row[7]])

lines = [
 r'\begin{table}[!t]',
 r'\centering',
 r'\caption{\textcolor{blue}{Negotiation success with 95\% Wilson confidence intervals over 100 OD pairs per city ($\epsilon=0.20$).}}',
 r'\label{tab:negotiation_ci}',
 r'\begingroup\color{blue}',
 r'\scriptsize',
 r'\setlength{\tabcolsep}{3pt}',
 r'\renewcommand{\arraystretch}{0.95}',
 r'\resizebox{\columnwidth}{!}{%',
 r'\begin{tabular}{lrrrr}',
 r'\hline',
 r'City & 1-round (95\% CI) & CARA (95\% CI) & Difference (pp) & $n$ \\',
 r'\hline',
]
for city, one, l1, u1, cara, l2, u2, diff in rows:
    lines.append(f'{city} & {one:.0f} [{l1:.2f}, {u1:.2f}] & {cara:.0f} [{l2:.2f}, {u2:.2f}] & +{diff:.0f} & {N} '+r'\\')
lines += [
 r'\hline',
 r'\end{tabular}}',
 r'\vspace{1pt}',
 r'\parbox{\columnwidth}{\tiny\textit{Note:} Intervals are 95\% Wilson score intervals for binary negotiation success. The difference is the observed CARA minus one-round success rate in percentage points, without a paired confidence interval or significance claim. Paired inference and retention intervals require per-OD and repeated-perturbation records.}',
 r'\endgroup',
 r'\end{table}',
]
(out/'table4b_negotiation_ci.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
print('\nCreated table4b_negotiation_ci.tex and table4b_negotiation_ci.csv')
