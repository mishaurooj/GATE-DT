"""Standalone synthetic ablations. Requires numpy, networkx, pandas (pip install numpy networkx pandas)."""
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

OUT = Path(__file__).resolve().parent / 'reviewer_ablation_outputs'
OUT.mkdir(exist_ok=True)
SEED = 2026
CITIES = {'MC1 (Delhi)': (13, 14, 1.20), 'MC2 (Mumbai)': (11, 12, 1.00), 'MC3 (Chennai)': (10, 11, .85)}

def make_network(rows, cols, factor, seed):
    rng=np.random.default_rng(seed)
    G=nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            u=r*cols+c
            G.add_node(u)
    for r in range(rows):
        for c in range(cols):
            u=r*cols+c
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr,cc=r+dr,c+dc
                if not (0<=rr<rows and 0<=cc<cols): continue
                v=rr*cols+cc
                # Synthetic one-way street segments; opposite directions can differ.
                if (r+c+min(u,v))%7==0 and u>v: continue
                G.add_edge(u,v, free=1.0+float(rng.uniform(0,.35)), cap=float(rng.uniform(15,35))/factor)
    return G

def allowed_turn(a,b,c):
    # Synthetic forbidden turns; deterministic, not real traffic regulations.
    return (a*17+b*11+c*7)%13 != 0

def restricted_path(G,origin,dest,weight):
    # State is (previous node, current node), so turn restrictions are respected.
    import heapq
    q=[(0.0,(-1,origin),(origin,))]
    best={(-1,origin):0.0}
    while q:
        cost,(prev,u),path=heapq.heappop(q)
        if cost>best.get((prev,u),float('inf'))+1e-10: continue
        if u==dest: return list(path),cost
        for v in G.successors(u):
            if prev!=-1 and not allowed_turn(prev,u,v): continue
            nc=cost+weight(u,v)
            state=(u,v)
            if nc<best.get(state,float('inf'))-1e-10:
                best[state]=nc
                heapq.heappush(q,(nc,state,path+(v,)))
    return None,float('inf')

def od_pairs(G,n,seed):
    rng=np.random.default_rng(seed)
    nodes=list(G.nodes)
    pairs=[]
    seen=set()
    for _ in range(10000):
        a,b=map(int,rng.choice(nodes,2,replace=False))
        if (a,b) in seen: continue
        seen.add((a,b))
        if nx.has_path(G,a,b): pairs.append((a,b))
        if len(pairs)==n: break
    if len(pairs)!=n: raise RuntimeError('Could not sample enough reachable OD pairs')
    return pairs

def edge_time(G,u,v,loads):
    d=G[u][v]
    return d['free']*(1+.15*(loads.get((u,v),0.0)/d['cap'])**4)

def network_cost(G,loads):
    return sum(x*edge_time(G,u,v,loads) for (u,v),x in loads.items())

def spillover(G,loads,top):
    # Mean load/capacity on links incident to high-load links, excluding top links.
    neighbor=set()
    for u,v in top:
        neighbor.update(G.out_edges(u));neighbor.update(G.in_edges(u))
        neighbor.update(G.out_edges(v));neighbor.update(G.in_edges(v))
    neighbor.difference_update(top)
    return float(np.mean([loads.get(e,0)/G[e[0]][e[1]]['cap'] for e in neighbor])) if neighbor else 0.0

def add_path(loads,path,demand):
    for u,v in zip(path[:-1],path[1:]): loads[(u,v)]=loads.get((u,v),0.0)+demand

def run_city(city,params,idx):
    rows,cols,factor=params
    G=make_network(rows,cols,factor,SEED+idx)
    U=G.to_undirected()
    pairs=od_pairs(G,100,SEED+idx+300)
    # A. Route feasibility: unconstrained undirected vs directed and turn restricted.
    base_valid=0;dir_valid=0;turn_valid=0;detours=[];no_route=0
    for a,b in pairs:
        p=nx.shortest_path(U,a,b,weight='free')
        base_valid+=int(all(U.has_edge(x,y) for x,y in zip(p,p[1:])))
        dir_valid+=int(all(G.has_edge(x,y) for x,y in zip(p,p[1:])))
        turn_valid+=int(all(allowed_turn(x,y,z) for x,y,z in zip(p,p[1:],p[2:])))
        rp,rc=restricted_path(G,a,b,lambda x,y:G[x][y]['free'])
        if rp is None: no_route+=1
        else:
            uc=sum(U[x][y]['free'] for x,y in zip(p,p[1:]))
            detours.append((rc/uc-1)*100)
    graph_row={'City':city,'OD':len(pairs),'Undirected path continuity (%)':100*base_valid/len(pairs),
       'Direction valid (%)':100*dir_valid/len(pairs),
       'Turn valid (%)':100*turn_valid/len(pairs),
       'Both valid (%)':100*sum(all(G.has_edge(x,y) for x,y in zip(nx.shortest_path(U,a,b,weight='free'),nx.shortest_path(U,a,b,weight='free')[1:])) and all(allowed_turn(x,y,z) for x,y,z in zip(nx.shortest_path(U,a,b,weight='free'),nx.shortest_path(U,a,b,weight='free')[1:],nx.shortest_path(U,a,b,weight='free')[2:])) for a,b in pairs)/len(pairs),
       'Restricted reachable (%)':100*(len(pairs)-no_route)/len(pairs),
       'Restricted detour mean (%)':float(np.mean(detours)) if detours else np.nan}
    # B. Fixed-demand traffic assignment: static free-flow vs iterative sequential rerouting.
    rng=np.random.default_rng(SEED+idx+900)
    demands={pair:float(rng.uniform(2.5,7.5))*factor for pair in pairs}
    static_routes={pair:restricted_path(G,*pair,lambda x,y:G[x][y]['free'])[0] for pair in pairs}
    static_loads={}
    for pair,path in static_routes.items():
        if path: add_path(static_loads,path,demands[pair])
    # Iterative route-load update using method of successive averages (MSA).
    current=dict(static_loads)
    iterations=20
    changes=[]
    for k in range(1,iterations+1):
        new={}
        switches=0
        for pair in pairs:
            p,_=restricted_path(G,*pair,lambda x,y:edge_time(G,x,y,current))
            if p is None: continue
            switches+=int(p!=static_routes[pair])
            add_path(new,p,demands[pair])
        alpha=1/(k+1)
        current={e:(1-alpha)*current.get(e,0)+alpha*new.get(e,0) for e in G.edges}
        changes.append(switches/len(pairs)*100)
    top=set(sorted(G.edges,key=lambda e:static_loads.get(e,0)/G[e[0]][e[1]]['cap'],reverse=True)[:max(1,len(G.edges)//10)])
    def measures(loads):
        vc=np.array([loads.get((u,v),0)/G[u][v]['cap'] for u,v in G.edges])
        return {'Mean V/C':float(vc.mean()),'P95 V/C':float(np.percentile(vc,95)),
                'Overcapacity links (%)':float(np.mean(vc>1)*100),
                'Network travel cost':network_cost(G,loads),
                'Hotspot mean V/C':float(np.mean([loads.get(e,0)/G[e[0]][e[1]]['cap'] for e in top])),
                'Neighbor spillover V/C':spillover(G,loads,top)}
    fb=[]
    for mode,loads in [('Static',static_loads),('Iterative',current)]:
        fb.append({'City':city,'Mode':mode,'OD':len(pairs),'Iterations':0 if mode=='Static' else iterations,
                   **measures(loads),'Route changes vs static (%)':0.0 if mode=='Static' else changes[-1]})
    return graph_row,fb

def fmt(v):
    if isinstance(v,(int,np.integer)): return str(v)
    if isinstance(v,(float,np.floating)): return f'{v:.3f}'
    return str(v)

def latex_table(df,caption,label):
    columns=list(df.columns)
    spec='l'*min(2,len(columns))+'r'*max(0,len(columns)-2)
    lines=['\\begin{table*}[!t]','\\centering',f'\\caption{{\\textcolor{{blue}}{{{caption}}}}}',f'\\label{{{label}}}',
           '\\begingroup\\color{blue}\\scriptsize','\\setlength{\\tabcolsep}{3pt}',
           '\\resizebox{\\textwidth}{!}{%',f'\\begin{{tabular}}{{{spec}}}','\\hline',
           ' & '.join(c.replace('%',r'\%') for c in columns)+r' \\', '\\hline']
    for row in df.itertuples(index=False,name=None): lines.append(' & '.join(fmt(v) for v in row)+r' \\')
    lines+=['\\hline','\\end{tabular}}','\\endgroup','\\end{table*}']
    return '\n'.join(lines)+'\n'

def main():
    graph=[];feedback=[]
    for i,(city,params) in enumerate(CITIES.items()):
        g,f=run_city(city,params,i)
        graph.append(g);feedback.extend(f)
    gd=pd.DataFrame(graph);fd=pd.DataFrame(feedback)
    gd.to_csv(OUT/'graph_restriction_ablation.csv',index=False)
    fd.to_csv(OUT/'traffic_feedback_ablation.csv',index=False)
    (OUT/'graph_restriction_ablation.tex').write_text(latex_table(gd,
      'Synthetic graph-restriction ablation: direction, turns, reachability and detour (100 OD pairs per city).',
      'tab:synthetic_graph_restrictions'),encoding='utf-8')
    (OUT/'traffic_feedback_ablation.tex').write_text(latex_table(fd,
      'Synthetic congestion-feedback ablation: static versus iterative traffic assignment (100 OD pairs per city).',
      'tab:synthetic_traffic_feedback'),encoding='utf-8')
    print('GRAPH RESTRICTION ABLATION\n',gd.to_string(index=False))
    print('\nTRAFFIC FEEDBACK ABLATION\n',fd.to_string(index=False))
    print('\nFiles written to',OUT)
    print('NOTE: Both experiments are synthetic; do not report as real road-network validation.')
if __name__=='__main__': main()
