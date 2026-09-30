"""Validate explicit finite passes, including a single-harness control arm."""
from collections import Counter

def validate_schedule(s):
    tasks,groups,hs=s['tasks'],s['groups'],s['harnesses'];n=len(tasks);passes=s['passes_per_cycle']
    if not n or not hs or len(set(hs))!=len(hs) or len(groups)!=n*passes:raise ValueError('Incomplete schedule')
    if s['task_count']!=n or s['groups_per_pass']!=n or s['groups_per_cycle']!=len(groups):raise ValueError('Metadata mismatch')
    if len({t['name'] for t in tasks})!=n or len({t['task_index'] for t in tasks})!=n:raise ValueError('Duplicate tasks')
    for epoch in range(passes):
        part=groups[epoch*n:(epoch+1)*n]
        if Counter(g['task_row'] for g in part)!=Counter(range(n)):raise ValueError('Task exposure mismatch')
        for tier in ['medium','hard']:
            counts=[sum(g['difficulty']==tier and g['harness']==h for g in part) for h in hs]
            if max(counts)-min(counts)>1:raise ValueError('Unbalanced difficulty')
        for i,g in enumerate(part):
            t=tasks[g['task_row']]
            if g['group_in_cycle']!=epoch*n+i or g['pass_index']!=epoch or g['harness'] not in hs or any(g[a]!=t[b] for a,b in [('task_name','name'),('task_index','task_index'),('difficulty','difficulty')]):raise ValueError('Group identity mismatch')
    return s
