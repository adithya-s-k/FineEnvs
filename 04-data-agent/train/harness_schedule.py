"""Deterministic, difficulty-balanced task/harness rotation without changing TRL."""
from collections import Counter
import random


def make_schedule(tasks, harnesses, *, seed=0, easy_start=32):
    n, h = len(tasks), len(harnesses)
    if not n or not h or n % h or len(set(harnesses)) != h:
        raise ValueError('A balanced rotation requires distinct harnesses and a task count divisible by them')
    if not 0 <= easy_start <= n or easy_start % h:
        raise ValueError('The easy introduction must contain complete rounds of harnesses')
    if any(t['difficulty'] != 'easy' for t in tasks[:easy_start]):
        raise ValueError('The introduction must consist of easy tasks')
    base, cursor = {}, 0
    for tier in ['easy', 'medium', 'hard']:
        for row, task in enumerate(tasks):
            if task['difficulty'] == tier:
                base[row] = cursor % h
                cursor += 1
    if len(base) != n:
        raise ValueError('Unknown task difficulty')
    groups = []
    for pass_index in range(h):
        rng = random.Random(seed + pass_index)
        introduction = list(range(easy_start)) if pass_index == 0 else []
        buckets = [[] for _ in harnesses]
        for row in range(n):
            if row not in introduction:
                buckets[(base[row] + pass_index) % h].append(row)
        for bucket in buckets:
            rng.shuffle(bucket)
        order = introduction + [bucket[i] for i in range(len(buckets[0])) for bucket in buckets]
        for row in order:
            groups.append({'group_in_cycle': len(groups), 'pass_index': pass_index,
                           'task_row': row, 'task_index': tasks[row]['task_index'],
                           'task_name': tasks[row]['name'], 'difficulty': tasks[row]['difficulty'],
                           'harness': harnesses[(base[row] + pass_index) % h]})
    result = {'schema_version': 1, 'mode': 'one_harness_per_task_per_pass', 'seed': seed,
              'harnesses': harnesses, 'tasks': tasks, 'task_count': n,
              'groups_per_pass': n, 'passes_per_cycle': h, 'groups_per_cycle': len(groups),
              'easy_start_task_count': easy_start, 'groups': groups}
    validate_schedule(result)
    return result


def validate_schedule(schedule):
    tasks, harnesses, groups = schedule['tasks'], schedule['harnesses'], schedule['groups']
    n, h = len(tasks), len(harnesses)
    if n == 0 or h == 0 or n % h or len(groups) != n * h or len(set(harnesses)) != h:
        raise ValueError('Incomplete rotation cycle')
    expected_metadata = {'schema_version': 1, 'mode': 'one_harness_per_task_per_pass',
                         'task_count': n, 'groups_per_pass': n, 'passes_per_cycle': h,
                         'groups_per_cycle': n * h}
    if any(schedule.get(k) != v for k, v in expected_metadata.items()):
        raise ValueError('Schedule metadata disagrees with the rotation')
    if any(t['difficulty'] not in {'easy', 'medium', 'hard'} for t in tasks):
        raise ValueError('Unknown task difficulty')
    if len({t['name'] for t in tasks}) != n or len({t['task_index'] for t in tasks}) != n:
        raise ValueError('Duplicate training tasks')
    pairs = set()
    for p in range(h):
        section = groups[p * n:(p + 1) * n]
        if {g['task_row'] for g in section} != set(range(n)):
            raise ValueError('Each pass must contain every task exactly once')
        if Counter(g['harness'] for g in section) != Counter({name: n // h for name in harnesses}):
            raise ValueError('Harness counts are not balanced')
        for tier in ['easy', 'medium', 'hard']:
            counts = [sum(g['harness'] == name and g['difficulty'] == tier for g in section)
                      for name in harnesses]
            if max(counts) - min(counts) > 1:
                raise ValueError('Difficulty counts are not balanced')
        for i, g in enumerate(section):
            row = g['task_row']
            task = tasks[row]
            if (g['group_in_cycle'] != p * n + i or g['pass_index'] != p
                    or g['task_name'] != task['name'] or g['task_index'] != task['task_index']
                    or g['difficulty'] != task['difficulty']):
                raise ValueError('Group identity disagrees with task metadata')
            pairs.add((row, g['harness']))
    if len(pairs) != n * h:
        raise ValueError('Rotation repeats a task/harness pair')
    return schedule
