"""Read-only audit of semantic-leakage raw results. Run from project root."""
from collections import Counter
from pathlib import Path
import pandas as pd
from run_experiment_fast import (SCHEDULE_FILE, EXPECTED_PER_RUN, load_requests,
                                 result_path, valid_answer)

schedule = pd.read_csv(SCHEDULE_FILE)
counts = Counter()
issues = Counter()
missing_runs = []
for _, run in schedule.iterrows():
    run_id, provider, mode = run['run_id'], run['model'], run['response_mode']
    expected = set(load_requests(run)['request_id'].astype(str))
    path = result_path(run)
    if not path.exists():
        missing_runs.append(run_id)
        counts[(provider, 'missing')] += EXPECTED_PER_RUN
        continue
    df = pd.read_csv(path, dtype={'request_id': str}, low_memory=False)
    needed = {'request_id', 'success', 'response_raw'}
    if not needed.issubset(df.columns):
        print(f'INVALID SCHEMA: {path} missing {sorted(needed - set(df.columns))}')
        issues['schema_errors'] += 1
        continue
    success = df['success'].astype(str).str.lower().eq('true')
    valid = df['response_raw'].apply(lambda x: valid_answer(x, mode))
    good_rows = df.loc[success & valid, 'request_id'].astype(str)
    good_ids = set(good_rows) & expected
    missing = len(expected - good_ids)
    counts[(provider, 'valid_unique')] += len(good_ids)
    counts[(provider, 'missing')] += missing
    issues['invalid_success_rows'] += int((success & ~valid).sum())
    issues['failed_rows'] += int((~success).sum())
    issues['duplicate_success_rows'] += int(good_rows.duplicated().sum())
    issues['unexpected_success_ids'] += len(set(good_rows) - expected)
    if missing == 0:
        counts[(provider, 'complete_runs')] += 1
    else:
        missing_runs.append(f'{run_id} ({missing} missing)')

print('\nREAD-ONLY AUDIT — no API calls, no files changed')
print(f'Scheduled runs: {len(schedule)}; expected responses: {len(schedule) * EXPECTED_PER_RUN:,}')
print('Provider     Complete runs  Valid unique   Missing')
for p in ('qwen', 'mistral', 'openai'):
    print(f'{p:<12} {counts[(p,"complete_runs")]:>5}/60       {counts[(p,"valid_unique")]:>10,} {counts[(p,"missing")]:>10,}')
print(f'ALL                       {sum(counts[(p,"valid_unique")] for p in ("qwen","mistral","openai")):>10,} {sum(counts[(p,"missing")] for p in ("qwen","mistral","openai")):>10,}')
print('Quality checks:', dict(issues))
print('Incomplete runs (first 12):', ', '.join(missing_runs[:12]) or 'none')
