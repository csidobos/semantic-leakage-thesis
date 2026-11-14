"""Resumable, sequential semantic-leakage experiment runner.

Run from project root:
    python run_experiment_safe.py                 # read-only preflight
    python run_experiment_safe.py --max-calls 3   # at most 3 new API calls
    python run_experiment_safe.py --all           # all remaining runs (costs money)

Existing raw CSVs are never deleted or rewritten. Successful IDs are reused.
"""
import argparse
import json
import csv
import hashlib
import os
import random
import re
import time
from datetime import datetime as _datetime
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEDULE_FILE = Path('data/run_schedule.csv')
RESULTS_DIR = Path('results/raw')
REQUEST_FILES = {(lang, mode): Path(f'data/{lang}_{mode}_requests.csv')
                 for lang in ('HU', 'EN') for mode in ('free_text', 'mc', 'mc_unknown')}
EXPECTED_PER_RUN = 2016
SEED = 42
PROVIDER_DELAY = {'openai': 0.0, 'mistral': 1.1, 'qwen': 1.0}
MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 5
# Stop rather than retry forever when the provider cannot serve requests.
MAX_CONSECUTIVE_FAILURES = 5
# When a provider rate-limits requests, wait longer than for ordinary errors.
RATE_LIMIT_DEFAULT_WAIT = 60
RATE_LIMIT_MAX_WAIT = 300
# Document requested settings without guessing provider-resolved default temperatures.
# Confirm these against the deployed API implementations before --all.
MODEL_SETTINGS = {
    'openai': {'model_id': 'gpt-6-luna', 'reasoning_request': 'effort=none',
               'temperature_request': 'omitted (provider default)'},
    'mistral': {'model_id': 'mistral-small-2603', 'reasoning_request': 'effort=none',
                'temperature_request': 'omitted (provider default)'},
    'qwen': {'model_id': 'qwen3.8-flash', 'reasoning_request': 'effort=none',
             'temperature_request': 'omitted (provider default)'},
}
MANIFEST_DIR = Path('results/config')


def ensure_manifest(run):
    """Create immutable run settings sidecar; never rewrite existing manifests."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{run['run_id']}_settings.json"
    settings = MODEL_SETTINGS[run['model']]
    expected = {
        'run_id': str(run['run_id']), 'provider': str(run['model']),
        'model_id': settings['model_id'], 'language': str(run['language']),
        'response_mode': str(run['response_mode']),
        'repetition': int(run['repetition']),
        'reasoning_request': settings['reasoning_request'],
        'temperature_request': settings['temperature_request'],
        'temperature_resolved': 'not verified by API response',
        'metadata_source': 'runner configuration; verify API files match',
    }
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved != expected:
            raise ValueError(f'Manifest mismatch for {path}; refusing mixed configurations')
    else:
        # Exclusive create prevents overwriting an existing sidecar.
        with path.open('x', encoding='utf-8') as f:
            json.dump(expected, f, ensure_ascii=False, indent=2)
            f.write('\n')
    return path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def shuffled_requests(df, lang, mode, repetition):
    seed_str = f'{SEED}|{lang}|{mode}|{repetition}'
    seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest()[:8], 16)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def valid_answer(raw, mode):
    if not isinstance(raw, str) or not raw.strip():
        return False
    if mode == 'free_text':
        return True  # Content classification is a separate, human-reviewed stage.
    allowed = 'ABCD' if mode == 'mc_unknown' else 'ABC'
    return re.fullmatch(rf'\s*[{allowed}]\s*[.]?\s*', raw, flags=re.IGNORECASE) is not None


def response_letter(raw):
    return raw.strip().upper().rstrip('.').strip()



def retry_delay(exc, attempt):
    """Return a conservative wait, respecting Retry-After where available."""
    default = min(120, RETRY_BASE_SECONDS * 2 ** (attempt - 1))
    status = getattr(exc, 'status_code', None)
    response = getattr(exc, 'response', None)
    if status is None and response is not None:
        status = getattr(response, 'status_code', None)
    msg = str(exc).lower()
    limited = str(status) == '429' or 'rate limit' in msg or 'rate_limit' in msg or 'too many requests' in msg
    if not limited:
        return default, False

    header = None
    for obj in (response, exc):
        headers = getattr(obj, 'headers', None)
        if headers is not None:
            try:
                header = headers.get('retry-after') or headers.get('Retry-After')
            except (AttributeError, TypeError):
                pass
        if header:
            break
    wait = None
    if header:
        try:
            wait = float(header)
        except (ValueError, TypeError):
            try:
                dt = parsedate_to_datetime(str(header))
                from datetime import timezone as _tz
                wait = (dt - _datetime.now(_tz.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
    # Never hammer a provider after a 429, even if Retry-After is absent.
    wait = max(RATE_LIMIT_DEFAULT_WAIT, default, wait or 0)
    return min(RATE_LIMIT_MAX_WAIT, wait), True


def result_path(run):
    return RESULTS_DIR / (f"{run['run_id']}_{run['model']}_{run['language']}_"
                          f"{run['response_mode']}_rep{int(run['repetition'])}.csv")


def get_successful_ids(path, expected_ids, mode):
    if not path.exists():
        return set(), 0
    df = pd.read_csv(path, dtype={'request_id': str}, low_memory=False)
    if 'request_id' not in df or 'success' not in df or 'response_raw' not in df:
        raise ValueError(f'Missing columns in existing results: {path}')
    # Historical runs used success=True for transport success only.
    successes = df['success'].astype(str).str.lower().eq('true')
    valid = df['response_raw'].apply(lambda x: valid_answer(x, mode))
    good = set(df.loc[successes & valid, 'request_id'].astype(str))
    unexpected = good - expected_ids
    if unexpected:
        raise ValueError(f'{path}: {len(unexpected)} unexpected request_id values')
    return good, int((successes & ~valid).sum())


def load_requests(run):
    lang, mode, rep = run['language'], run['response_mode'], int(run['repetition'])
    source = REQUEST_FILES[(lang, mode)]
    df = pd.read_csv(source, dtype={'request_id': str}, low_memory=False)
    required = {'request_id', 'repetition', 'language', 'prompt_text'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{source}: missing columns {sorted(missing)}')
    df = df.loc[df['repetition'].astype(int).eq(rep)].copy()
    if len(df) != EXPECTED_PER_RUN or df['request_id'].isna().any() or df['request_id'].duplicated().any():
        raise ValueError(f'{source}: repetition {rep} must have {EXPECTED_PER_RUN} unique IDs')
    if not df['language'].eq(lang).all():
        raise ValueError(f'{source}: unexpected language')
    if 'response_mode' in df and not df['response_mode'].eq(mode).all():
        raise ValueError(f'{source}: unexpected response_mode')
    if df['prompt_text'].isna().any():
        raise ValueError(f'{source}: empty prompts')
    if mode != 'free_text':
        expected_letters = 'ABCD' if mode == 'mc_unknown' else 'ABC'
        for letter in expected_letters:
            if f'option_{letter}_class' not in df:
                raise ValueError(f'{source}: missing option_{letter}_class')
            if df[f'option_{letter}_class'].isna().any():
                raise ValueError(f'{source}: empty option_{letter}_class')
    return shuffled_requests(df, lang, mode, rep)


def get_provider_functions():
    # Lazy import: preflight does not require API credentials or SDK imports.
    from api.openai_api import get_response as openai
    from api.mistral_api import get_response as mistral
    from api.qwen_api import get_response as qwen
    return {'openai': openai, 'mistral': mistral, 'qwen': qwen}


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    # Match the existing CSV schema if a pilot file already exists.
    if exists:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            columns = next(csv.reader(f))
        if set(row) - set(columns):
            # Extra metadata belongs in a separate manifest, not in a mismatched CSV row.
            row = {k: row.get(k) for k in columns}
    else:
        columns = list(row)
    with path.open('a', encoding='utf-8-sig' if not exists else 'utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in columns})
        f.flush()
        os.fsync(f.fileno())


def update_status(schedule, run_id):
    # Atomic replace of schedule; backup is made before first change.
    backup = SCHEDULE_FILE.with_suffix('.csv.backup')
    if not backup.exists():
        import shutil
        shutil.copy2(SCHEDULE_FILE, backup)
    schedule.loc[schedule['run_id'].eq(run_id), 'status'] = 'completed'
    tmp = SCHEDULE_FILE.with_suffix('.csv.tmp')
    schedule.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, SCHEDULE_FILE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--max-calls', type=int, metavar='N', help='Max NEW API calls across runs')
    scope.add_argument('--all', action='store_true', help='Run all remaining requests (PAID)')
    parser.add_argument('--provider', choices=('qwen', 'mistral', 'openai'), help='Only run this provider')
    args = parser.parse_args()
    if args.max_calls is not None and args.max_calls < 1:
        parser.error('--max-calls must be positive')
    if not SCHEDULE_FILE.exists():
        raise FileNotFoundError(f'Missing {SCHEDULE_FILE}; run from project root')
    schedule = pd.read_csv(SCHEDULE_FILE)
    needed = {'run_id', 'model', 'language', 'response_mode', 'repetition', 'status'}
    if needed - set(schedule):
        raise ValueError(f'Schedule missing: {sorted(needed - set(schedule))}')
    if schedule['run_id'].duplicated().any():
        raise ValueError('Duplicate run_id in schedule')
    if len(schedule) != 180:
        raise ValueError(f'Expected 180 scheduled runs, found {len(schedule)}')
    if not set(schedule['model']).issubset(PROVIDER_DELAY):
        raise ValueError('Unknown provider in schedule')
    if not set(schedule['status']).issubset({'pending', 'completed'}):
        raise ValueError('Unknown status in schedule')

    total_missing = 0
    invalid_historical = 0
    plans = []
    for _, run in schedule.iterrows():
        requests = load_requests(run)
        ids = set(requests['request_id'].astype(str))
        output = result_path(run)
        done, invalid = get_successful_ids(output, ids, run['response_mode'])
        missing = requests.loc[~requests['request_id'].isin(done)]
        total_missing += len(missing)
        invalid_historical += invalid
        if len(missing) and (args.provider is None or run['model'] == args.provider):
            plans.append((run, missing, output, len(done)))
        elif run['status'] != 'completed':
            print(f'Already complete on disk: {run["run_id"]} (schedule status will update in live mode)')

    completed_on_disk = sum(1 for _, r in schedule.iterrows() if len(get_successful_ids(result_path(r), set(load_requests(r)['request_id']), r['response_mode'])[0]) == EXPECTED_PER_RUN)
    print(f'Schedule: {len(schedule)} runs; successful completed runs on disk: {completed_on_disk}')
    print(f'Remaining API responses (all providers): {total_missing}; historical invalid MC responses: {invalid_historical}')
    if args.provider:
        print(f'Provider filter: {args.provider}; eligible runs: {len(plans)}')
    if plans:
        first = plans[0]
        print(f'Next: {first[0]["run_id"]}, {len(first[1])} requests remaining')
    if not args.all and args.max_calls is None:
        print('PREFLIGHT ONLY: no API calls, no files modified.')
        return
    if args.all:
        confirmation = input(f'This can send up to {total_missing:,} PAID API calls. Type RUN ALL: ')
        if confirmation != 'RUN ALL':
            print('Cancelled.'); return
    providers = get_provider_functions()
    budget = float('inf') if args.all else args.max_calls
    used = 0
    consecutive_failures = 0
    for run, missing, output, already_done in plans:
        run_id, model, mode = run['run_id'], run['model'], run['response_mode']
        ensure_manifest(run)
        print(f'\n{run_id} {model}/{run["language"]}/{mode}: {len(missing)} remaining')
        for _, req in missing.iterrows():
            if used >= budget:
                print(f'Stopped at requested limit: {used} API calls.')
                return
            used += 1
            started = timestamp()
            raw, error, attempts, success = None, None, 0, False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                attempts = attempt
                try:
                    raw = providers[model](req['prompt_text'])
                    if not valid_answer(raw, mode):
                        raise ValueError(f'Invalid {mode} response: {str(raw)[:160]!r}')
                    success = True
                    error = None
                    break
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'[:1000]
                    if attempt < MAX_ATTEMPTS:
                        wait, rate_limited = retry_delay(exc, attempt)
                        reason = 'RATE LIMIT' if rate_limited else 'retry'
                        print(f'  {reason} {attempt}: waiting {wait:.0f}s: {error}')
                        time.sleep(wait)
            row = req.to_dict()
            row.update(run_id=run_id, model=model, response_raw=raw, success=success,
                       attempts=attempts, error_message=error,
                       started_at=started, finished_at=timestamp())
            append_row(output, row)
            print(f'  {used}: {req["request_id"]} -> {str(raw)[:60]!r} {"OK" if success else "FAILED"}')
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError('5 consecutive failures: stopped to protect API budget')
            time.sleep(PROVIDER_DELAY[model])
        good, _ = get_successful_ids(output, set(load_requests(run)['request_id']), mode)
        if len(good) == EXPECTED_PER_RUN:
            update_status(schedule, run_id)
            print(f'{run_id} COMPLETED')
        else:
            print(f'{run_id}: {len(good)}/{EXPECTED_PER_RUN} valid successful requests; can resume')
    print(f'Finished. New API calls this session: {used}')


if __name__ == '__main__':
    main()
