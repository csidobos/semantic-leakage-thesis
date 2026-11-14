
"""Independent, resumable Grok worker.

Run from thesis project root:

    python run_grok_experiment_fixed.py
        Read-only preflight.

    python run_grok_experiment_fixed.py --pilot
        Up to 6 new requests, one per language/mode.

    python run_grok_experiment_fixed.py --max-calls 10
        At most 10 paid requests.

    python run_grok_experiment_fixed.py --all
        Run all remaining requests after confirmation.

Does not modify the original schedule or other providers' results.
"""

import argparse
import json
import time
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from run_experiment_parallel import (
    EXPECTED_PER_RUN,
    append_row,
    load_requests,
    retry_delay,
    timestamp,
    valid_answer,
)

MODEL = "grok-4.20-0309-non-reasoning"

OUT = Path("results/grok/raw")
CONFIG = Path("results/grok/config")

MODES = ("free_text", "mc", "mc_unknown")
LANGUAGES = ("HU", "EN")
REPETITIONS = range(1, 11)


# --------------------------------------------------
# 1. ANSWER NORMALIZATION
# --------------------------------------------------

def normalize_answer(raw, mode):
    """Extract an unambiguous MC letter without changing raw text."""

    if not isinstance(raw, str):
        return raw

    answer = raw.strip()

    if mode not in ("mc", "mc_unknown"):
        return answer

    allowed = "ABC" if mode == "mc" else "ABCD"

    # Plain or bold single letter: A, **A**
    plain = re.fullmatch(
        rf"\**\s*([{allowed}])\s*\**\.?",
        answer,
        flags=re.IGNORECASE,
    )

    if plain:
        return plain.group(1).upper()

    # Repeated identical choice:
    # A
    #
    # **Válasz:** A
    repeated = re.fullmatch(
        rf"\**\s*([{allowed}])\s*\**\s+"
        rf"\**(?:Válasz|Válaszom|Answer|My answer):\**\s*"
        rf"\**\s*([{allowed}])\s*\**",
        answer,
        flags=re.IGNORECASE,
    )

    if repeated:
        first = repeated.group(1).upper()
        second = repeated.group(2).upper()

        if first == second:
            return first

    return answer


def is_valid_answer(raw, mode):
    normalized = normalize_answer(raw, mode)
    return valid_answer(normalized, mode)


# --------------------------------------------------
# 2. EXISTING RESULTS
# --------------------------------------------------

def grok_successful_ids(path, expected_ids, mode):
    """Find successful and recoverable historical responses."""

    if not path.exists():
        return set(), 0

    df = pd.read_csv(
        path,
        dtype={"request_id": str},
        low_memory=False,
    )

    required = {
        "request_id",
        "success",
        "response_raw",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns in existing results: {path}"
        )

    valid = df["response_raw"].apply(
        lambda raw: is_valid_answer(raw, mode)
    )

    successful = (
        df["success"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # Only recover recorded format-validation failures.
    if "error_message" in df.columns:
        format_failure = (
            df["error_message"]
            .fillna("")
            .astype(str)
            .str.match(
                r"^(?:ValueError:\s*)?Invalid "
            )
        )
    else:
        format_failure = pd.Series(
            False,
            index=df.index,
        )

    recovered = valid & ~successful & format_failure

    done = set(
        df.loc[
            valid & (successful | recovered),
            "request_id",
        ].astype(str)
    )

    unexpected = done - expected_ids

    if unexpected:
        raise ValueError(
            f"{path}: {len(unexpected)} unexpected request IDs"
        )

    return done, int(recovered.sum())


# --------------------------------------------------
# 3. RUN DEFINITIONS
# --------------------------------------------------

def runs():

    for lang in LANGUAGES:
        for mode in MODES:
            for rep in REPETITIONS:

                run_id = (
                    f"GROK_{lang}_{mode}_rep{rep:02d}"
                )

                run = {
                    "run_id": run_id,
                    "model": "grok",
                    "language": lang,
                    "response_mode": mode,
                    "repetition": rep,
                }

                yield run, OUT / f"{run_id}.csv"


# --------------------------------------------------
# 4. CONFIGURATION MANIFEST
# --------------------------------------------------

def manifest(run):

    CONFIG.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = CONFIG / (
        f"{run['run_id']}_settings.json"
    )

    expected = {
        "run_id": run["run_id"],
        "provider": "grok",
        "model_id": MODEL,
        "language": run["language"],
        "response_mode": run["response_mode"],
        "repetition": run["repetition"],
        "reasoning_request": "non-reasoning model",
        "temperature_request": "omitted (provider default)",
        "temperature_resolved": "not verified by API response",
        "seed": 42,
        "request_source": "original data/*_requests.csv",
    }

    if path.exists():

        existing = json.loads(
            path.read_text(encoding="utf-8")
        )

        if existing != expected:
            raise ValueError(
                f"Configuration mismatch: {path}"
            )

    else:

        with path.open(
            "x",
            encoding="utf-8",
        ) as f:

            json.dump(
                expected,
                f,
                ensure_ascii=False,
                indent=2,
            )

            f.write("\n")


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--pilot",
        action="store_true",
        help="Up to 6 paid calls",
    )

    group.add_argument(
        "--max-calls",
        type=int,
        metavar="N",
        help="At most N new requests",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Run all remaining requests",
    )

    args = parser.parse_args()

    if (
        args.max_calls is not None
        and args.max_calls < 1
    ):
        parser.error(
            "--max-calls must be positive"
        )

    # ----------------------------------------------
    # BUILD PREFLIGHT PLAN
    # ----------------------------------------------

    plans = []

    missing_total = 0
    recovered_total = 0

    for run, output in runs():

        requests = load_requests(run)

        expected_ids = set(
            requests["request_id"].astype(str)
        )

        done, recovered = grok_successful_ids(
            output,
            expected_ids,
            run["response_mode"],
        )

        recovered_total += recovered

        missing = requests.loc[
            ~requests["request_id"].astype(str).isin(done)
        ]

        plans.append(
            (run, output, missing, len(done))
        )

        missing_total += len(missing)

    total_expected = (
        len(LANGUAGES)
        * len(MODES)
        * len(REPETITIONS)
        * EXPECTED_PER_RUN
    )

    print("\nGROK PREFLIGHT")
    print("-" * 65)

    print(
        f"Valid unique responses: "
        f"{total_expected - missing_total:,}"
        f"/{total_expected:,}"
    )

    print(
        f"Missing responses: {missing_total:,}"
    )

    complete_blocks = sum(
        done_count == EXPECTED_PER_RUN
        for _, _, _, done_count in plans
    )

    print(
        f"Complete blocks: "
        f"{complete_blocks}/{len(plans)}"
    )

    print(
        "Historical format-failure rows "
        f"recovered: {recovered_total:,}"
    )

    # ----------------------------------------------
    # SUMMARY BY LANGUAGE AND MODE
    # ----------------------------------------------

    print("\nMISSING BY LANGUAGE AND MODE")
    print("-" * 65)

    summary = defaultdict(
        lambda: [0, 0]
    )

    for run, output, missing, done_count in plans:

        key = (
            run["language"],
            run["response_mode"],
        )

        summary[key][0] += done_count
        summary[key][1] += len(missing)

    print(
        f"{'LANG':5} {'MODE':12} "
        f"{'VALID':>8} {'MISSING':>8}"
    )

    for (language, mode), (
        done, missing
    ) in sorted(summary.items()):

        print(
            f"{language:5} "
            f"{mode:12} "
            f"{done:8} "
            f"{missing:8}"
        )

    # ----------------------------------------------
    # INCOMPLETE BLOCKS
    # ----------------------------------------------

    print("\nINCOMPLETE BLOCKS")
    print("-" * 65)

    incomplete = [
        (run, len(missing))
        for run, _, missing, _ in plans
        if len(missing) > 0
    ]

    if not incomplete:
        print("None")
    else:
        for run, count in incomplete:
            print(
                f"{run['run_id']:32} "
                f"missing={count}"
            )

    # ----------------------------------------------
    # READ-ONLY MODE
    # ----------------------------------------------

    if not (
        args.pilot
        or args.max_calls
        or args.all
    ):

        print(
            "\nPREFLIGHT ONLY: "
            "no API calls, no files changed."
        )

        return

    # ----------------------------------------------
    # CONFIRM FULL RUN
    # ----------------------------------------------

    if args.all:

        confirmation = input(
            f"Up to {missing_total:,} PAID "
            "Grok requests. Type RUN ALL: "
        )

        if confirmation != "RUN ALL":
            print("Cancelled.")
            return

    # ----------------------------------------------
    # SELECT WORK
    # ----------------------------------------------

    if args.pilot:

        chosen = []

        for lang in LANGUAGES:
            for mode in MODES:

                matches = [
                    (r, p, m.head(1))
                    for r, p, m, _ in plans
                    if (
                        r["language"] == lang
                        and r["response_mode"] == mode
                        and len(m) > 0
                    )
                ]

                if matches:
                    chosen.append(matches[0])

        work = chosen
        budget = len(chosen)

    else:

        work = [
            (r, p, m)
            for r, p, m, _ in plans
            if len(m) > 0
        ]

        budget = (
            float("inf")
            if args.all
            else args.max_calls
        )

    # ----------------------------------------------
    # API CLIENT
    # ----------------------------------------------

    from api.grok_api import get_response
    import api.grok_api as grok_api

    if grok_api.MODEL != MODEL:
        raise ValueError(
            f"API model {grok_api.MODEL!r} "
            f"differs from {MODEL!r}"
        )

    used = 0
    consecutive_failures = 0

    # ----------------------------------------------
    # EXECUTE REQUESTS
    # ----------------------------------------------

    for run, output, missing in work:

        if used >= budget:
            break

        manifest(run)

        print(
            f"\n{run['run_id']}: "
            f"{len(missing)} queued"
        )

        for _, req in missing.iterrows():

            if used >= budget:
                break

            used += 1

            started = timestamp()

            raw = None
            normalized = None
            error = None
            attempts = 0
            success = False

            for attempt in range(1, 5):

                attempts = attempt

                try:

                    raw = get_response(
                        req["prompt_text"]
                    )

                    normalized = normalize_answer(
                        raw,
                        run["response_mode"],
                    )

                    if not valid_answer(
                        normalized,
                        run["response_mode"],
                    ):

                        error = (
                            f"Invalid "
                            f"{run['response_mode']} "
                            f"response: "
                            f"{str(raw)[:160]!r}"
                        )

                        break

                    success = True
                    error = None

                    break

                except Exception as exc:

                    error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )[:1000]

                    if attempt < 4:

                        wait, _ = retry_delay(
                            exc,
                            attempt,
                        )

                        print(
                            f"Retry {attempt}: "
                            f"waiting {wait:.0f}s: "
                            f"{error}"
                        )

                        time.sleep(wait)

            row = req.to_dict()

            row.update(
                run_id=run["run_id"],
                model="grok",
                response_raw=raw,
                response_normalized=normalized,
                success=success,
                attempts=attempts,
                error_message=error,
                started_at=started,
                finished_at=timestamp(),
            )

            append_row(output, row)

            print(
                f"{used}: "
                f"{req['request_id']} -> "
                f"{str(raw)[:100]!r} "
                f"{'OK' if success else 'FAILED'}"
            )

            if success:

                consecutive_failures = 0

            else:

                consecutive_failures += 1

                if consecutive_failures >= 5:
                    raise RuntimeError(
                        "5 consecutive failures: "
                        "stopped to protect API budget"
                    )

    print(
        f"\nFinished. "
        f"New requests attempted: {used}"
    )

    print(
        "Original schedule and other providers' "
        "results were not modified."
    )


if __name__ == "__main__":
    main()
