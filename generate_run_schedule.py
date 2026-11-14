import random
import pandas as pd


# -------------------------
# SETTINGS
# -------------------------

MODELS = [
    "openai",
    "mistral",
    "qwen",
]

LANGUAGES = [
    "HU",
    "EN",
]

RESPONSE_MODES = [
    "free_text",
    "mc",
    "mc_unknown",
]

REPETITIONS = range(1, 11)

RANDOM_SEED = 42

OUTPUT_FILE = "data/run_schedule.csv"


# -------------------------
# Run schedule elkészítése
# -------------------------

random.seed(RANDOM_SEED)

runs = []
run_number = 1

for repetition in REPETITIONS:

    # Egy repetitionön belül minden
    # model × language × response_mode kombináció
    combinations = []

    for model in MODELS:
        for language in LANGUAGES:
            for response_mode in RESPONSE_MODES:
                combinations.append({
                    "model": model,
                    "language": language,
                    "response_mode": response_mode,
                })

    # A 18 kombináció sorrendjének randomizálása
    random.shuffle(combinations)

    for combo in combinations:

        runs.append({
            "run_id": f"RUN_{run_number:03d}",
            "repetition": repetition,
            "model": combo["model"],
            "language": combo["language"],
            "response_mode": combo["response_mode"],
            "status": "pending",
        })

        run_number += 1


# -------------------------
# Mentés
# -------------------------

schedule = pd.DataFrame(runs)

schedule.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("-------------------------")
print("RUN SCHEDULE CREATED")
print(f"Runs: {len(schedule)}")
print(f"Saved to: {OUTPUT_FILE}")
print()
print(schedule.head(10).to_string(index=False))