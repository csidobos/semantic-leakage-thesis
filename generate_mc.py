import random
import pandas as pd


# -------------------------
# SETTINGS
# -------------------------

LANGUAGES = {
    "HU": {
        "master_file": "data/stimulus_master_HU.xlsx",
        "free_text_preview": "data/HU_free_text_preview.csv",
        "mc_output": "data/HU_mc_requests.csv",
        "mc_unknown_output": "data/HU_mc_unknown_requests.csv",
        "instruction": (
            "Válaszd ki a szöveg legjobb kiegészítését az alábbi lehetőségek közül. "
            "A válaszod kizárólag a választott opció betűjele legyen. "
            "Ne adj magyarázatot vagy indoklást."
        ),
        "unknown_text": "Nem állapítható meg",
    },

    "EN": {
        "master_file": "data/stimulus_master_EN.xlsx",
        "free_text_preview": "data/EN_free_text_preview.csv",
        "mc_output": "data/EN_mc_requests.csv",
        "mc_unknown_output": "data/EN_mc_unknown_requests.csv",
        "instruction": (
            "Choose the best completion for the text from the options below. "
            "Your response should contain only the letter of the selected option. "
            "Do not provide an explanation or justification."
        ),
        "unknown_text": "Cannot be determined",
    },
}

REPETITIONS = range(1, 11)
RANDOM_SEED = 42


# -------------------------
# Segédfüggvény
# -------------------------

def make_prompt(stimulus_text, option_mapping, instruction):

    option_lines = []

    for letter in sorted(option_mapping.keys()):
        option_lines.append(
            f"{letter}: {option_mapping[letter]['text']}"
        )

    return (
        f"{instruction}\n\n"
        f"{stimulus_text}\n\n"
        + "\n".join(option_lines)
    )


# -------------------------
# GENERATE ONE LANGUAGE
# -------------------------

def generate_language(language, config):

    print()
    print(f"--- {language} ---")

    # -------------------------
    # Adatok betöltése
    # -------------------------

    master = pd.read_excel(config["master_file"])
    stimuli = pd.read_csv(config["free_text_preview"])

    assert (master["language"] == language).all(), (
        f"HIBA: a {language} master language oszlopa nem mindenhol {language}."
    )

    assert (stimuli["language"] == language).all(), (
        f"HIBA: a {language} preview language oszlopa nem mindenhol {language}."
    )

    options = master[
        ["item_id", "Sztereotip", "Anti", "Semleges"]
    ].copy()

    stimuli = stimuli.merge(
        options,
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    if stimuli[
        ["Sztereotip", "Anti", "Semleges"]
    ].isna().any().any():
        raise ValueError(
            "Van olyan stimulus, amelyhez hiányzik MC válaszopció."
        )

    # -------------------------
    # MC requestek
    # -------------------------

    mc_rows = []
    mc_unknown_rows = []

    mc_request_number = 1
    mc_unknown_request_number = 1

    for repetition in REPETITIONS:

        for _, row in stimuli.iterrows():

            # -------------------------
            # Sima MC
            # -------------------------

            mc_options = [
                {
                    "class": "Sztereotíp",
                    "text": row["Sztereotip"],
                },
                {
                    "class": "Anti",
                    "text": row["Anti"],
                },
                {
                    "class": "Semleges",
                    "text": row["Semleges"],
                },
            ]

            mc_seed = (
                f"{RANDOM_SEED}|mc|{row['item_id']}|"
                f"{row['condition']}|{row['paraphrase']}|"
                f"{row['distance']}|{repetition}"
            )

            mc_rng = random.Random(mc_seed)
            mc_rng.shuffle(mc_options)

            mc_mapping = {
                "A": mc_options[0],
                "B": mc_options[1],
                "C": mc_options[2],
            }

            mc_prompt = make_prompt(
                row["stimulus_text"],
                mc_mapping,
                config["instruction"],
            )

            mc_rows.append({
                "request_id":
                    f"{language}_MC_REQ_{mc_request_number:06d}",
                "repetition": repetition,
                "item_id": row["item_id"],
                "domain": row["domain"],
                "language": row["language"],
                "dimension": row["dimension"],
                "group": row["group"],
                "condition": row["condition"],
                "paraphrase": row["paraphrase"],
                "distance": row["distance"],
                "response_mode": "mc",
                "stimulus_text": row["stimulus_text"],

                "option_A": mc_mapping["A"]["text"],
                "option_A_class": mc_mapping["A"]["class"],
                "option_B": mc_mapping["B"]["text"],
                "option_B_class": mc_mapping["B"]["class"],
                "option_C": mc_mapping["C"]["text"],
                "option_C_class": mc_mapping["C"]["class"],

                "prompt_text": mc_prompt,
            })

            mc_request_number += 1

            # -------------------------
            # MC + Unknown
            # -------------------------

            unknown_options = [
                {
                    "class": "Sztereotíp",
                    "text": row["Sztereotip"],
                },
                {
                    "class": "Anti",
                    "text": row["Anti"],
                },
                {
                    "class": "Semleges",
                    "text": row["Semleges"],
                },
                {
                    "class": "Unknown",
                    "text": config["unknown_text"],
                },
            ]

            unknown_seed = (
                f"{RANDOM_SEED}|mc_unknown|{row['item_id']}|"
                f"{row['condition']}|{row['paraphrase']}|"
                f"{row['distance']}|{repetition}"
            )

            unknown_rng = random.Random(unknown_seed)
            unknown_rng.shuffle(unknown_options)

            unknown_mapping = {
                "A": unknown_options[0],
                "B": unknown_options[1],
                "C": unknown_options[2],
                "D": unknown_options[3],
            }

            unknown_prompt = make_prompt(
                row["stimulus_text"],
                unknown_mapping,
                config["instruction"],
            )

            mc_unknown_rows.append({
                "request_id":
                    f"{language}_MCU_REQ_{mc_unknown_request_number:06d}",
                "repetition": repetition,
                "item_id": row["item_id"],
                "domain": row["domain"],
                "language": row["language"],
                "dimension": row["dimension"],
                "group": row["group"],
                "condition": row["condition"],
                "paraphrase": row["paraphrase"],
                "distance": row["distance"],
                "response_mode": "mc_unknown",
                "stimulus_text": row["stimulus_text"],

                "option_A": unknown_mapping["A"]["text"],
                "option_A_class": unknown_mapping["A"]["class"],
                "option_B": unknown_mapping["B"]["text"],
                "option_B_class": unknown_mapping["B"]["class"],
                "option_C": unknown_mapping["C"]["text"],
                "option_C_class": unknown_mapping["C"]["class"],
                "option_D": unknown_mapping["D"]["text"],
                "option_D_class": unknown_mapping["D"]["class"],

                "prompt_text": unknown_prompt,
            })

            mc_unknown_request_number += 1

    # -------------------------
    # DataFrame-ek
    # -------------------------

    mc_df = pd.DataFrame(mc_rows)
    mc_unknown_df = pd.DataFrame(mc_unknown_rows)

    # -------------------------
    # Ellenőrzések
    # -------------------------

    expected = len(stimuli) * len(REPETITIONS)

    assert len(mc_df) == expected
    assert len(mc_unknown_df) == expected

    assert not mc_df["request_id"].duplicated().any()
    assert not mc_unknown_df["request_id"].duplicated().any()

    # Ugyanaz a stimulus/repetition csak egyszer szerepelhet
    key_columns = [
        "item_id",
        "condition",
        "paraphrase",
        "distance",
        "repetition",
    ]

    assert not mc_df.duplicated(
        subset=key_columns
    ).any()

    assert not mc_unknown_df.duplicated(
        subset=key_columns
    ).any()

    # -------------------------
    # Mentés
    # -------------------------

    mc_df.to_csv(
        config["mc_output"],
        index=False,
        encoding="utf-8-sig",
    )

    mc_unknown_df.to_csv(
        config["mc_unknown_output"],
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------
    # Eredmény
    # -------------------------

    print(f"Stimuli: {len(stimuli)}")
    print(f"Repetitions: {len(REPETITIONS)}")
    print(f"MC requests: {len(mc_df)}")
    print(f"MC + unknown requests: {len(mc_unknown_df)}")
    print(f"Saved: {config['mc_output']}")
    print(f"Saved: {config['mc_unknown_output']}")

    print()
    print("Example MC prompt:")
    print("-------------------------")
    print(mc_df.iloc[0]["prompt_text"])

    print()
    print("Option classes:")
    print(
        "A =", mc_df.iloc[0]["option_A_class"],
        "| B =", mc_df.iloc[0]["option_B_class"],
        "| C =", mc_df.iloc[0]["option_C_class"],
    )


# -------------------------
# Mindkét nyelv generálása
# -------------------------

for language, config in LANGUAGES.items():
    generate_language(language, config)


print()
print("-------------------------")
print("MC GENERATION COMPLETE")
print("-------------------------")


