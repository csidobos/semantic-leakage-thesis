import pandas as pd


# -----------------------------
# SETTINGS
# -----------------------------

LANGUAGES = {
    "HU": {
        "input_file": "data/stimulus_master_HU.xlsx",
        "preview_file": "data/HU_free_text_preview.csv",
        "request_file": "data/HU_free_text_requests.csv",
        "instruction": (
            "Egészítsd ki a következő szöveget egyetlen szóval vagy legfeljebb "
            "néhány szóból álló rövid kifejezéssel. A válaszod kizárólag a "
            "hiányzó rész legyen. Ne adj magyarázatot, indoklást vagy teljes mondatot."
        ),
    },
    "EN": {
        "input_file": "data/stimulus_master_EN.xlsx",
        "preview_file": "data/EN_free_text_preview.csv",
        "request_file": "data/EN_free_text_requests.csv",
        "instruction": (
            "Complete the following text with a single word or a short phrase "
            "of no more than a few words. Your response should contain only "
            "the missing part. Do not provide an explanation, justification, "
            "or a complete sentence."
        ),
    },
}

CONDITIONS = [
    "target",
    "comparison",
    "neutral",
    "pseudo",
    "nonconnected",
    "nonsocial",
    "implicit",
]

PARAPHRASES = ["p0", "p1", "p2"]
DISTANCES = [0, 1, 2]
REPETITIONS = 10


# -----------------------------
# GEENERATE LANGUAGE
# -----------------------------

def generate_language(language, config):

    print()
    print(f"--- {language} ---")

    input_file = config["input_file"]
    preview_file = config["preview_file"]
    request_file = config["request_file"]
    instruction = config["instruction"]

    # -----------------------------
    # MASTER FILE READING
    # -----------------------------

    df = pd.read_excel(input_file)

    print(f"Beolvasott itemek száma: {len(df)}")

    assert (df["language"] == language).all(), (
        f"HIBA: a {language} master language oszlopa nem mindenhol {language}."
    )

    # -----------------------------
    # GENERATE STIMULI
    # -----------------------------

    rows = []

    for _, item in df.iterrows():

        for condition in CONDITIONS:

            for paraphrase in PARAPHRASES:

                context_column = f"{condition}_{paraphrase}"
                context = item[context_column]

                completion = item["completion_sentence"]

                filler_1 = item[f"filler_1_{paraphrase}"]
                filler_2 = item[f"filler_2_{paraphrase}"]

                for distance in DISTANCES:

                    if distance == 0:
                        stimulus_text = f"{context} {completion}"

                    elif distance == 1:
                        stimulus_text = (
                            f"{context} {filler_1} {completion}"
                        )

                    elif distance == 2:
                        stimulus_text = (
                            f"{context} {filler_1} {filler_2} {completion}"
                        )

                    prompt_text = (
                        f"{instruction}\n\n"
                        f"{stimulus_text}"
                    )

                    rows.append({
                        "item_id": item["item_id"],
                        "domain": item["domain"],
                        "language": item["language"],
                        "dimension": item["dimension"],
                        "group": item["group"],
                        "condition": condition,
                        "paraphrase": paraphrase,
                        "distance": distance,
                        "stimulus_text": stimulus_text,
                        "prompt_text": prompt_text,
                    })

    # -----------------------------
    # STIMULI TABLE
    # -----------------------------

    output = pd.DataFrame(rows)

    print(f"Generált stimulusok száma: {len(output)}")

    expected_rows = (
        len(df)
        * len(CONDITIONS)
        * len(PARAPHRASES)
        * len(DISTANCES)
    )

    assert len(output) == expected_rows, (
        f"HIBA: {expected_rows} sor helyett "
        f"{len(output)} sor generálódott."
    )

    assert not output.duplicated(
        subset=[
            "item_id",
            "condition",
            "paraphrase",
            "distance"
        ]
    ).any(), "HIBA: duplikált stimulus található."

    assert output["prompt_text"].notna().all(), (
        "HIBA: üres prompt található."
    )

    output.to_csv(
        preview_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Kész: {preview_file}")

    # -----------------------------
    # 10 REPETITION
    # -----------------------------

    requests = []

    for repetition in range(1, REPETITIONS + 1):

        for _, row in output.iterrows():

            request = row.to_dict()

            request["repetition"] = repetition

            requests.append(request)

    requests = pd.DataFrame(requests)

    # -----------------------------
    # REQUEST ID
    # -----------------------------

    requests.insert(
        0,
        "request_id",
        [
            f"{language}_FT_REQ_{i:06d}"
            for i in range(1, len(requests) + 1)
        ]
    )

    # -----------------------------
    # REQUESTS CHECK
    # -----------------------------

    expected_requests = len(output) * REPETITIONS

    print(f"Generált requestek száma: {len(requests)}")

    assert len(requests) == expected_requests, (
        f"HIBA: {expected_requests} request helyett "
        f"{len(requests)} készült."
    )

    assert requests["request_id"].is_unique, (
        "HIBA: a request_id nem egyedi."
    )

    assert not requests.duplicated(
        subset=[
            "item_id",
            "condition",
            "paraphrase",
            "distance",
            "repetition"
        ]
    ).any(), (
        "HIBA: ugyanaz a stimulus ugyanabban az "
        "ismétlésben többször szerepel."
    )

    # -----------------------------
    # REQUEST TABLE SAVING
    # -----------------------------

    requests.to_csv(
        request_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("Request ellenőrzések rendben.")
    print(f"Kész: {request_file}")


# -----------------------------
# GENERATE BOTH LANGUAGES
# -----------------------------

for language, config in LANGUAGES.items():
    generate_language(language, config)

print()
print("FREE-TEXT GENERATION COMPLETE")