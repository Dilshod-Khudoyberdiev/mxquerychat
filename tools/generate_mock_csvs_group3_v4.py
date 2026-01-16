"""
generate_mock_csvs_group3_v4.py

Group 3 = Distribution / allocation tables (shares)
Creates these CSVs in training_data/mock_csv_v3/:

- ticket_verteilung_bundesland.csv
- angebot_verteilung_bundesland.csv
- plan_verteilung_bundesland.csv
- verteilung_kopf.csv
- verteilung_positionen.csv

Depends on Group 1 + 2 CSVs.
Run:
  python tools/generate_mock_csvs_group3_v4.py
"""

import random
from pathlib import Path
from datetime import date

import pandas as pd

random.seed(42)

INPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def write_csv_for_excel(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")

# Load references
bundesland_df = pd.read_csv(INPUT_DIR / "regionen_bundesland.csv")
ticket_produkte_df = pd.read_csv(INPUT_DIR / "ticket_produkte.csv")
tarifverbuende_df = pd.read_csv(INPUT_DIR / "tarifverbuende.csv")

bundesland_codes2 = bundesland_df["bundesland_code2"].tolist()
bundesland_ids = bundesland_df["bundesland_id"].astype(int).tolist()
ticket_codes = ticket_produkte_df["ticket_code"].astype(int).tolist()
tarif_ids = tarifverbuende_df["tarifverbund_id"].astype(int).tolist()

years = [2024, 2025]
months = list(range(1, 13))

def random_shares(n: int):
    """Create n shares that sum to 1.0."""
    values = [random.random() for _ in range(n)]
    total = sum(values)
    shares = [v / total for v in values]
    # Round nicely but keep sum ~1 (good enough for mock data)
    shares = [round(s, 4) for s in shares]
    diff = round(1.0 - sum(shares), 4)
    shares[-1] = round(shares[-1] + diff, 4)
    return shares

# -----------------------
# 1) ticket_verteilung_bundesland.csv
# keys: jahr, monat, tarifverbund_id, ticket_code + bundesland_code2 + anteil
# -----------------------
ticket_rows = []
for jahr in years:
    for monat in months:
        for tarifverbund_id in random.sample(tarif_ids, k=min(8, len(tarif_ids))):
            for ticket_code in random.sample(ticket_codes, k=min(5, len(ticket_codes))):
                selected_states = random.sample(bundesland_codes2, k=random.randint(3, 7))
                shares = random_shares(len(selected_states))
                for bl_code2, anteil in zip(selected_states, shares):
                    ticket_rows.append((jahr, monat, tarifverbund_id, ticket_code, bl_code2, anteil))

ticket_verteilung_df = pd.DataFrame(
    ticket_rows,
    columns=["jahr", "monat", "tarifverbund_id", "ticket_code", "bundesland_code2", "anteil"],
)
write_csv_for_excel(ticket_verteilung_df, OUTPUT_DIR / "ticket_verteilung_bundesland.csv")

# -----------------------
# 2) angebot_verteilung_bundesland.csv
# keys: jahr, monat, tarifverbund_id, angebot_gruppe + bundesland_code2 + anteil
# -----------------------
angebot_gruppen = [100, 101, 102, 200, 201]
angebot_rows = []

for jahr in years:
    for monat in months:
        for tarifverbund_id in random.sample(tarif_ids, k=min(8, len(tarif_ids))):
            for angebot_gruppe in random.sample(angebot_gruppen, k=3):
                selected_states = random.sample(bundesland_codes2, k=random.randint(3, 7))
                shares = random_shares(len(selected_states))
                for bl_code2, anteil in zip(selected_states, shares):
                    angebot_rows.append((jahr, monat, tarifverbund_id, angebot_gruppe, bl_code2, anteil))

angebot_verteilung_df = pd.DataFrame(
    angebot_rows,
    columns=["jahr", "monat", "tarifverbund_id", "angebot_gruppe", "bundesland_code2", "anteil"],
)
write_csv_for_excel(angebot_verteilung_df, OUTPUT_DIR / "angebot_verteilung_bundesland.csv")

# -----------------------
# 3) plan_verteilung_bundesland.csv
# keys: jahr, monat, tarifverbund_id + bundesland_code2 + anteil
# -----------------------
plan_rows = []
for jahr in years:
    for monat in months:
        for tarifverbund_id in random.sample(tarif_ids, k=min(8, len(tarif_ids))):
            selected_states = random.sample(bundesland_codes2, k=random.randint(3, 7))
            shares = random_shares(len(selected_states))
            for bl_code2, anteil in zip(selected_states, shares):
                plan_rows.append((jahr, monat, tarifverbund_id, bl_code2, anteil))

plan_verteilung_df = pd.DataFrame(
    plan_rows,
    columns=["jahr", "monat", "tarifverbund_id", "bundesland_code2", "anteil"],
)
write_csv_for_excel(plan_verteilung_df, OUTPUT_DIR / "plan_verteilung_bundesland.csv")

# -----------------------
# 4) verteilung_kopf.csv and 5) verteilung_positionen.csv
# Generic header/detail distribution model
# -----------------------
kopf_rows = []
pos_rows = []

verteilung_id = 1
pos_id = 1

verteilung_typen = ["ticket", "angebot", "plan"]

for _ in range(80):  # 80 distribution rules
    typ = random.choice(verteilung_typen)
    tarifverbund_id = random.choice(tarif_ids)

    start_year = random.choice(years)
    start_month = random.randint(1, 12)
    end_year = start_year
    end_month = min(12, start_month + random.randint(0, 5))

    start_date = date(start_year, start_month, 1).strftime("%Y-%m-%d")
    end_date = date(end_year, end_month, 28).strftime("%Y-%m-%d")

    kopf_rows.append((verteilung_id, typ, tarifverbund_id, start_date, end_date))

    # positions: which states + shares
    selected_ids = random.sample(bundesland_ids, k=random.randint(3, 7))
    shares = random_shares(len(selected_ids))

    for bl_id, anteil in zip(selected_ids, shares):
        pos_rows.append((pos_id, verteilung_id, bl_id, anteil))
        pos_id += 1

    verteilung_id += 1

verteilung_kopf_df = pd.DataFrame(
    kopf_rows,
    columns=["verteilung_id", "typ", "tarifverbund_id", "start_datum", "end_datum"],
)
verteilung_positionen_df = pd.DataFrame(
    pos_rows,
    columns=["position_id", "verteilung_id", "bundesland_id", "anteil"],
)

write_csv_for_excel(verteilung_kopf_df, OUTPUT_DIR / "verteilung_kopf.csv")
write_csv_for_excel(verteilung_positionen_df, OUTPUT_DIR / "verteilung_positionen.csv")

print("✅ Group 3 done. Generated distribution tables in:", OUTPUT_DIR)
print(" - ticket_verteilung_bundesland.csv")
print(" - angebot_verteilung_bundesland.csv")
print(" - plan_verteilung_bundesland.csv")
print(" - verteilung_kopf.csv")
print(" - verteilung_positionen.csv")
