"""
generate_mock_csvs_group3_v3.py

Group 3 (distribution tables) with friendly naming.

Reads from: training_data/mock_csv_v3/
- ticket_verkaeufe.csv
- sonstige_angebote.csv
- plan_umsatz.csv
- regionen_bundesland.csv

Writes:
- ticket_verteilung_bundesland.csv
- angebot_verteilung_bundesland.csv
- plan_verteilung_bundesland.csv
- verteilung_kopf.csv
- verteilung_positionen.csv
"""

import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)

INPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_START = datetime(2024, 1, 1)
AUDIT_END = datetime(2026, 1, 1)

def write_csv_for_excel(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")

def random_datetime(start: datetime, end: datetime) -> str:
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")

def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["erstellt_von_user_id"] = [random.randint(1, 80) for _ in range(len(df))]
    df["erstellt_am"] = [random_datetime(AUDIT_START, AUDIT_END) for _ in range(len(df))]
    df["geaendert_von_user_id"] = [pd.NA if random.random() < 0.45 else random.randint(1, 80) for _ in range(len(df))]
    df["geaendert_am"] = [pd.NA if random.random() < 0.45 else random_datetime(AUDIT_START, AUDIT_END) for _ in range(len(df))]
    return df

def random_shares(total_items: int):
    raw = [random.random() for _ in range(total_items)]
    s = sum(raw)
    shares = [x / s for x in raw]
    shares = [round(x, 4) for x in shares]
    diff = round(1.0 - sum(shares), 4)
    shares[0] = round(shares[0] + diff, 4)
    return shares

regionen = pd.read_csv(INPUT_DIR / "regionen_bundesland.csv")
ticket_verkaeufe = pd.read_csv(INPUT_DIR / "ticket_verkaeufe.csv")
sonstige_angebote = pd.read_csv(INPUT_DIR / "sonstige_angebote.csv")
plan_umsatz = pd.read_csv(INPUT_DIR / "plan_umsatz.csv")

bundesland_codes = regionen["bundesland_code2"].astype(str).tolist()
bundesland_ids = regionen["bundesland_id"].astype(int).tolist()

# 1) ticket_verteilung_bundesland
ticket_context = (
    ticket_verkaeufe[["jahr", "monat", "tarifverbund_id", "ticket_code"]]
    .drop_duplicates()
    .sample(n=min(2000, len(ticket_verkaeufe)), random_state=42)
)

rows = []
for _, ctx in ticket_context.iterrows():
    chosen = random.sample(bundesland_codes, k=random.randint(3, 8))
    shares = random_shares(len(chosen))
    for code, share in zip(chosen, shares):
        rows.append((int(ctx["jahr"]), int(ctx["monat"]), int(ctx["tarifverbund_id"]), int(ctx["ticket_code"]), code, float(share)))

ticket_verteilung = pd.DataFrame(rows, columns=["jahr", "monat", "tarifverbund_id", "ticket_code", "bundesland_code2", "anteil"])
write_csv_for_excel(ticket_verteilung, OUTPUT_DIR / "ticket_verteilung_bundesland.csv")

# 2) angebot_verteilung_bundesland
angebot_context = (
    sonstige_angebote[["jahr", "monat", "tarifverbund_id", "angebot_gruppe"]]
    .drop_duplicates()
    .sample(n=min(1500, len(sonstige_angebote)), random_state=42)
)

rows = []
for _, ctx in angebot_context.iterrows():
    chosen = random.sample(bundesland_codes, k=random.randint(3, 8))
    shares = random_shares(len(chosen))
    for code, share in zip(chosen, shares):
        rows.append((int(ctx["jahr"]), int(ctx["monat"]), int(ctx["tarifverbund_id"]), int(ctx["angebot_gruppe"]), code, float(share)))

angebot_verteilung = pd.DataFrame(rows, columns=["jahr", "monat", "tarifverbund_id", "angebot_gruppe", "bundesland_code2", "anteil"])
write_csv_for_excel(angebot_verteilung, OUTPUT_DIR / "angebot_verteilung_bundesland.csv")

# 3) plan_verteilung_bundesland
plan_context = (
    plan_umsatz[["jahr", "monat", "tarifverbund_id"]]
    .drop_duplicates()
    .sample(n=min(1200, len(plan_umsatz)), random_state=42)
)

rows = []
for _, ctx in plan_context.iterrows():
    chosen = random.sample(bundesland_codes, k=random.randint(3, 8))
    shares = random_shares(len(chosen))
    for code, share in zip(chosen, shares):
        rows.append((int(ctx["jahr"]), int(ctx["monat"]), int(ctx["tarifverbund_id"]), code, float(share)))

plan_verteilung = pd.DataFrame(rows, columns=["jahr", "monat", "tarifverbund_id", "bundesland_code2", "anteil"])
write_csv_for_excel(plan_verteilung, OUTPUT_DIR / "plan_verteilung_bundesland.csv")

# 4) verteilung_kopf (header)
organisation_ids = list(range(1, 51))
KOPF_COUNT = 120
kopf_rows = []
for verteilung_id in range(1, KOPF_COUNT + 1):
    organisation_id = random.choice(organisation_ids)
    typ_code = random.choice([1, 2, 3])
    start_datum = datetime(2024, random.randint(1, 12), 1)
    end_datum = start_datum + timedelta(days=random.randint(60, 365))
    kopf_rows.append((verteilung_id, organisation_id, typ_code, start_datum.strftime("%Y-%m-%d"), end_datum.strftime("%Y-%m-%d")))

verteilung_kopf = pd.DataFrame(kopf_rows, columns=["verteilung_id", "organisation_id", "typ_code", "start_datum", "end_datum"])
verteilung_kopf = add_audit_columns(verteilung_kopf)
write_csv_for_excel(verteilung_kopf, OUTPUT_DIR / "verteilung_kopf.csv")

# 5) verteilung_positionen (detail)
pos_rows = []
pos_id = 1
for verteilung_id in verteilung_kopf["verteilung_id"].astype(int).tolist():
    chosen_ids = random.sample(bundesland_ids, k=random.randint(3, 8))
    shares = random_shares(len(chosen_ids))
    for bl_id, share in zip(chosen_ids, shares):
        pos_rows.append((pos_id, verteilung_id, bl_id, float(share)))
        pos_id += 1

verteilung_positionen = pd.DataFrame(pos_rows, columns=["position_id", "verteilung_id", "bundesland_id", "anteil"])
verteilung_positionen = add_audit_columns(verteilung_positionen)
write_csv_for_excel(verteilung_positionen, OUTPUT_DIR / "verteilung_positionen.csv")

print("Group 3 done. Output folder:", OUTPUT_DIR)
