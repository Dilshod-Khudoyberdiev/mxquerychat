"""
generate_mock_csvs_group2_v4.py

Group 2 (fact tables) with friendly naming.
Reads Group 1 from: training_data/mock_csv_v3/
Writes into: training_data/mock_csv_v3/

Tables:
- ticket_verkaeufe.csv
- sonstige_angebote.csv
- plan_umsatz.csv
"""

import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)

INPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def write_csv_for_excel(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")

def random_date(start: datetime, end: datetime) -> str:
    delta = end - start
    days = random.randint(0, delta.days)
    return (start + timedelta(days=days)).strftime("%Y-%m-%d")

# Load Group 1 reference tables
postleitzahlen = pd.read_csv(INPUT_DIR / "postleitzahlen.csv")
meldestellen = pd.read_csv(INPUT_DIR / "meldestellen.csv")
tarifverbuende = pd.read_csv(INPUT_DIR / "tarifverbuende.csv")
ticket_produkte = pd.read_csv(INPUT_DIR / "ticket_produkte.csv")

plz_codes = postleitzahlen["plz"].dropna().astype(str).tolist()
meldestelle_codes = meldestellen["meldestelle_code"].dropna().astype(str).tolist()
tarifverbund_ids = tarifverbuende["tarifverbund_id"].dropna().astype(int).tolist()
ticket_codes = ticket_produkte["ticket_code"].dropna().astype(int).tolist()

# 1) ticket_verkaeufe (was d_tickets)
start_period = datetime(2024, 1, 1)
end_period = datetime(2025, 12, 31)

TARGET_ROWS = 15000
rows = []
for _ in range(TARGET_ROWS):
    jahr = random.choice([2024, 2025])
    monat = random.randint(1, 12)

    meldestelle_code = random.choice(meldestelle_codes)
    tarifverbund_id = random.choice(tarifverbund_ids)
    ticket_code = random.choice(ticket_codes)

    anzahl = random.randint(5, 250)

    match = ticket_produkte[ticket_produkte["ticket_code"] == ticket_code].iloc[0]
    preis_eur = float(match["preis_eur"])

    umsatz_eur = round(anzahl * preis_eur, 2)
    gueltig_ab = random_date(start_period, end_period)
    plz = random.choice(plz_codes)

    rows.append((monat, jahr, meldestelle_code, tarifverbund_id, ticket_code, anzahl, preis_eur, umsatz_eur, gueltig_ab, plz))

ticket_verkaeufe = pd.DataFrame(
    rows,
    columns=["monat", "jahr", "meldestelle_code", "tarifverbund_id", "ticket_code", "anzahl", "preis_eur", "umsatz_eur", "gueltig_ab", "plz"],
)
write_csv_for_excel(ticket_verkaeufe, OUTPUT_DIR / "ticket_verkaeufe.csv")

# 2) sonstige_angebote (was rest_angebot)
ticketgruppen = [100, 110, 120, 130, 200, 210]
REST_ROWS = 5000
rows = []
for _ in range(REST_ROWS):
    jahr = random.choice([2024, 2025])
    monat = random.randint(1, 12)
    meldestelle_code = random.choice(meldestelle_codes)
    tarifverbund_id = random.choice(tarifverbund_ids)
    angebot_gruppe = random.choice(ticketgruppen)
    umsatz_eur = round(random.uniform(500.0, 35000.0), 2)
    rows.append((monat, jahr, meldestelle_code, tarifverbund_id, angebot_gruppe, umsatz_eur))

sonstige_angebote = pd.DataFrame(
    rows,
    columns=["monat", "jahr", "meldestelle_code", "tarifverbund_id", "angebot_gruppe", "umsatz_eur"],
)
write_csv_for_excel(sonstige_angebote, OUTPUT_DIR / "sonstige_angebote.csv")

# 3) plan_umsatz (was solleinnahmen)
rows = []
for jahr in [2024, 2025]:
    for monat in range(1, 13):
        for tarifverbund_id in tarifverbund_ids:
            umsatz_eur = round(random.uniform(50000.0, 800000.0), 2)
            rows.append((monat, jahr, tarifverbund_id, umsatz_eur))

plan_umsatz = pd.DataFrame(rows, columns=["monat", "jahr", "tarifverbund_id", "umsatz_eur"])
write_csv_for_excel(plan_umsatz, OUTPUT_DIR / "plan_umsatz.csv")

print("Group 2 done. Output folder:", OUTPUT_DIR)
