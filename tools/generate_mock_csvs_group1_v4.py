"""
generate_mock_csvs_group1_v4.py

Group 1 = Reference tables (foundation)
Creates these CSVs in training_data/mock_csv_v3/:

- regionen_bundesland.csv
- postleitzahlen.csv
- ticket_produkte.csv
- tarifverbuende.csv
- meldestellen.csv

These tables are "lookup" tables used by other groups.
No customer schema. No personal data.

Run:
  python tools/generate_mock_csvs_group1_v4.py
"""

import random
from pathlib import Path
from datetime import datetime

import pandas as pd

# -----------------------
# Settings
# -----------------------
random.seed(42)

OUTPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Helper: write CSV in Excel-friendly encoding
# -----------------------
def write_csv_for_excel(df: pd.DataFrame, path: Path) -> None:
    """Write UTF-8 with BOM so Excel shows umlauts correctly."""
    df.to_csv(path, index=False, encoding="utf-8-sig")


# -----------------------
# 1) regionen_bundesland.csv
# -----------------------
bundeslaender = [
    (1, "Baden-Württemberg", "DE-BW", "08"),
    (2, "Bayern", "DE-BY", "09"),
    (3, "Berlin", "DE-BE", "11"),
    (4, "Brandenburg", "DE-BB", "12"),
    (5, "Bremen", "DE-HB", "04"),
    (6, "Hamburg", "DE-HH", "02"),
    (7, "Hessen", "DE-HE", "06"),
    (8, "Mecklenburg-Vorpommern", "DE-MV", "13"),
    (9, "Niedersachsen", "DE-NI", "03"),
    (10, "Nordrhein-Westfalen", "DE-NW", "05"),
    (11, "Rheinland-Pfalz", "DE-RP", "07"),
    (12, "Saarland", "DE-SL", "10"),
    (13, "Sachsen", "DE-SN", "14"),
    (14, "Sachsen-Anhalt", "DE-ST", "15"),
    (15, "Schleswig-Holstein", "DE-SH", "01"),
    (16, "Thüringen", "DE-TH", "16"),
]

regionen_bundesland_df = pd.DataFrame(
    bundeslaender,
    columns=["bundesland_id", "bundesland_name", "iso_code", "bundesland_code2"],
)

write_csv_for_excel(regionen_bundesland_df, OUTPUT_DIR / "regionen_bundesland.csv")


# -----------------------
# 2) postleitzahlen.csv
# -----------------------
# We generate synthetic but realistic-looking PLZ entries.
# We do NOT try to demonstrate official full PLZ coverage. This is mock data.
cities = [
    ("10115", "Berlin", "11"),
    ("20095", "Hamburg", "02"),
    ("80331", "München", "09"),
    ("50667", "Köln", "05"),
    ("60311", "Frankfurt am Main", "06"),
    ("70173", "Stuttgart", "08"),
    ("40213", "Düsseldorf", "05"),
    ("04109", "Leipzig", "14"),
    ("01067", "Dresden", "14"),
    ("30159", "Hannover", "03"),
    ("28195", "Bremen", "04"),
    ("24103", "Kiel", "01"),
    ("66111", "Saarbrücken", "10"),
    ("99084", "Erfurt", "16"),
    ("39104", "Magdeburg", "15"),
    ("14467", "Potsdam", "12"),
    ("19053", "Schwerin", "13"),
    ("56068", "Koblenz", "07"),
    ("55116", "Mainz", "07"),
    ("97070", "Würzburg", "09"),
]

# Expand the list to create more PLZ rows (still synthetic).
postleitzahlen_rows = []
plz_id = 1

for base_plz, ort, bl_code2 in cities:
    # Create variations around the base PLZ (synthetic).
    base_int = int(base_plz)
    for i in range(0, 25):  # 25 variants per city -> about 500 rows total
        plz_value = str(base_int + i).zfill(5)
        kreis_code = f"K{bl_code2}{random.randint(100,999)}"
        postleitzahlen_rows.append((plz_id, plz_value, ort, kreis_code, bl_code2))
        plz_id += 1

postleitzahlen_df = pd.DataFrame(
    postleitzahlen_rows,
    columns=["plz_id", "plz", "ort", "kreis_code", "bundesland_code2"],
)

write_csv_for_excel(postleitzahlen_df, OUTPUT_DIR / "postleitzahlen.csv")


# -----------------------
# 3) ticket_produkte.csv
# -----------------------
ticket_produkte_rows = [
    (1, "Deutschlandticket", 49, 49.00),
    (2, "Deutschlandticket (Ermäßigt)", 48, 39.00),
    (3, "Monatskarte", 20, 79.00),
    (4, "Wochenkarte", 21, 29.00),
    (5, "Tageskarte", 10, 9.90),
    (6, "Kurzstrecke", 11, 2.50),
    (7, "Einzelticket", 12, 3.40),
    (8, "Gruppenticket", 13, 12.50),
]

ticket_produkte_df = pd.DataFrame(
    ticket_produkte_rows,
    columns=["ticket_id", "ticket_name", "ticket_code", "preis_eur"],
)

write_csv_for_excel(ticket_produkte_df, OUTPUT_DIR / "ticket_produkte.csv")


# -----------------------
# 4) tarifverbuende.csv
# -----------------------
tarif_names = [
    ("VBB", "Verkehrsverbund Berlin-Brandenburg", "aktiv"),
    ("HVV", "Hamburger Verkehrsverbund", "aktiv"),
    ("MVV", "Münchner Verkehrs- und Tarifverbund", "aktiv"),
    ("VRS", "Verkehrsverbund Rhein-Sieg", "aktiv"),
    ("RMV", "Rhein-Main-Verkehrsverbund", "aktiv"),
    ("VVS", "Verkehrs- und Tarifverbund Stuttgart", "aktiv"),
    ("VRR", "Verkehrsverbund Rhein-Ruhr", "aktiv"),
    ("GVH", "Großraum-Verkehr Hannover", "aktiv"),
    ("VGN", "Verkehrsverbund Großraum Nürnberg", "aktiv"),
    ("MDV", "Mitteldeutscher Verkehrsverbund", "aktiv"),
    ("VVO", "Verkehrsverbund Oberelbe", "aktiv"),
    ("SH-Tarif", "Nah.SH (Schleswig-Holstein)", "aktiv"),
    ("AVV", "Aachener Verkehrsverbund", "aktiv"),
    ("VRM", "Verkehrsverbund Rhein-Mosel", "aktiv"),
    ("VRT", "Verkehrsverbund Region Trier", "inaktiv"),
    ("HNV", "Heilbronner-Hohenloher-Haller Nahverkehr", "aktiv"),
]

tarifverbuende_rows = []
created_at = datetime(2024, 1, 1).strftime("%Y-%m-%d")

for idx, (kuerzel, name, status) in enumerate(tarif_names, start=1):
    tarifverbuende_rows.append((idx, name, kuerzel, status, created_at, created_at))

tarifverbuende_df = pd.DataFrame(
    tarifverbuende_rows,
    columns=["tarifverbund_id", "name", "kuerzel", "status", "erstellt_am", "geaendert_am"],
)

write_csv_for_excel(tarifverbuende_df, OUTPUT_DIR / "tarifverbuende.csv")


# -----------------------
# 5) meldestellen.csv
# -----------------------
# Meldestellen represent reporting offices.
# We generate synthetic reporting offices linked to a synthetic organisation_id.
meldestellen_rows = []
meldestelle_id = 1

# We pick some city labels from postleitzahlen table for realism.
city_names = sorted(list(set(postleitzahlen_df["ort"].tolist())))

for city in city_names:
    # Create multiple offices per city
    for i in range(1, 7):  # 6 offices per city
        meldestelle_nummer = random.randint(1000, 9999)
        meldestelle_name = f"Meldestelle {city} {i}"
        meldestelle_code = f"MS-{city[:3].upper()}-{meldestelle_nummer}"
        organisation_id = random.randint(1, 25)

        meldestellen_rows.append(
            (meldestelle_id, meldestelle_nummer, meldestelle_name, meldestelle_code, organisation_id)
        )
        meldestelle_id += 1

meldestellen_df = pd.DataFrame(
    meldestellen_rows,
    columns=["meldestelle_id", "meldestelle_nummer", "meldestelle_name", "meldestelle_code", "organisation_id"],
)

write_csv_for_excel(meldestellen_df, OUTPUT_DIR / "meldestellen.csv")


print("✅ Group 1 done. Generated reference tables in:", OUTPUT_DIR)
print(" - regionen_bundesland.csv")
print(" - postleitzahlen.csv")
print(" - ticket_produkte.csv")
print(" - tarifverbuende.csv")
print(" - meldestellen.csv")
