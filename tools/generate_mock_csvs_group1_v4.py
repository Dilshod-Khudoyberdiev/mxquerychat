"""
generate_mock_csvs_group1_v3.py

Group 1 (reference tables) with "friendly" non-customer naming.
Writes to: training_data/mock_csv_v3/

Tables:
- regionen_bundesland.csv
- ticket_produkte.csv
- tarifverbuende.csv
- meldestellen.csv
- postleitzahlen.csv
"""

import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)

OUTPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_START = datetime(2024, 1, 1)
AUDIT_END = datetime(2026, 1, 1)

def random_datetime(start: datetime, end: datetime) -> str:
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")

def maybe(value, null_probability=0.35):
    return pd.NA if random.random() < null_probability else value

def write_csv_for_excel(df: pd.DataFrame, path: Path) -> None:
    # UTF-8 with BOM so Excel shows umlauts correctly
    df.to_csv(path, index=False, encoding="utf-8-sig")

def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["erstellt_von_user_id"] = [random.randint(1, 80) for _ in range(len(df))]
    df["erstellt_am"] = [random_datetime(AUDIT_START, AUDIT_END) for _ in range(len(df))]

    geaendert_von = []
    geaendert_am = []
    for created in df["erstellt_am"]:
        if random.random() < 0.45:
            geaendert_von.append(pd.NA)
            geaendert_am.append(pd.NA)
        else:
            geaendert_von.append(random.randint(1, 80))
            created_dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
            geaendert_am.append(random_datetime(created_dt, AUDIT_END))

    df["geaendert_von_user_id"] = geaendert_von
    df["geaendert_am"] = geaendert_am
    return df

# 1) Regionen / Bundesländer
bundeslaender_rows = [
    (1,  "Schleswig-Holstein",       "DE-SH", "01"),
    (2,  "Hamburg",                  "DE-HH", "02"),
    (3,  "Niedersachsen",            "DE-NI", "03"),
    (4,  "Bremen",                   "DE-HB", "04"),
    (5,  "Nordrhein-Westfalen",      "DE-NW", "05"),
    (6,  "Hessen",                   "DE-HE", "06"),
    (7,  "Rheinland-Pfalz",          "DE-RP", "07"),
    (8,  "Baden-Württemberg",        "DE-BW", "08"),
    (9,  "Bayern",                   "DE-BY", "09"),
    (10, "Saarland",                 "DE-SL", "10"),
    (11, "Berlin",                   "DE-BE", "11"),
    (12, "Brandenburg",              "DE-BB", "12"),
    (13, "Mecklenburg-Vorpommern",   "DE-MV", "13"),
    (14, "Sachsen",                  "DE-SN", "14"),
    (15, "Sachsen-Anhalt",           "DE-ST", "15"),
    (16, "Thüringen",                "DE-TH", "16"),
]
regionen_bundesland = pd.DataFrame(
    bundeslaender_rows,
    columns=["bundesland_id", "bundesland_name", "iso_code", "bundesland_code2"],
)
regionen_bundesland = add_audit_columns(regionen_bundesland)

write_csv_for_excel(regionen_bundesland, OUTPUT_DIR / "regionen_bundesland.csv")

# 2) Ticket-Produkte (Deutschlandticket + Varianten)
ticket_rows = [
    (1, "Deutschlandticket",                 1, 49.00),
    (2, "Deutschlandticket (Preis 2025)",    2, 58.00),
    (3, "Deutschlandticket (Preis 2026)",    3, 63.00),
    (4, "Deutschlandticket Job",             4, 49.00),
    (5, "Deutschlandticket Sozial/Ermäßigt", 5, 39.00),
]
ticket_produkte = pd.DataFrame(
    ticket_rows, columns=["ticket_id", "ticket_name", "ticket_code", "preis_eur"]
)
ticket_produkte = add_audit_columns(ticket_produkte)

write_csv_for_excel(ticket_produkte, OUTPUT_DIR / "ticket_produkte.csv")

# 3) Tarifverbünde (frendlier name for tariforganisation)
tarifverbuende_names = [
    ("Hamburger Verkehrsverbund", "HVV"),
    ("Verkehrsverbund Berlin-Brandenburg", "VBB"),
    ("Rhein-Main-Verkehrsverbund", "RMV"),
    ("Verkehrsverbund Rhein-Neckar", "VRN"),
    ("Verkehrs- und Tarifverbund Stuttgart", "VVS"),
    ("Karlsruher Verkehrsverbund", "KVV"),
    ("Münchner Verkehrs- und Tarifverbund", "MVV"),
    ("Verkehrsverbund Großraum Nürnberg", "VGN"),
    ("Großraum-Verkehr Hannover", "GVH"),
    ("Verkehrsverbund Bremen/Niedersachsen", "VBN"),
    ("Verkehrsverbund Rhein-Ruhr", "VRR"),
    ("Verkehrsverbund Rhein-Sieg", "VRS"),
    ("Verkehrsverbund Oberelbe", "VVO"),
    ("Mitteldeutscher Verkehrsverbund", "MDV"),
    ("Verkehrsverbund Mittelsachsen", "VMS"),
    ("Aachener Verkehrsverbund", "AVV"),
    ("Saarländischer Verkehrsverbund", "saarVV"),
    ("Verkehrsverbund Region Trier", "VRT"),
    ("Bodensee-Oberschwaben Verkehrsverbund", "bodo"),
    ("Naldo Verkehrsverbund", "naldo"),
]
rows = []
for i, (name, short) in enumerate(tarifverbuende_names, start=1):
    status = "aktiv" if random.random() < 0.9 else "inaktiv"
    rows.append((i, name, short, status))

tarifverbuende = pd.DataFrame(rows, columns=["tarifverbund_id", "name", "kuerzel", "status"])
tarifverbuende["erstellt_am"] = [random_datetime(AUDIT_START, AUDIT_END) for _ in range(len(tarifverbuende))]
tarifverbuende["geaendert_am"] = [maybe(random_datetime(AUDIT_START, AUDIT_END), 0.5) for _ in range(len(tarifverbuende))]

write_csv_for_excel(tarifverbuende, OUTPUT_DIR / "tarifverbuende.csv")

# 4) Meldestellen (reporting offices)
cities = [
    ("Berlin", "BER"), ("Hamburg", "HAM"), ("München", "MUC"), ("Köln", "CGN"), ("Frankfurt am Main", "FRA"),
    ("Stuttgart", "STR"), ("Düsseldorf", "DUS"), ("Dortmund", "DTM"), ("Essen", "ESS"), ("Leipzig", "LEJ"),
    ("Bremen", "BRE"), ("Dresden", "DRS"), ("Hannover", "HAJ"), ("Nürnberg", "NUE"), ("Duisburg", "DUI"),
]
meldestellen_rows = []
meldestelle_nummer = 1000
organisation_ids = list(range(1, 51))  # synthetic "Organisationen" (not customer)

meldestelle_id = 1
for city, code in cities:
    for n in range(random.randint(3, 7)):
        meldestelle_nummer += 1
        meldestellen_rows.append(
            (
                meldestelle_id,
                meldestelle_nummer,
                f"Meldestelle {city} {n+1}",
                f"MS-{code}-{n+1:03d}",
                random.choice(organisation_ids),
            )
        )
        meldestelle_id += 1

meldestellen = pd.DataFrame(
    meldestellen_rows,
    columns=["meldestelle_id", "meldestelle_nummer", "meldestelle_name", "meldestelle_code", "organisation_id"],
)
meldestellen = add_audit_columns(meldestellen)

write_csv_for_excel(meldestellen, OUTPUT_DIR / "meldestellen.csv")

# 5) Postleitzahlen
state_plz_ranges = {
    "01": [(24000, 25999)],
    "02": [(20000, 22999)],
    "03": [(26000, 31999)],
    "04": [(27500, 28999)],
    "05": [(40000, 53999)],
    "06": [(34000, 36999), (60000, 65999)],
    "07": [(54000, 56999)],
    "08": [(68000, 79999), (88000, 89999)],
    "09": [(80000, 87999), (90000, 97999)],
    "10": [(66000, 66999)],
    "11": [(10000, 14999)],
    "12": [(15000, 19999)],
    "13": [(17000, 19999)],
    "14": [(1000, 9999)],
    "15": [(6000, 6999)],
    "16": [(7000, 9999)],
}

cities_by_state = {
    "01": ["Kiel", "Lübeck", "Flensburg", "Neumünster", "Elmshorn"],
    "02": ["Hamburg"],
    "03": ["Hannover", "Braunschweig", "Oldenburg", "Osnabrück", "Göttingen"],
    "04": ["Bremen", "Bremerhaven"],
    "05": ["Köln", "Düsseldorf", "Dortmund", "Essen", "Duisburg", "Bochum", "Bonn", "Münster", "Bielefeld"],
    "06": ["Frankfurt am Main", "Wiesbaden", "Kassel", "Darmstadt", "Offenbach am Main"],
    "07": ["Mainz", "Koblenz", "Trier", "Kaiserslautern", "Ludwigshafen"],
    "08": ["Stuttgart", "Karlsruhe", "Mannheim", "Freiburg im Breisgau", "Heidelberg", "Ulm"],
    "09": ["München", "Nürnberg", "Augsburg", "Regensburg", "Würzburg", "Ingolstadt"],
    "10": ["Saarbrücken", "Neunkirchen", "Völklingen", "Homburg"],
    "11": ["Berlin"],
    "12": ["Potsdam", "Cottbus", "Brandenburg an der Havel", "Frankfurt (Oder)"],
    "13": ["Rostock", "Schwerin", "Neubrandenburg", "Stralsund"],
    "14": ["Dresden", "Leipzig", "Chemnitz", "Zwickau", "Görlitz"],
    "15": ["Magdeburg", "Halle (Saale)", "Dessau-Roßlau", "Wittenberg"],
    "16": ["Erfurt", "Jena", "Weimar", "Gera"],
}

def generate_plz(state_code: str, count: int):
    ranges = state_plz_ranges[state_code]
    out = []
    for _ in range(count):
        start, end = random.choice(ranges)
        out.append(f"{random.randint(start, end):05d}")
    return out

plz_rows = []
plz_id = 1
for state_code in state_plz_ranges.keys():
    for plz_code in generate_plz(state_code, 25):  # ~400 rows
        ort = random.choice(cities_by_state[state_code])
        kreis_code = maybe(f"{random.randint(10000, 99999)}", 0.35)
        plz_rows.append((plz_id, plz_code, ort, kreis_code, state_code))
        plz_id += 1

postleitzahlen = pd.DataFrame(
    plz_rows,
    columns=["plz_id", "plz", "ort", "kreis_code", "bundesland_code2"],
)
postleitzahlen = add_audit_columns(postleitzahlen)

write_csv_for_excel(postleitzahlen, OUTPUT_DIR / "postleitzahlen.csv")

print(" Group 1 done. Output folder:", OUTPUT_DIR)
