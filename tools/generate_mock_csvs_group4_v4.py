"""
generate_mock_csvs_group4_v4.py

Group 4 = Users + permissions (optional demo domain)
Creates these CSVs in training_data/mock_csv_v3/:

- nutzer.csv
- nutzer_bundesland_rechte.csv
- nutzer_tarifverbund_rechte.csv
- nutzer_zast_rechte.csv
- nutzer_zentral_zast_rechte.csv

Depends on Group 1 tables.
Run:
  python tools/generate_mock_csvs_group4_v4.py
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)

INPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR = Path("training_data/mock_csv_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def write_csv_for_excel(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")

# Load reference tables
bundesland_df = pd.read_csv(INPUT_DIR / "regionen_bundesland.csv")
tarif_df = pd.read_csv(INPUT_DIR / "tarifverbuende.csv")

bundesland_ids = bundesland_df["bundesland_id"].astype(int).tolist()
tarif_ids = tarif_df["tarifverbund_id"].astype(int).tolist()

# -----------------------
# 1) nutzer.csv
# -----------------------
first_names = ["Anna", "Ben", "Clara", "David", "Emilia", "Felix", "Greta", "Hannes", "Isabel", "Jonas"]
last_names = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Hoffmann", "Klein"]

NUTZER_COUNT = 80
nutzer_rows = []

for nutzer_id in range(1, NUTZER_COUNT + 1):
    fn = random.choice(first_names)
    ln = random.choice(last_names)
    benutzername = f"{fn.lower()}.{ln.lower()}{nutzer_id}"
    email = f"user{nutzer_id}@example.org"

    nutzer_rows.append((nutzer_id, benutzername, fn, ln, email))

nutzer_df = pd.DataFrame(
    nutzer_rows,
    columns=["nutzer_id", "benutzername", "vorname", "nachname", "email"],
)
write_csv_for_excel(nutzer_df, OUTPUT_DIR / "nutzer.csv")

# -----------------------
# 2) nutzer_bundesland_rechte.csv
# -----------------------
nb_rows = []
rechte_id = 1

for nutzer_id in nutzer_df["nutzer_id"].astype(int).tolist():
    for bl_id in random.sample(bundesland_ids, k=random.randint(1, 4)):
        nb_rows.append((
            rechte_id,
            nutzer_id,
            bl_id,
            random.choice([True, False]),  # darf_upload
            random.choice([True, False]),  # darf_download
            random.choice([True, False]),  # darf_historie_sehen
        ))
        rechte_id += 1

nutzer_bundesland_rechte_df = pd.DataFrame(
    nb_rows,
    columns=["rechte_id", "nutzer_id", "bundesland_id", "darf_upload", "darf_download", "darf_historie_sehen"],
)
write_csv_for_excel(nutzer_bundesland_rechte_df, OUTPUT_DIR / "nutzer_bundesland_rechte.csv")

# -----------------------
# 3) nutzer_tarifverbund_rechte.csv
# -----------------------
nt_rows = []
rechte_id = 1

for nutzer_id in nutzer_df["nutzer_id"].astype(int).tolist():
    for tarifverbund_id in random.sample(tarif_ids, k=random.randint(1, 3)):
        nt_rows.append((
            rechte_id,
            nutzer_id,
            tarifverbund_id,
            random.choice([True, False]),  # darf_upload
            random.choice([True, False]),  # darf_download
            random.choice([True, False]),  # darf_historie_sehen
        ))
        rechte_id += 1

nutzer_tarifverbund_rechte_df = pd.DataFrame(
    nt_rows,
    columns=["rechte_id", "nutzer_id", "tarifverbund_id", "darf_upload", "darf_download", "darf_historie_sehen"],
)
write_csv_for_excel(nutzer_tarifverbund_rechte_df, OUTPUT_DIR / "nutzer_tarifverbund_rechte.csv")

# -----------------------
# 4) nutzer_zast_rechte.csv
# -----------------------
zast_ids = [f"ZAST-{i:03d}" for i in range(1, 21)]

nz_rows = []
rechte_id = 1

for nutzer_id in nutzer_df["nutzer_id"].astype(int).tolist():
    for zast_id in random.sample(zast_ids, k=random.randint(0, 2)):
        nz_rows.append((
            rechte_id,
            nutzer_id,
            zast_id,
            random.choice([True, False]),  # darf_upload
            random.choice([True, False]),  # darf_download
            random.choice([True, False]),  # darf_ansehen
        ))
        rechte_id += 1

nutzer_zast_rechte_df = pd.DataFrame(
    nz_rows,
    columns=["rechte_id", "nutzer_id", "zast_id", "darf_upload", "darf_download", "darf_ansehen"],
)
write_csv_for_excel(nutzer_zast_rechte_df, OUTPUT_DIR / "nutzer_zast_rechte.csv")

# -----------------------
# 5) nutzer_zentral_zast_rechte.csv
# -----------------------
zentral_ids = [f"ZZAST-{i:03d}" for i in range(1, 11)]

nzz_rows = []
rechte_id = 1

for nutzer_id in nutzer_df["nutzer_id"].astype(int).tolist():
    for zentral_id in random.sample(zentral_ids, k=random.randint(0, 1)):
        nzz_rows.append((
            rechte_id,
            nutzer_id,
            zentral_id,
            random.choice([True, False]),  # darf_upload
            random.choice([True, False]),  # darf_download
            random.choice([True, False]),  # darf_ansehen
        ))
        rechte_id += 1

nutzer_zentral_zast_rechte_df = pd.DataFrame(
    nzz_rows,
    columns=["rechte_id", "nutzer_id", "zentral_zast_id", "darf_upload", "darf_download", "darf_ansehen"],
)
write_csv_for_excel(nutzer_zentral_zast_rechte_df, OUTPUT_DIR / "nutzer_zentral_zast_rechte.csv")

print("✅ Group 4 done. Generated user + permission tables in:", OUTPUT_DIR)
print(" - nutzer.csv")
print(" - nutzer_bundesland_rechte.csv")
print(" - nutzer_tarifverbund_rechte.csv")
print(" - nutzer_zast_rechte.csv")
print(" - nutzer_zentral_zast_rechte.csv")
