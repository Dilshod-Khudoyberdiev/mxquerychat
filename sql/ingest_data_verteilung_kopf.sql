-- Ingest data for verteilung_kopf
CREATE OR REPLACE TABLE verteilung_kopf AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/verteilung_kopf.csv', HEADER=true);
