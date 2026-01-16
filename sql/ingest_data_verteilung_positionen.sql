-- Ingest data for verteilung_positionen
CREATE OR REPLACE TABLE verteilung_positionen AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/verteilung_positionen.csv', HEADER=true);
