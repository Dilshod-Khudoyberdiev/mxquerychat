-- Ingest data for postleitzahlen
CREATE OR REPLACE TABLE postleitzahlen AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/postleitzahlen.csv', HEADER=true);
