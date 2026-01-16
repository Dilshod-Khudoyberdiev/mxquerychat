-- Ingest data for plan_umsatz
CREATE OR REPLACE TABLE plan_umsatz AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/plan_umsatz.csv', HEADER=true);
