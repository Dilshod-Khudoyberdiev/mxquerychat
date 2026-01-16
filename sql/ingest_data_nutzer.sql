-- Ingest data for nutzer
CREATE OR REPLACE TABLE nutzer AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/nutzer.csv', HEADER=true);
