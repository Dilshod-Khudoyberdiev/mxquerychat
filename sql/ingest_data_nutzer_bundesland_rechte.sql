-- Ingest data for nutzer_bundesland_rechte
CREATE OR REPLACE TABLE nutzer_bundesland_rechte AS
SELECT *
FROM read_csv_auto('training_data/mock_csv_v3/nutzer_bundesland_rechte.csv', HEADER=true);
